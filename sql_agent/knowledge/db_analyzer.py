import logging
import re
from collections import defaultdict
from langchain_community.utilities import SQLDatabase

logger = logging.getLogger(__name__)

# Column name suffixes that identify keys, hashes, dates — never categorical filter values.
# Uses endswith() — exact suffix match, no accidental substring blocking.
SKIP_SUFFIXES = ("sk", "bk", "sourceid", "hash", "number", "date", "ltz", "est", "contactid")

# Columns whose names start with this prefix are datawarehouse metadata — skip always.
SKIP_PREFIX = "dw"

# Cardinality window: only profile columns with distinct count in [MIN, MAX].
# Below MIN = constant value (useless). Above MAX = too many to enumerate in prompt.
MIN_PROFILE_CARDINALITY = 2
MAX_PROFILE_CARDINALITY = 30

# String column types to profile (SQLite: TEXT; MSSQL: NVARCHAR/VARCHAR/etc.)
_STRING_TYPES = {"TEXT", "NVARCHAR", "VARCHAR", "CHAR", "NCHAR", "NTEXT"}

# Regex: matches SQLite ("TableName"), MSSQL ([schema].[TableName]),
# and Synapse Serverless (curated.[TableName]) DDL headers
_RE_TABLE = re.compile(
    r'CREATE TABLE (?:(?:\[[\w ]+\]|[\w]+)\.)?(?:\[([^\]]+)\]|"([^"]+)"|([\w]+))',
    re.IGNORECASE,
)
# Regex: matches "[ColName] TYPE" (MSSQL), '"ColName" TYPE' (SQLite),
# and bare 'col_name TYPE' (DuckDB / PostgreSQL)
_RE_COLUMN = re.compile(r'\s+(?:\[([^\]]+)\]|"([^"]+)"|(\w+))\s+\[?(\w+)')


def _parse_ddl(info: str) -> dict[str, list[tuple[str, str]]]:
    """Return {table: [(col_name, col_type), ...]} parsed from LangChain's get_table_info()."""
    table_col_map: dict[str, list[tuple[str, str]]] = {}
    current_table = None
    for line in info.splitlines():
        m = _RE_TABLE.match(line)
        if m:
            current_table = m.group(1) or m.group(2) or m.group(3)
            table_col_map[current_table] = []
        elif current_table:
            cm = _RE_COLUMN.match(line)
            if cm:
                col_name = cm.group(1) or cm.group(2) or cm.group(3)
                col_type = cm.group(4)
                table_col_map[current_table].append((col_name, col_type))
    return table_col_map


def _quote(name: str, is_mssql: bool) -> str:
    return f"[{name}]" if is_mssql else f'"{name}"'


def _table_ref(table: str, db: SQLDatabase) -> str:
    """Return schema-qualified table reference for the current dialect."""
    is_mssql = db.dialect == "mssql"
    schema = getattr(db, "_schema", None)
    q = _quote(table, is_mssql)
    if is_mssql and schema:
        return f"[{schema}].{q}"
    return q


def build_db_context(db: SQLDatabase) -> str:
    # Fetch DDL once — passed to both layers to avoid duplicate Synapse round-trips.
    info = db.get_table_info()
    logger.debug("[db_analyzer] Raw DDL (first 500 chars): %s", info[:500])
    logger.info("[db_analyzer] get_table_info: %d chars, usable tables: %s",
                len(info), db.get_usable_table_names()[:3])

    relationships = _discover_relationships(db, info)
    profiles = _profile_columns(db, info)

    parts = []
    if relationships:
        parts.append("TABLE RELATIONSHIPS (auto-discovered join paths):")
        for rel in relationships[:15]:
            parts.append(f"  {rel}")
    if profiles:
        parts.append("\nCOLUMN VALUE PROFILES (sample distinct values):")
        for line in profiles[:15]:
            parts.append(f"  {line}")

    result = "\n".join(parts)
    logger.info("[db_analyzer] db_context built: %d relationships, %d profile lines (capped)",
                min(len(relationships), 15), min(len(profiles), 15))
    return result


def _discover_relationships(db: SQLDatabase, info: str) -> list[str]:
    """Layer 1: Find SK/BK columns shared across tables = join keys."""
    table_col_map = _parse_ddl(info)

    col_to_tables: dict[str, list[str]] = defaultdict(list)
    for table, cols in table_col_map.items():
        for col_name, _ in cols:
            col_lower = col_name.lower()
            if col_lower.endswith("sk") or col_lower.endswith("bk"):
                col_to_tables[col_name].append(table)

    relationships = []
    for col, tables in col_to_tables.items():
        if len(tables) >= 2:
            for i in range(len(tables)):
                for j in range(i + 1, len(tables)):
                    relationships.append(f"{tables[i]}.{col} = {tables[j]}.{col}")
    return relationships


def _profile_columns(db: SQLDatabase, info: str) -> list[str]:
    """Layer 2: Get distinct values for low-cardinality string columns.

    Selection rules (no keyword lists — robust to new tables/columns):
    1. Skip if column type is not a string type.
    2. Skip if column name starts with SKIP_PREFIX ('dw') — metadata columns.
    3. Skip if column name ends with any SKIP_SUFFIX (sk, bk, sourceid, etc.) — keys/hashes/dates.
    4. Fetch TOP(MAX+1)/LIMIT(MAX+1) distinct values — if row count outside
       [MIN_PROFILE_CARDINALITY, MAX_PROFILE_CARDINALITY] skip; otherwise use result directly.
       This avoids a separate COUNT query per column.
    """
    table_col_map = _parse_ddl(info)

    is_mssql = db.dialect == "mssql"
    limit = MAX_PROFILE_CARDINALITY + 1  # fetch one extra to detect over-limit
    profile_lines = []

    for table, cols in table_col_map.items():
        tref = _table_ref(table, db)
        for col_name, col_type in cols:
            # Rule 1: string types only
            if col_type.upper() not in _STRING_TYPES:
                continue
            col_lower = col_name.lower()
            # Rule 2: skip datawarehouse metadata prefix
            if col_lower.startswith(SKIP_PREFIX):
                continue
            # Rule 3: skip key/hash/date suffixes
            if any(col_lower.endswith(s) for s in SKIP_SUFFIXES):
                continue
            try:
                cq = _quote(col_name, is_mssql)
                # Rule 4+5: one query — fetch MAX+1 distinct values ordered.
                # Count returned rows in Python to gate cardinality.
                if is_mssql:
                    sql = f"SELECT DISTINCT TOP {limit} {cq} FROM {tref} WHERE {cq} IS NOT NULL ORDER BY {cq}"
                else:
                    sql = f"SELECT DISTINCT {cq} FROM {tref} WHERE {cq} IS NOT NULL ORDER BY {cq} LIMIT {limit}"
                result = db.run(sql)
                if not result or result == "[]":
                    continue
                # Count rows: LangChain returns "[('val1',), ('val2',), ...]"
                rows = [r for r in result.strip("[]").split("), (") if r.strip()]
                row_count = len(rows)
                if not (MIN_PROFILE_CARDINALITY <= row_count <= MAX_PROFILE_CARDINALITY):
                    continue
                profile_lines.append(f"{table}.{col_name}: {result}")
            except Exception as e:
                logger.debug("[db_analyzer] Skipped profile for %s.%s: %s", table, col_name, e)

    return profile_lines