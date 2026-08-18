import logging
import re
from pathlib import Path

from langchain_community.utilities import SQLDatabase

from sql_agent.config.settings import (
    DB_PATH, DB_BACKEND,
    SYNAPSE_SERVER, SYNAPSE_DATABASE, SYNAPSE_AUTH, SYNAPSE_SCHEMA,
    CLIENT_ID, CLIENT_SECRET, TENANT_ID,
)

logger = logging.getLogger(__name__)

# Any SQL statement matching one of these patterns is blocked.
# Catches: plain DML, CTEs ending in DML (WITH x AS (...) DELETE ...),
# MERGE, BULK INSERT, stored-proc execution, and dynamic SQL.
_WRITE_PATTERN = re.compile(
    r"""
    \b(
        INSERT\s              |
        UPDATE\s              |
        DELETE\s              |
        DROP\s                |
        CREATE\s              |
        ALTER\s               |
        TRUNCATE\s            |
        MERGE\s               |
        BULK\s+INSERT\s       |
        EXEC\s                |
        EXECUTE\s             |
        sp_executesql\s*\(    |
        xp_\w+
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Strip single-line (--) and block (/* */) SQL comments before checking.
_COMMENT_PATTERN = re.compile(r"--[^\n]*|/\*.*?\*/", re.DOTALL)

def _assert_read_only(statement: str) -> None:
    clean = _COMMENT_PATTERN.sub(" ", statement)
    if _WRITE_PATTERN.search(clean):
        # Log internally (never expose the SQL to the caller/user)
        logger.warning(
            "[db] Blocked write operation — statement preview: %s", statement[:200]
        )
        raise PermissionError("Write operations are not permitted.")


def get_db() -> SQLDatabase:
    if DB_BACKEND == "synapse":
        return _get_synapse_db()
    elif DB_BACKEND == "duckdb":
        return _get_duckdb_db()
    return _get_sqlite_db()

def _get_sqlite_db() -> SQLDatabase:
    if not DB_PATH.exists():
        logger.error("[db] SQLite database file not found at path: %s", DB_PATH)
        raise ConnectionError("Database is currently unavailable.")
    logger.info("Connecting to SQLite (read-only): %s", DB_PATH)
    # mode=ro enforces read-only at the driver level — no writes possible.
    uri = f"sqlite:///file:{DB_PATH}?mode=ro&uri=true"
    db = SQLDatabase.from_uri(uri)
    logger.info("SQLite connected — dialect: %s, tables: %s", db.dialect, db.get_usable_table_names())
    return db


def _get_synapse_db() -> SQLDatabase:
    from sqlalchemy import create_engine, event
    from sqlalchemy.engine import URL

    if not SYNAPSE_SERVER or not SYNAPSE_DATABASE:
        logger.error("[db] Missing required Synapse configuration (SYNAPSE_SERVER / SYNAPSE_DATABASE)")
        raise ConnectionError("Database is currently unavailable.")

    logger.info(
        "Connecting to Azure Synapse — server: %s, database: %s, schema: %s",
        SYNAPSE_SERVER, SYNAPSE_DATABASE, SYNAPSE_SCHEMA,
    )

    # Build the ODBC connection string.
    # ActiveDirectoryInteractive will open a browser/MFA prompt on first connection.
    conn_parts = [
        "Driver={ODBC Driver 18 for SQL Server}",
        f"Server=tcp:{SYNAPSE_SERVER}",
        f"Database={SYNAPSE_DATABASE}",
        f"Authentication={SYNAPSE_AUTH}",
        "Encrypt=yes",
        "TrustServerCertificate=no",
        "Connection Timeout=120",
    ]
    # Add user credentials if present
    if CLIENT_ID:
        conn_parts.append(f"UID={CLIENT_ID}")
    if CLIENT_SECRET:
        conn_parts.append(f"PWD={CLIENT_SECRET}")
    if TENANT_ID:
        conn_parts.append(f"Authority Id={TENANT_ID}")

    conn_str = ";".join(conn_parts) + ";"
    connection_url = URL.create("mssql+pyodbc", query={"odbc_connect": conn_str})
    engine = create_engine(
    connection_url,
    fast_executemany=True,
    isolation_level="AUTOCOMMIT",
    )
    # Enforce read-only at the SQLAlchemy event level (defence-in-depth).
    # THE STRONGEST PROTECTION is a database-level SELECT-only user on Synapse:
    #   EXEC sp_addrolemember 'db_datareader', '<your_service_account>';
    # That makes writes impossible regardless of application code.
    @event.listens_for(engine, "before_cursor_execute")
    def _block_writes(conn, cursor, statement, parameters, context, executemany):
        _assert_read_only(statement)

    db = SQLDatabase(engine, schema=SYNAPSE_SCHEMA)
    logger.info("Synapse connected — dialect: %s, tables: %s", db.dialect, db.get_usable_table_names())
    return db


def _get_duckdb_db() -> SQLDatabase:
    """Connect to a DuckDB file in read-only mode."""
    from sqlalchemy import create_engine, event

    if not DB_PATH.exists():
        logger.error("[db] DuckDB database file not found at path: %s", DB_PATH)
        raise ConnectionError("Database is currently unavailable.")

    logger.info("Connecting to DuckDB (read-only): %s", DB_PATH)
    # read_only must be passed via connect_args (not URI params),
    # because duckdb-engine converts URI params to SET statements
    # and read_only is a connection-level flag, not a runtime config.
    uri = f"duckdb:///{DB_PATH}"
    engine = create_engine(
        uri,
        connect_args={"read_only": True},
    )

    # Defense-in-depth: block writes at the SQLAlchemy event level too
    @event.listens_for(engine, "before_cursor_execute")
    def _block_writes(conn, cursor, statement, parameters, context, executemany):
        _assert_read_only(statement)

    db = SQLDatabase(engine)
    logger.info("DuckDB connected — dialect: %s, tables: %s", db.dialect, db.get_usable_table_names())
    return db