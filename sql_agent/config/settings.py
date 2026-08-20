import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ── Active profile ────────────────────────────────────────────────────────────
AGENT_PROFILE = os.getenv("AGENT_PROFILE", "client")

# ── Profile-aware database defaults ──────────────────────────────────────────
# The active profile's config.yaml provides db_backend and db_path.
# Explicit env vars (DB_BACKEND, DB_PATH) override the profile values,
# so local dev flexibility is preserved while AGENT_PROFILE alone is
# sufficient to fully switch database + rules + examples.
from sql_agent.config.profile_loader import load_profile_config  # noqa: E402

_profile_config = load_profile_config()

DB_DIR = BASE_DIR / "database"

# Profile db_path is a filename (e.g. "tpcds_sf1.duckdb"), resolved relative
# to the database directory.  An explicit DB_PATH env var overrides entirely.
_profile_db_path = _profile_config.get("db_path", "")
_profile_db_backend = _profile_config.get("db_backend", "")

DB_BACKEND = os.environ.get("DB_BACKEND") or _profile_db_backend or "sqlite"
DB_PATH = Path(
    os.environ.get("DB_PATH")
    or (str(DB_DIR / _profile_db_path) if _profile_db_path else str(DB_DIR / "data.db"))
)

logger = logging.getLogger(__name__)
logger.info(
    "[settings] profile=%s → DB_BACKEND=%s, DB_PATH=%s",
    AGENT_PROFILE, DB_BACKEND, DB_PATH,
)

# ── LLM provider ──────────────────────────────────────────────────────────────
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "azure").lower().strip()

# ── Azure OpenAI ──────────────────────────────────────────────────────────────
AZURE_OPENAI_API_KEY      = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_ENDPOINT     = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_DEPLOYMENT   = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.2")
AZURE_OPENAI_API_VERSION  = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
AZURE_OPENAI_TEMPERATURE  = float(os.getenv("AZURE_OPENAI_TEMPERATURE", "0"))

# ── Database backend ──────────────────────────────────────────────────────────
ROW_LIMIT = int(os.getenv("ROW_LIMIT", "20"))

# Azure Synapse / SQL Server (used only when DB_BACKEND=synapse)
SYNAPSE_SERVER   = os.getenv("SYNAPSE_SERVER", "")
SYNAPSE_DATABASE = os.getenv("SYNAPSE_DATABASE", "")
SYNAPSE_AUTH     = os.getenv("SYNAPSE_AUTH", "ActiveDirectoryInteractive")
SYNAPSE_SCHEMA   = os.getenv("SYNAPSE_SCHEMA", "curated")

# Azure AD credentials for Synapse
CLIENT_ID = os.getenv("CLIENT_ID", "")
CLIENT_SECRET = os.getenv("CLIENT_SECRET", "")
TENANT_ID = os.getenv("TENANT_ID", "")

MAX_SCHEMA_ITERATIONS = int(os.getenv("MAX_SCHEMA_ITERATIONS", "3"))  # initial fetch + 2 loop re-fetches maximum

LOG_LEVEL = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)
