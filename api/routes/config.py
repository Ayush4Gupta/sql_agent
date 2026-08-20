import logging

from fastapi import APIRouter

from sql_agent.config.settings import AGENT_PROFILE, DB_BACKEND, LLM_PROVIDER

router = APIRouter(tags=["config"])
logger = logging.getLogger(__name__)


@router.get("/api/config")
async def get_config():
    """Return agent configuration for the frontend header badge."""
    return {
        "profile": AGENT_PROFILE,
        "db_backend": DB_BACKEND,
        "llm_provider": LLM_PROVIDER,
    }
