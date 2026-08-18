import logging

from fastapi import APIRouter

from api.dependencies import db

router = APIRouter(tags=["health"])
logger = logging.getLogger(__name__)


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "agent": "initialized",
        "database": db.dialect,
        "tables": db.get_usable_table_names(),
    }
