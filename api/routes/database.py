import logging

from fastapi import APIRouter, HTTPException

from api.dependencies import db
from sql_agent.utils.error_sanitizer import sanitize_error

router = APIRouter(prefix="/database", tags=["database"])
logger = logging.getLogger(__name__)


@router.get("/tables")
async def get_tables():
    """Return all usable table names and their count."""
    try:
        tables = db.get_usable_table_names()
        return {"tables": tables, "count": len(tables)}
    except Exception as e:
        safe_message = sanitize_error(e, context="GET /database/tables")
        raise HTTPException(status_code=500, detail=safe_message)


@router.get("/schema/{table_name}")
async def get_table_schema(table_name: str):
    """Return the CREATE TABLE statement and sample rows for a given table."""
    try:
        tables = db.get_usable_table_names()
        if table_name not in tables:
            raise HTTPException(
                status_code=404, detail="The requested table was not found."
            )
        schema = db.get_table_info_no_throw([table_name])
        return {"table": table_name, "schema": schema}
    except HTTPException:
        raise
    except Exception as e:
        safe_message = sanitize_error(e, context=f"GET /database/schema/{table_name}")
        raise HTTPException(status_code=500, detail=safe_message)
