import logging

from fastapi import APIRouter, HTTPException

from api.dependencies import agent
from api.schemas import QueryRequest, QueryResponse, TableData

from sql_agent.config.settings import ROW_LIMIT

from sql_agent.utils.error_sanitizer import sanitize_error

router = APIRouter(tags=["query"])
logger = logging.getLogger(__name__)


@router.post("/query", response_model=QueryResponse)
async def execute_query(request: QueryRequest):
    """
    Execute a natural language question against the SQL database.

    The agent will:
    1. List available tables
    2. Fetch relevant schema information (iteratively if needed)
    3. Analyse the schema and decide which tables are sufficient
    4. Generate and validate a SQL query
    5. Execute the query and return a natural-language answer + tabular data

    Row limit is configured via ROW_LIMIT in .env (default 20).
    """
    logger.info(
        "POST /query — question: %s | row_limit: %d",
        request.question,
        ROW_LIMIT,  # ← Changed from request.row_limit
            "node_timings":      {},
    )
    try:
        inputs = {
            "messages":         [{"role": "user", "content": request.question}],
            "schema_iterations": 0,
            "retrieved_tables":  [],
            "raw_results":       {},       # ← NEW: will be populated by run_query node
            "row_limit":         ROW_LIMIT,   # ← NEW: carried through graph state
            "node_timings":      {},
        }

        messages      = []
        final_answer  = ""
        step_num      = 0
        final_state   = None              # ← NEW: keep reference to last graph state

        for step in agent.stream(inputs, stream_mode="values"):
            step_num   += 1
            final_state = step            # ← always update so we have the final state

            last_message = step["messages"][-1]
            msg_type     = getattr(last_message, "type", "unknown")
            logger.debug(
                "  Stream step %d — type: %s, tool_calls: %s",
                step_num,
                msg_type,
                bool(getattr(last_message, "tool_calls", None)),
            )

            message_data = {
                "role":    last_message.type    if hasattr(last_message, "type")    else "unknown",
                "content": last_message.content if hasattr(last_message, "content") else "",
            }

            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                message_data["tool_calls"] = last_message.tool_calls

            messages.append(message_data)

            if (
                hasattr(last_message, "type")
                and last_message.type == "ai"
                and last_message.content
            ):
                final_answer = last_message.content

        # ── Extract structured table data from final state ─────────────────────
        table_data: TableData | None = None
        if final_state:
            raw = final_state.get("raw_results") or {}
            cols = raw.get("columns") or []
            rows = raw.get("rows") or []
            if cols and rows:
                table_data = TableData(columns=cols, rows=rows)
                logger.info(
                    "POST /query — table_data: %d rows × %d cols",
                    len(rows), len(cols),
                )
            else:
                logger.info("POST /query — no tabular data captured (aggregation or error)")

        logger.info(
            "POST /query complete — %d stream steps, answer length: %d chars",
            step_num,
            len(final_answer),
        )
        # Extract node timings from final state
        node_timings = None
        if final_state:
            timings = final_state.get("node_timings") or {}
            if timings:
                node_timings = timings
                logger.info("POST /query node_timings: %s", timings)

        return QueryResponse(
            question=request.question,
            messages=messages,
            final_answer=final_answer,
            table_data=table_data,
            node_timings=node_timings,
        )

    # except Exception as e:
    #     logger.exception("POST /query failed for question: %s", request.question)
    #     raise HTTPException(status_code=500, detail=f"Query execution failed: {str(e)}")
    except Exception as e:
        # sanitize_error already logs the full exception with an error_id
        safe_message = sanitize_error(e, context=f"POST /query | question: {request.question}")
        raise HTTPException(status_code=500, detail=safe_message)