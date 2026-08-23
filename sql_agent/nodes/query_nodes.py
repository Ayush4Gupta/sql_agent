import logging

from langchain_core.messages import ToolMessage
from sqlalchemy import text
from sql_agent.utils.error_sanitizer import sanitize_error
logger = logging.getLogger(__name__)


# ── run_query node ─────────────────────────────────────────────────────────────
# Replaces the LangChain ToolNode wrapper so we can capture structured (tabular)
# results alongside the plain-string result the LLM receives.

def create_run_query_node(db, run_query_tool):
    """
    Custom run_query node.

    Behaviour
    ---------
    1. Reads the SQL from the last message's tool_call (set by check_query).
    2. Executes the SQL **once** through the engine.
    3. Hard-caps the fetched rows at ``row_limit`` (from state, default 20) via
       ``fetchmany`` — this is a defense-in-depth guarantee that prevents
       unbounded result sets regardless of whether the LLM included a LIMIT
       clause.  On TPC-DS ``store_sales`` (~2.9M rows) an omitted LIMIT would
       otherwise pull the entire table into memory.
    4. Derives both the plain-string ToolMessage (for the LLM) and the
       structured ``{columns, rows}`` dict (for the UI) from the single fetch.
    5. Falls back gracefully on execution errors.
    """
    def run_query(state):
        last_message = state["messages"][-1]
        tool_call    = last_message.tool_calls[0]
        sql          = tool_call["args"]["query"]
        tool_call_id = tool_call["id"]
        row_limit    = state.get("row_limit") or 20

        logger.info("[run_query] Executing SQL (row_limit=%d): %s", row_limit, sql)

        structured: dict = {"columns": [], "rows": []}
        result_str = ""

        try:
            with db._engine.connect() as conn:
                cursor  = conn.execute(text(sql))
                columns = list(cursor.keys())

                # Hard-cap: fetch at most row_limit rows regardless of query
                raw_rows  = cursor.fetchmany(row_limit)
                rows      = [list(row) for row in raw_rows]
                truncated = len(rows) == row_limit and cursor.fetchone() is not None

                structured = {"columns": columns, "rows": rows}

                # Build the string representation the LLM receives
                # (mirrors what LangChain's db.run() produces)
                result_str = str([dict(zip(columns, row)) for row in rows])

                if truncated:
                    result_str += f"\n[Results truncated to {row_limit} rows]"
                    logger.info(
                        "[run_query] Result truncated to %d rows (query returned more)",
                        row_limit,
                    )
                else:
                    logger.info(
                        "[run_query] Captured %d rows × %d columns",
                        len(rows), len(columns),
                    )
        except Exception as exc:
            result_str = f"Error executing query: {sanitize_error(exc, context='run_query')}"
            logger.error("[run_query] Execution failed: %s", exc)

        tool_message = ToolMessage(
            content=result_str,
            tool_call_id=tool_call_id,
            name="sql_db_query",
        )

        return {
            "messages":   [tool_message],
            "raw_results": structured,
        }

    return run_query



# ── generate_query node ────────────────────────────────────────────────────────

def create_generate_query_node(llm, run_query_tool, get_schema_tool,
                                dialect: str,
                                db_context: str = "", examples: list = None):
    from sql_agent.prompts.sql_prompts import generate_query_system_prompt
    from sql_agent.knowledge.knowledge_loader import find_relevant_examples, build_examples_context

    _examples = examples or []

    def generate_query(state):
        logger.info("[generate_query] Invoking LLM to generate SQL query")

        # ── Extract user question ──────────────────────────────────────────────
        user_question = next(
            (m["content"] if isinstance(m, dict) else m.content
             for m in state["messages"]
             if (isinstance(m, dict) and m.get("role") == "human")
             or getattr(m, "type", None) == "human"),
            ""
        )

        # ── Row limit from state (set by API, default 20) ──────────────────────
        row_limit = state.get("row_limit") or 20

        # ── Build prompts ──────────────────────────────────────────────────────
        relevant         = find_relevant_examples(user_question, _examples, n=2)
        examples_context = build_examples_context(relevant)

        system_prompt = generate_query_system_prompt(
            dialect=dialect,
            db_context=db_context,
            examples_context=examples_context,
            row_limit=row_limit,           # ← NEW: injected into prompt
        )

        system_message  = {"role": "system", "content": system_prompt}
        llm_with_tools  = llm.bind_tools([run_query_tool, get_schema_tool])
        response        = llm_with_tools.invoke([system_message] + state["messages"])

        if getattr(response, "tool_calls", None):
            tool_name = response.tool_calls[0]["name"]
            tool_args = response.tool_calls[0]["args"]
            if tool_name == "sql_db_query":
                logger.info(
                    "[generate_query] LLM produced SQL query: %s",
                    tool_args.get("query", ""),
                )
            elif tool_name == "sql_db_schema":
                logger.info(
                    "[generate_query] LLM requested additional schema for: %s",
                    tool_args.get("table_names", ""),
                )
            else:
                logger.warning(
                    "[generate_query] LLM called unexpected tool: %s", tool_name
                )
        else:
            content_preview = (response.content or "")[:200]
            logger.info(
                "[generate_query] LLM returned final answer (preview): %s",
                content_preview,
            )

        return {"messages": [response]}

    return generate_query


def create_check_query_node(llm, run_query_tool, system_prompt: str):
    def check_query(state):
        last_message = state["messages"][-1]
        tool_call    = last_message.tool_calls[0]
        query        = tool_call["args"]["query"]

        logger.info("[check_query] Validating SQL query: %s", query)

        # Extract user question for context
        user_question = next(
            (m["content"] if isinstance(m, dict) else m.content
             for m in state["messages"]
             if (isinstance(m, dict) and m.get("role") == "human")
             or getattr(m, "type", None) == "human"),
            ""
        )
        system_message = {"role": "system", "content": system_prompt}
        user_content = f"User Question: {user_question}\n\nCandidate SQL Query to validate:\n```sql\n{query}\n```"
        user_message   = {"role": "user",   "content": user_content}

        llm_with_tools = llm.bind_tools([run_query_tool], tool_choice="any")
        response       = llm_with_tools.invoke([system_message, user_message])

        if getattr(response, "tool_calls", None):
            validated_query = response.tool_calls[0]["args"].get("query", "")
            if validated_query != query:
                logger.info(
                    "[check_query] Query was modified by validator.\n  BEFORE: %s\n  AFTER : %s",
                    query,
                    validated_query,
                )
            else:
                logger.debug("[check_query] Query passed validation unchanged")
        else:
            response = last_message

        response.id = last_message.id
        return {"messages": [response]}

    return check_query
