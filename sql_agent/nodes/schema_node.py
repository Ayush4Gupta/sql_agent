import logging
from typing import Literal

from langchain_core.messages import AIMessage, SystemMessage
from pydantic import BaseModel

from sql_agent.prompts.sql_prompts import schema_analysis_prompt

logger = logging.getLogger(__name__)


# ── Structured output schema ──────────────────────────────────────────────────

class SchemaDecision(BaseModel):
    decision: Literal["ANALYSIS_COMPLETE", "NEED_MORE_SCHEMAS"]
    tables_needed: list[str] = []   # optional — empty when ANALYSIS_COMPLETE
    reasoning: str = ""             # optional fallback


# ── call_get_schema node ──────────────────────────────────────────────────────

def create_call_get_schema_node(llm, get_schema_tool, db_context: str = ""):
    def call_get_schema(state):
        iteration = (state.get("schema_iterations") or 0) + 1
        already = set(state.get("retrieved_tables") or [])
        logger.info(
            "[call_get_schema] Iteration %d — already have schemas for: %s",
            iteration,
            sorted(already) if already else "(none)",
        )

        llm_with_tools = llm.bind_tools([get_schema_tool], tool_choice="any")
        context_addition = f"\n\nDATABASE RELATIONSHIPS TO CONSIDER:\n{db_context}" if db_context else ""
        system_hint = {
            "role": "system",
            "content": (
                "You are selecting database table schemas needed to answer a SQL question. "
                "Be generous — select ALL tables that might be needed including dimension "
                "tables (for readable values), date tables (for filtering), and any "
                "fact tables involved in multi-step calculations. "
                "You can request multiple tables in one call by comma-separating them."
                + context_addition
            )
        }
        response = llm_with_tools.invoke([system_hint] + state["messages"])

        # Track which tables are being requested
        new_tables = []
        if response.tool_calls:
            raw = response.tool_calls[0]["args"].get("table_names", "")
            new_tables = [t.strip() for t in raw.split(",") if t.strip()]
            logger.info("[call_get_schema] LLM requested schemas for: %s", new_tables)
        else:
            logger.warning("[call_get_schema] LLM made no tool call — no tables requested")

        # Merge with already retrieved tables (deduplicate)
        merged = list(already | set(new_tables))
        new_only = [t for t in new_tables if t not in already]
        if new_only:
            logger.debug("[call_get_schema] Net-new tables this iteration: %s", new_only)
        else:
            logger.warning(
                "[call_get_schema] All requested tables were already retrieved — "
                "no new schemas will be fetched"
            )

        return {
            "messages": [response],
            "retrieved_tables": merged,
            "schema_iterations": iteration,
        }

    return call_get_schema


# ── schema_analysis node ──────────────────────────────────────────────────────

def create_schema_analysis_node(llm, db_context: str = ""):
    llm_structured = llm.with_structured_output(SchemaDecision)
    # Build once at node creation — db_context is static (built at startup)
    system_msg = SystemMessage(content=schema_analysis_prompt(db_context=db_context))
    def schema_analysis(state):
        iteration = state.get("schema_iterations") or 0
        retrieved = state.get("retrieved_tables") or []
        logger.info(
            "[schema_analysis] Evaluating schema completeness "
            "(iteration %d, retrieved: %s)",
            iteration,
            retrieved,
        )

        try:
            result: SchemaDecision = llm_structured.invoke(
                [system_msg] + state["messages"]
            )
        except Exception as exc:
            logger.warning(
                "[schema_analysis] Structured output failed (%s) — "
                "forcing ANALYSIS_COMPLETE to unblock pipeline",
                exc,
            )
            result = SchemaDecision(
                decision="ANALYSIS_COMPLETE",
                tables_needed=[],
                reasoning=f"Structured output parse failed: {exc}",
            )

        if result.decision == "ANALYSIS_COMPLETE":
            logger.info(
                "[schema_analysis] Decision: ANALYSIS_COMPLETE — proceeding to query generation"
            )
        else:
            logger.info(
                "[schema_analysis] Decision: NEED_MORE_SCHEMAS — tables needed: %s",
                result.tables_needed,
            )

        logger.debug("[schema_analysis] Reasoning: %s", result.reasoning)

        # Store as AIMessage so future nodes see the decision in history
        summary = AIMessage(
            content=(
                f"[Schema Analysis] Decision: {result.decision}. "
                f"Reasoning: {result.reasoning}. "
                f"Tables still needed: {result.tables_needed}"
            )
        )
        return {"messages": [summary]}

    return schema_analysis