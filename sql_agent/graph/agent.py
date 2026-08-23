import logging
from typing import Annotated, Any

from sql_agent.knowledge.db_analyzer import build_db_context
from sql_agent.knowledge.knowledge_loader import load_examples

from sql_agent.config.llm_factory import get_llm
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

import time

from sql_agent.config.settings import MAX_SCHEMA_ITERATIONS
logger = logging.getLogger(__name__)
from sql_agent.database.db import get_db
from sql_agent.nodes.list_tables import create_list_tables_node
from sql_agent.nodes.query_nodes import (
    create_check_query_node,
    create_generate_query_node,
    create_run_query_node,          # ← NEW import
)
from sql_agent.nodes.schema_node import create_call_get_schema_node, create_schema_analysis_node
from sql_agent.prompts.sql_prompts import check_query_system_prompt, generate_query_system_prompt
from sql_agent.tools.sql_tools import get_sql_tools, split_sql_tools


# ── Extended State ─────────────────────────────────────────────────────────────

def _merge_timings(existing: dict | None, new: dict | None) -> dict:
    """Reducer: merge node-timing dicts so each node appends its entry."""
    merged = dict(existing or {})
    merged.update(new or {})
    return merged


class AgentState(MessagesState):
    schema_iterations: int                                          # how many schema fetches have run
    retrieved_tables: list[str]                                     # table names whose schemas are in context
    raw_results: dict                                               # {"columns": [...], "rows": [[...], ...]}
    row_limit: int                                                  # max rows to return (default 20)
    node_timings: Annotated[dict[str, float], _merge_timings]       # {node_name: elapsed_seconds}


# ── Agent Builder ──────────────────────────────────────────────────────────────

def build_agent():
    logger.info(
        "Building agent — max_schema_iterations: %s",
        MAX_SCHEMA_ITERATIONS,
    )

    db = get_db()

    # Build database context once at startup
    logger.info("Building database context (Layer 1 + 2)...")
    db_context = build_db_context(db)

    # Load verified examples (Layer 3)
    examples = load_examples()

    # Provider-agnostic LLM — controlled by LLM_PROVIDER env var
    llm = get_llm()

    tools = get_sql_tools(db=db, llm=llm)
    tool_map = split_sql_tools(tools)

    list_tables_tool = tool_map["sql_db_list_tables"]
    get_schema_tool  = tool_map["sql_db_schema"]
    run_query_tool   = tool_map["sql_db_query"]

    # ── Nodes ──────────────────────────────────────────────────────────────────
    get_schema_node      = ToolNode([get_schema_tool], name="get_schema")

    # ↓ CHANGED: was ToolNode([run_query_tool]); now a custom node that also
    #   captures structured tabular data into state["raw_results"]
    run_query_node       = create_run_query_node(db=db, run_query_tool=run_query_tool)

    list_tables_node     = create_list_tables_node(list_tables_tool)
    call_get_schema_node = create_call_get_schema_node(llm, get_schema_tool, db_context=db_context)
    schema_analysis_node = create_schema_analysis_node(llm, db_context=db_context)
    generate_query_node  = create_generate_query_node(
        llm=llm,
        run_query_tool=run_query_tool,
        get_schema_tool=get_schema_tool,
        dialect=db.dialect,
        db_context=db_context,
        examples=examples,
    )
    check_query_node     = create_check_query_node(
        llm=llm,
        run_query_tool=run_query_tool,
        system_prompt=check_query_system_prompt(dialect=db.dialect),
    )

    # ── Routing: after generate_query ─────────────────────────────────────────

    def should_continue(state: AgentState) -> str:
        last_message = state["messages"][-1]
        if not getattr(last_message, "tool_calls", None):
            logger.info("[should_continue] No tool call — routing to END")
            return END

        tool_call = last_message.tool_calls[0]
        if tool_call["name"] == "sql_db_query":
            logger.info("[should_continue] SQL query tool called — routing to check_query")
            return "check_query"
        elif tool_call["name"] == "sql_db_schema":
            logger.info(
                "[should_continue] Schema tool called mid-generation — routing to get_schema"
            )
            return "get_schema"
        else:
            logger.warning(
                "[should_continue] Unknown tool call: %s — routing to END", tool_call["name"]
            )
            return END

    # ── Routing: after schema_analysis ────────────────────────────────────────

    def route_schema_analysis(state: AgentState) -> str:
        iterations = state.get("schema_iterations") or 0

        # Layer 1: Hard iteration cap — always terminates
        if iterations >= MAX_SCHEMA_ITERATIONS:
            logger.warning(
                "[route_schema_analysis] Hit MAX_SCHEMA_ITERATIONS (%d) — "
                "forcing generate_query with available schemas",
                MAX_SCHEMA_ITERATIONS,
            )
            return "generate_query"

        # Layer 2: Read schema_analysis decision from last message
        last_message = state["messages"][-1]
        if "NEED_MORE_SCHEMAS" in last_message.content:
            logger.info(
                "[route_schema_analysis] NEED_MORE_SCHEMAS — looping back to call_get_schema "
                "(iteration %d / %d)",
                iterations,
                MAX_SCHEMA_ITERATIONS,
            )
            return "call_get_schema"

        # Layer 3: Default — ANALYSIS_COMPLETE or unrecognised → proceed
        logger.info(
            "[route_schema_analysis] ANALYSIS_COMPLETE — routing to generate_query "
            "(retrieved: %s)",
            state.get("retrieved_tables") or [],
        )
        return "generate_query"
    
    def timed(name, fn):
        """Wrap a node function to record wall-clock elapsed time into state."""
        def wrapper(state):
            t0 = time.perf_counter()
            # ToolNode (LangChain Runnable) uses .invoke(); plain functions use fn(state)
            result = fn.invoke(state) if hasattr(fn, "invoke") else fn(state)
            elapsed = round(time.perf_counter() - t0, 3)
            logger.info("[TIMING] %s → %.3fs", name, elapsed)

            # Inject timing into the state update dict so it accumulates
            if isinstance(result, dict):
                result["node_timings"] = {name: elapsed}
            return result
        return wrapper

    def route_call_get_schema(state: AgentState) -> str:
        last_message = state["messages"][-1]
        if getattr(last_message, "tool_calls", None):
            logger.info("[route_call_get_schema] Tool calls present — routing to get_schema")
            return "get_schema"
        logger.info("[route_call_get_schema] No tool call made — routing to generate_query")
        return "generate_query"

    # ── Graph ──────────────────────────────────────────────────────────────────

    builder = StateGraph(AgentState)

    builder.add_node("list_tables",     timed("list_tables",     list_tables_node))
    builder.add_node("call_get_schema", timed("call_get_schema", call_get_schema_node))
    builder.add_node("get_schema",      timed("get_schema",      get_schema_node))
    builder.add_node("schema_analysis", timed("schema_analysis", schema_analysis_node))
    builder.add_node("generate_query",  timed("generate_query",  generate_query_node))
    builder.add_node("check_query",     timed("check_query",     check_query_node))
    builder.add_node("run_query",       timed("run_query",       run_query_node))   # now a plain function node

    builder.add_edge(START,              "list_tables")
    builder.add_edge("list_tables",      "call_get_schema")
    builder.add_conditional_edges(
        "call_get_schema",
        route_call_get_schema,
        {
            "get_schema":     "get_schema",
            "generate_query": "generate_query",
        }
    )
    builder.add_edge("get_schema",       "schema_analysis")

    builder.add_conditional_edges(
        "schema_analysis",
        route_schema_analysis,
        {
            "call_get_schema": "call_get_schema",
            "generate_query":  "generate_query",
        }
    )

    builder.add_conditional_edges(
        "generate_query",
        should_continue,
        {
            "check_query": "check_query",
            "get_schema":  "get_schema",
            END:           END,
        },
    )
    builder.add_edge("check_query", "run_query")
    builder.add_edge("run_query",   "generate_query")

    agent = builder.compile()
    logger.info("Agent graph compiled successfully")
    return agent, db
