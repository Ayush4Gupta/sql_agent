import logging

logger = logging.getLogger(__name__)


def run_query(agent, question: str):
    logger.info("=== New query ===")
    logger.info("Question: %s", question)

    inputs = {
        "messages": [{"role": "user", "content": question}],
        "schema_iterations": 0,
        "retrieved_tables": [],
        "raw_results": {},
        "row_limit": 20,
    }

    step_num = 0
    for step in agent.stream(inputs, stream_mode="values"):
        step_num += 1
        last_msg = step["messages"][-1]
        msg_type = getattr(last_msg, "type", "unknown")
        has_tool_calls = bool(getattr(last_msg, "tool_calls", None))

        logger.debug(
            "Stream step %d — message type: %s, has_tool_calls: %s",
            step_num,
            msg_type,
            has_tool_calls,
        )
        last_msg.pretty_print()

    logger.info("=== Query complete (%d stream steps) ===", step_num)