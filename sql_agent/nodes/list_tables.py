import logging

from langchain_core.messages import AIMessage

logger = logging.getLogger(__name__)


def create_list_tables_node(list_tables_tool):
    def list_tables(state):
        logger.info("[list_tables] Fetching all table names from database")

        tool_call = {
            "name": "sql_db_list_tables",
            "args": {},
            "id": "list_tables_call",
            "type": "tool_call",
        }

        tool_call_message = AIMessage(content="", tool_calls=[tool_call])
        tool_message = list_tables_tool.invoke(tool_call)
        response = AIMessage(content=f"Available tables: {tool_message.content}")

        table_names = [t.strip() for t in tool_message.content.split(",") if t.strip()]
        logger.info("[list_tables] Found %d tables: %s", len(table_names), table_names)

        return {"messages": [tool_call_message, tool_message, response]}

    return list_tables