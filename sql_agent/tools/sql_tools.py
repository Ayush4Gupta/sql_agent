import logging

from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_core.tools import BaseTool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_community.utilities import SQLDatabase

logger = logging.getLogger(__name__)


def get_sql_tools(db: SQLDatabase, llm: BaseChatModel) -> list[BaseTool]:
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = toolkit.get_tools()
    logger.debug("SQL tools loaded: %s", [t.name for t in tools])
    return tools


def split_sql_tools(tools: list[BaseTool]) -> dict[str, BaseTool]:
    tool_map = {tool.name: tool for tool in tools}
    logger.debug("Tool map keys: %s", list(tool_map.keys()))
    return tool_map