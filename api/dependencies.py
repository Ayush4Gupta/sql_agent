import logging

from sql_agent.config.logging_config import configure_logging
from sql_agent.graph.agent import build_agent

# Configure logging before anything else so all subsequent loggers inherit the config
configure_logging()

logger = logging.getLogger(__name__)
logger.info("[dependencies] Initialising agent and database...")

agent, db = build_agent()

logger.info("[dependencies] Agent and database ready.")
