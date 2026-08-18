import logging
import logging.handlers
import os
from pathlib import Path

from sql_agent.config.settings import LOG_LEVEL

# ── Log file location ─────────────────────────────────────────────────────────

LOGS_DIR = Path(__file__).resolve().parents[2] / "logs"
LOGS_DIR.mkdir(exist_ok=True)
LOG_FILE = LOGS_DIR / "sql_agent.log"

# ── Format ────────────────────────────────────────────────────────────────────

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# ── Third-party loggers to silence (they are very noisy at DEBUG) ─────────────

_NOISY_LOGGERS = [
    "httpx",
    "httpcore",
    "openai",
    "langchain",
    "langchain_core",
    "langchain_community",
    "langgraph",
    "urllib3",
    "asyncio",
    "uvicorn.access",
]


def configure_logging() -> None:
    """
    Call once at application startup (main.py).

    Behaviour:
      - Console handler : INFO and above
      - Rotating file handler : DEBUG and above  →  logs/sql_agent.log
        (max 5 MB per file, keeps last 3 files)
      - All sql_agent.* loggers : level controlled by LOG_LEVEL env-var (default INFO)
      - Noisy third-party libs : forced to WARNING to keep output clean
    """
    root_logger = logging.getLogger()
    # Only configure once even if called multiple times
    if root_logger.handlers:
        return

    root_logger.setLevel(logging.DEBUG)  # root catches everything; handlers filter

    formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)

    # ── Console handler ───────────────────────────────────────────────────────
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # ── Rotating file handler ─────────────────────────────────────────────────
    file_handler = logging.handlers.RotatingFileHandler(
        LOG_FILE,
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # ── Silence noisy third-party libraries ───────────────────────────────────
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    # ── Set level for our own package ─────────────────────────────────────────
    # Set to DEBUG so messages flow through to handlers; console handler is
    # already capped at INFO, file handler captures DEBUG and above.
    logging.getLogger("sql_agent").setLevel(logging.DEBUG)

    logging.getLogger(__name__).info(
        "Logging configured — console: INFO+, file: DEBUG+ → %s", LOG_FILE
    )
