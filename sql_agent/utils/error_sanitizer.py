import logging
import uuid

logger = logging.getLogger(__name__)


# ── Safe user-facing messages ──────────────────────────────────────────────────
# These are ALL the messages the frontend will ever see.
# Each key maps to one friendly sentence — no internal details.

_USER_MESSAGES = {
    "query_failed":    "Unable to process your request. Please try rephrasing your question.",
    "no_data":         "No matching records were found for your query.",
    "timeout":         "The query took too long to complete. Please try a more specific question.",
    "permission":      "This type of operation is not permitted.",
    "connection":      "The database is currently unavailable. Please try again shortly.",
    "invalid_input":   "Your question could not be understood. Please try rephrasing it.",
    "schema_error":    "Unable to read the database structure. Please try again.",
    "unknown":         "An unexpected error occurred. Please try again.",
}


def sanitize_error(exc: Exception, context: str = "") -> str:
    """
    Convert any internal exception into a safe, user-facing string.

    Logs the full error internally (with a unique error_id so support can
    correlate logs to user reports). Returns only a generic message externally.

    Parameters
    ----------
    exc     : The original exception.
    context : Optional description of where the error happened (for logs only).

    Returns
    -------
    A safe string suitable for display to the end user.
    """
    error_id = uuid.uuid4().hex[:8].upper()     # e.g. "A3F1B2C9" — for log correlation
    exc_type = type(exc).__name__
    exc_msg  = str(exc)

    # Always log the full details internally
    logger.error(
        "[error_sanitizer] error_id=%s | context=%s | type=%s | detail=%s",
        error_id,
        context or "unspecified",
        exc_type,
        exc_msg,
    )

    # Classify the exception to pick the right user message
    category = _classify(exc, exc_msg)
    user_message = _USER_MESSAGES.get(category, _USER_MESSAGES["unknown"])

    # Include error ID in the user message so support can look it up
    return f"{user_message} (Ref: {error_id})"


def _classify(exc: Exception, msg: str) -> str:
    """Map an exception to one of the safe message keys."""
    msg_lower = msg.lower()

    # Timeout patterns
    if any(w in msg_lower for w in ("timeout", "timed out", "query timeout")):
        return "timeout"

    # Permission / write-block patterns
    if isinstance(exc, PermissionError):
        return "permission"

    # Connection / config patterns
    if isinstance(exc, (ConnectionError, ConnectionRefusedError, ConnectionAbortedError)):
        return "connection"
    if any(w in msg_lower for w in (
        "login failed", "cannot open", "server not found",
        "network-related", "odbc", "driver", "tcp provider",
    )):
        return "connection"

    # SQL syntax / execution patterns — do NOT expose table/column names
    if any(w in msg_lower for w in (
        "invalid column", "invalid object", "syntax error",
        "multi-part identifier", "ambiguous column",
        "sqlalchemy", "operationalerror", "programmingerror",
    )):
        return "query_failed"

    # Schema / metadata errors
    if any(w in msg_lower for w in ("information_schema", "no such table", "schema")):
        return "schema_error"

    # File / config not found
    if isinstance(exc, (FileNotFoundError, ValueError)):
        return "connection"

    return "unknown"