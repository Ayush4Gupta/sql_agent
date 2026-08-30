"""
Profile loader — reads deployment/benchmark profiles from profiles/<name>/.

Each profile provides:
  - config.yaml   — DB backend, paths, table prefix, etc.
  - rules.yaml    — Business rules injected into LLM prompts
  - query_examples.json — Verified NL→SQL few-shot examples (optional)
"""
import logging
import os
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)

# profiles/ lives at the project root, two levels above sql_agent/config/
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PROFILES_DIR = _PROJECT_ROOT / "profiles"


def _active_profile_name() -> str:
    """Return the active profile name from env, defaulting to 'client'."""
    return os.getenv("AGENT_PROFILE", "client")


def load_profile_config() -> dict:
    """Load config.yaml from the active profile directory."""
    name = _active_profile_name()
    config_path = _PROFILES_DIR / name / "config.yaml"
    if not config_path.exists():
        logger.warning(
            "[profile] No config.yaml found for profile '%s' at %s — using empty config",
            name, config_path,
        )
        return {"profile_name": name}
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    config.setdefault("profile_name", name)
    logger.info("[profile] Loaded config for profile '%s' from %s", name, config_path)
    return config


def load_profile_rules() -> str:
    """Load business rules text from the active profile's rules.yaml.

    Returns the rules as a single string ready to inject into prompts.
    If no rules.yaml exists, returns an empty string (no rules = baseline).
    Set DISABLE_RULES=true to force an empty return (eval baseline condition).
    """
    if os.getenv("DISABLE_RULES", "").lower() in ("true", "1", "yes"):
        logger.info("[profile] DISABLE_RULES is set — returning empty rules")
        return ""
    name = _active_profile_name()
    rules_path = _PROFILES_DIR / name / "rules.yaml"
    if not rules_path.exists():
        logger.info("[profile] No rules.yaml for profile '%s' — running without business rules", name)
        return ""
    with open(rules_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    rules = data.get("rules", [])
    if not rules:
        logger.info("[profile] rules.yaml for profile '%s' has no rules entries", name)
        return ""

    # Build a numbered rule block for prompt injection
    lines = ["MANDATORY QUERY RULES:\n"]
    for i, rule in enumerate(rules, 1):
        rule_name = rule.get("name", f"Rule {i}")
        rule_text = rule.get("text", "").strip()
        lines.append(f"{i}. {rule_name.upper()}: {rule_text}\n")

    result = "\n".join(lines)
    logger.info("[profile] Loaded %d business rules for profile '%s'", len(rules), name)
    return result


def load_profile_schema_routing_rules() -> str:
    """Load schema routing rules from the active profile's rules.yaml.

    These are instructions that tell the schema_analysis node which tables
    to fetch for specific KPIs/query types.
    Returns empty string if not present.
    """
    if os.getenv("DISABLE_RULES", "").lower() in ("true", "1", "yes"):
        return ""
    name = _active_profile_name()
    rules_path = _PROFILES_DIR / name / "rules.yaml"
    if not rules_path.exists():
        return ""
    with open(rules_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    routing = data.get("schema_routing_rules", "")
    if routing:
        logger.info("[profile] Loaded schema routing rules for profile '%s'", name)
    return routing.strip() if isinstance(routing, str) else ""


def load_profile_check_rules() -> str:
    """Load check_query validation rules from the active profile's rules.yaml.

    These are the checklist items the check_query node uses to validate SQL.
    Returns empty string if not present.
    """
    if os.getenv("DISABLE_RULES", "").lower() in ("true", "1", "yes"):
        return ""
    name = _active_profile_name()
    rules_path = _PROFILES_DIR / name / "rules.yaml"
    if not rules_path.exists():
        return ""
    with open(rules_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    check_rules = data.get("check_query_rules", "")
    if check_rules:
        logger.info("[profile] Loaded check_query rules for profile '%s'", name)
    return check_rules.strip() if isinstance(check_rules, str) else ""


def get_profile_examples_path() -> Path | None:
    """Return the path to the active profile's query_examples.json, or None."""
    name = _active_profile_name()
    path = _PROFILES_DIR / name / "query_examples.json"
    if path.exists():
        logger.info("[profile] Found query_examples.json for profile '%s'", name)
        return path
    # Fallback: check the legacy location in knowledge/
    legacy = Path(__file__).resolve().parents[1] / "knowledge" / "query_examples.json"
    if legacy.exists():
        logger.info("[profile] Using legacy query_examples.json at %s", legacy)
        return legacy
    logger.info("[profile] No query_examples.json found for profile '%s'", name)
    return None
