import json
import logging
from pathlib import Path

from sql_agent.config.profile_loader import get_profile_examples_path

logger = logging.getLogger(__name__)

# Legacy fallback — used when no profile examples file exists
_LEGACY_EXAMPLES_PATH = Path(__file__).parent / "query_examples.json"


def load_examples() -> list[dict]:
    """Load verified query examples from the active profile or legacy location."""
    examples_path = get_profile_examples_path()
    if examples_path is None:
        logger.info("[knowledge_loader] No query_examples.json found — no examples loaded")
        return []
    with open(examples_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if not content:
            logger.warning("[knowledge_loader] query_examples.json is empty — no examples loaded")
            return []
    examples = json.loads(content)
    logger.info("[knowledge_loader] Loaded %d query examples from %s", len(examples), examples_path)
    return examples


def find_relevant_examples(question: str, examples: list[dict], n: int = 2) -> list[dict]:
    """Return top-n examples most relevant to the question by keyword overlap."""
    question_lower = question.lower()
    scored = []
    for ex in examples:
        score = sum(1 for kw in ex.get("keywords", []) if kw in question_lower)
        if score > 0:
            scored.append((score, ex))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [ex for _, ex in scored[:n]]


def build_examples_context(relevant: list[dict]) -> str:
    """Format examples into a prompt-ready string."""
    if not relevant:
        return ""
    lines = ["VERIFIED QUERY EXAMPLES (adapt these patterns — do not reinvent join chains):"]
    for ex in relevant:
        lines.append(f"\n  Question: {ex['question']}")
        tables = ex.get('required_tables', [])
        if tables:
            lines.append(f"  Required tables: {', '.join(tables)}")
        lines.append(f"  SQL:\n{ex['sql']}")
    return "\n".join(lines)