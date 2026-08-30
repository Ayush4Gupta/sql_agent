#!/usr/bin/env python
"""
Evaluation Harness — runs the 40-question set through the agent under
two conditions (baseline + flat_rules) and scores against gold SQL.

Usage:
    python eval/run_harness.py                     # full 40 questions
    python eval/run_harness.py --subset 5          # first N questions only
    python eval/run_harness.py --questions Q01,Q03  # specific questions
"""
import argparse
import decimal
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

# Project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

# Quiet down noisy loggers during eval
logging.basicConfig(level=logging.WARNING)
logging.getLogger("sql_agent").setLevel(logging.WARNING)
logging.getLogger("langchain").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

from sql_agent.database.db import get_db
from sql_agent.graph.agent import build_agent

logger = logging.getLogger("eval.harness")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))
logger.addHandler(handler)


# ─── Result comparison ──────────────────────────────────────────────────────

def normalize_value(v):
    """Normalize a single value for type-tolerant comparison."""
    if v is None:
        return None
    if isinstance(v, decimal.Decimal):
        return float(v)
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        # Try to parse as number
        try:
            return float(v)
        except (ValueError, TypeError):
            return v.strip().lower()
    return v


def normalize_rows(rows):
    """Normalize a list of row-tuples/lists for order-independent comparison."""
    if not rows:
        return set()
    normalized = []
    for row in rows:
        if isinstance(row, dict):
            row = tuple(row.values())
        elif isinstance(row, (list, tuple)):
            row = tuple(row)
        else:
            row = (row,)
        normalized.append(tuple(normalize_value(v) for v in row))
    return set(normalized)


def compare_results(agent_result, gold_result):
    """Compare agent result vs gold SQL result, order-independently and type-tolerantly.

    Returns (match: bool, details: str).
    """
    if not agent_result and not gold_result:
        return True, "both empty"
    if not agent_result:
        return False, "agent returned no results"
    if not gold_result:
        return False, "gold returned no results (bug in gold SQL?)"

    agent_rows = normalize_rows(agent_result)
    gold_rows = normalize_rows(gold_result)

    if agent_rows == gold_rows:
        return True, "exact match"

    # Check if gold is a subset (agent returned extra rows but got the key ones)
    if gold_rows.issubset(agent_rows):
        return True, f"gold is subset (agent has {len(agent_rows) - len(gold_rows)} extra rows)"

    # For single-value results, check approximate numeric match
    if len(gold_rows) == 1 and len(agent_rows) == 1:
        gold_row = list(gold_rows)[0]
        agent_row = list(agent_rows)[0]
        if len(gold_row) == 1 and len(agent_row) == 1:
            g, a = gold_row[0], agent_row[0]
            if isinstance(g, float) and isinstance(a, float) and g != 0:
                pct_diff = abs(g - a) / abs(g)
                if pct_diff < 0.001:  # within 0.1%
                    return True, f"numeric match within 0.1% (diff={pct_diff:.6f})"

    return False, f"mismatch: agent={sorted(agent_rows)[:3]}... gold={sorted(gold_rows)[:3]}..."


# ─── Extract agent SQL from final state ─────────────────────────────────────

def extract_agent_sql(state):
    """Extract the SQL query the agent generated from its message history."""
    messages = state.get("messages", [])
    # Walk backwards to find the last sql_db_query tool call
    for msg in reversed(messages):
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            for tc in tool_calls:
                if tc.get("name") == "sql_db_query":
                    return tc.get("args", {}).get("query", "")
    return ""


def extract_token_usage(state):
    """Extract total token usage from AIMessages in the state."""
    messages = state.get("messages", [])
    total_input = 0
    total_output = 0
    for msg in messages:
        usage = getattr(msg, "usage_metadata", None)
        if usage and isinstance(usage, dict):
            total_input += usage.get("input_tokens", 0)
            total_output += usage.get("output_tokens", 0)
    if total_input or total_output:
        return {"input_tokens": total_input, "output_tokens": total_output,
                "total_tokens": total_input + total_output}
    return None


# ─── Run a single question under one condition ──────────────────────────────

def run_single(question, agent, db, condition_name):
    """Run one question through the agent and score against gold SQL.

    Returns a result dict.
    """
    qid = question["id"]
    gold_sql = question["gold_sql"].strip()

    result = {
        "id": qid,
        "condition": condition_name,
        "question": question["question"],
        "category": question["category"],
        "ambiguous_rule": question.get("ambiguous_rule", ""),
        "rules_exercised": question.get("rules_exercised", []),
    }

    # 1. Run agent
    start = time.time()
    try:
        inputs = {
            "messages": [{"role": "user", "content": question["question"]}],
            "schema_iterations": 0,
            "retrieved_tables": [],
            "raw_results": {},
            "row_limit": 20,
            "node_timings": {},
        }
        final_state = agent.invoke(inputs)
        elapsed = time.time() - start

        result["latency_s"] = round(elapsed, 2)
        result["node_timings"] = final_state.get("node_timings", {})
        result["agent_sql"] = extract_agent_sql(final_state)
        result["token_usage"] = extract_token_usage(final_state)
        result["agent_error"] = None

        # Extract agent's raw results (already executed by run_query node)
        raw = final_state.get("raw_results") or {}
        cols = raw.get("columns", [])
        rows = raw.get("rows", [])
        agent_result = [tuple(row) for row in rows] if rows else []

    except Exception as e:
        elapsed = time.time() - start
        result["latency_s"] = round(elapsed, 2)
        result["node_timings"] = {}
        result["agent_sql"] = ""
        result["token_usage"] = None
        result["agent_error"] = str(e)[:500]
        agent_result = []

    # 2. Execute gold SQL
    try:
        gold_raw = db.run(gold_sql)
        # db.run returns a string repr of list of tuples — eval it
        # Need Decimal in scope since DuckDB returns Decimal for numeric types
        if isinstance(gold_raw, str):
            gold_result = eval(gold_raw, {"Decimal": decimal.Decimal, "__builtins__": {}})
        else:
            gold_result = gold_raw
    except Exception as e:
        result["pass"] = False
        result["comparison"] = f"gold SQL error: {e}"
        result["gold_result_preview"] = ""
        return result

    # 3. Compare
    match, details = compare_results(agent_result, gold_result)
    result["pass"] = match
    result["comparison"] = details

    # Previews for report
    gold_str = str(gold_result)
    result["gold_result_preview"] = gold_str[:200] if len(gold_str) > 200 else gold_str
    agent_str = str(agent_result)
    result["agent_result_preview"] = agent_str[:200] if len(agent_str) > 200 else agent_str

    return result


# ─── Build report ────────────────────────────────────────────────────────────

def json_serializable(obj):
    """Make objects JSON-serializable."""
    if isinstance(obj, decimal.Decimal):
        return float(obj)
    if isinstance(obj, set):
        return list(obj)
    return str(obj)


def build_report(results, timestamp):
    """Build markdown report and raw JSON from results."""
    conditions = {}
    for r in results:
        cond = r["condition"]
        if cond not in conditions:
            conditions[cond] = []
        conditions[cond].append(r)

    lines = []
    lines.append(f"# Evaluation Report — {timestamp}")
    lines.append("")

    # Overall accuracy per condition
    lines.append("## Overall Accuracy")
    lines.append("")
    lines.append("| Condition | Total | Pass | Fail | Accuracy |")
    lines.append("|---|---|---|---|---|")
    for cond, cond_results in conditions.items():
        total = len(cond_results)
        passed = sum(1 for r in cond_results if r["pass"])
        failed = total - passed
        acc = f"{100 * passed / total:.1f}%" if total else "N/A"
        lines.append(f"| {cond} | {total} | {passed} | {failed} | {acc} |")
    lines.append("")

    # Per-category breakdown
    lines.append("## Accuracy by Category")
    lines.append("")
    lines.append("| Condition | Category | Total | Pass | Fail | Accuracy |")
    lines.append("|---|---|---|---|---|---|")
    for cond, cond_results in conditions.items():
        cats = {}
        for r in cond_results:
            cat = r["category"]
            if cat not in cats:
                cats[cat] = {"pass": 0, "fail": 0}
            if r["pass"]:
                cats[cat]["pass"] += 1
            else:
                cats[cat]["fail"] += 1
        for cat in ["ambiguous", "governed", "baseline"]:
            if cat in cats:
                p, f = cats[cat]["pass"], cats[cat]["fail"]
                t = p + f
                acc = f"{100 * p / t:.1f}%" if t else "N/A"
                lines.append(f"| {cond} | {cat} | {t} | {p} | {f} | {acc} |")
    lines.append("")

    # Ambiguous questions detail
    lines.append("## Ambiguous Questions — Per-Question Detail")
    lines.append("")
    lines.append("| QID | Rule | Condition | Pass | Agent SQL (first 80 chars) |")
    lines.append("|---|---|---|---|---|")
    for r in sorted(results, key=lambda x: (x["id"], x["condition"])):
        if r["category"] == "ambiguous":
            sql_preview = r.get("agent_sql", "")[:80].replace("|", "\\|").replace("\n", " ")
            mark = "PASS" if r["pass"] else "FAIL"
            lines.append(f"| {r['id']} | {r['ambiguous_rule']} | {r['condition']} | {mark} | `{sql_preview}` |")
    lines.append("")

    # Latency and tokens
    lines.append("## Performance")
    lines.append("")
    for cond, cond_results in conditions.items():
        latencies = [r["latency_s"] for r in cond_results if r.get("latency_s")]
        avg_lat = sum(latencies) / len(latencies) if latencies else 0

        token_totals = [r["token_usage"]["total_tokens"] for r in cond_results
                        if r.get("token_usage") and r["token_usage"]]
        if token_totals:
            avg_tokens = sum(token_totals) / len(token_totals)
            lines.append(f"**{cond}**: avg latency = {avg_lat:.1f}s, "
                         f"avg tokens = {avg_tokens:.0f} ({len(token_totals)}/{len(cond_results)} reported)")
        else:
            lines.append(f"**{cond}**: avg latency = {avg_lat:.1f}s, token usage not available")
    lines.append("")

    # Failed questions detail
    lines.append("## Failed Questions")
    lines.append("")
    for r in results:
        if not r["pass"]:
            lines.append(f"### {r['id']} [{r['condition']}] — {r['question']}")
            lines.append(f"- **Category**: {r['category']}")
            if r.get("agent_error"):
                lines.append(f"- **Agent error**: {r['agent_error']}")
            else:
                lines.append(f"- **Comparison**: {r['comparison']}")
                lines.append(f"- **Agent SQL**: ```{r.get('agent_sql', 'N/A')}```")
                lines.append(f"- **Agent result**: {r.get('agent_result_preview', 'N/A')}")
                lines.append(f"- **Gold result**: {r.get('gold_result_preview', 'N/A')}")
            lines.append("")

    return "\n".join(lines)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SQL Agent Evaluation Harness")
    parser.add_argument("--subset", type=int, default=0,
                        help="Run only the first N questions (0 = all)")
    parser.add_argument("--questions", type=str, default="",
                        help="Comma-separated question IDs to run (e.g. Q01,Q03,Q35)")
    args = parser.parse_args()

    # Load questions
    questions_path = ROOT / "eval" / "questions.yaml"
    with open(questions_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    questions = data["questions"]

    # Filter
    if args.questions:
        ids = set(args.questions.split(","))
        questions = [q for q in questions if q["id"] in ids]
    elif args.subset > 0:
        questions = questions[:args.subset]

    logger.info("Running %d questions x 2 conditions = %d agent invocations",
                len(questions), len(questions) * 2)

    # Get shared DB for gold SQL execution
    db = get_db()

    all_results = []
    conditions = [
        ("baseline", {"DISABLE_RULES": "true"}),
        ("flat_rules", {"DISABLE_RULES": ""}),
    ]

    for cond_name, env_overrides in conditions:
        logger.info("=" * 60)
        logger.info("CONDITION: %s", cond_name)
        logger.info("=" * 60)

        # Set env vars for this condition
        for k, v in env_overrides.items():
            os.environ[k] = v

        # Rebuild agent for this condition (picks up DISABLE_RULES)
        agent, _ = build_agent()

        for i, q in enumerate(questions):
            logger.info("[%s] %d/%d  %s: %s",
                        cond_name, i + 1, len(questions), q["id"], q["question"][:60])
            try:
                result = run_single(q, agent, db, cond_name)
                mark = "PASS" if result["pass"] else "FAIL"
                logger.info("[%s] %s  %s  (%.1fs)  %s",
                            cond_name, q["id"], mark, result["latency_s"],
                            result["comparison"][:80])
            except Exception as e:
                logger.error("[%s] %s  ERROR: %s", cond_name, q["id"], e)
                result = {
                    "id": q["id"], "condition": cond_name,
                    "question": q["question"], "category": q["category"],
                    "ambiguous_rule": q.get("ambiguous_rule", ""),
                    "rules_exercised": q.get("rules_exercised", []),
                    "pass": False, "comparison": f"harness error: {e}",
                    "latency_s": 0, "agent_sql": "", "token_usage": None,
                    "agent_error": str(e)[:500], "node_timings": {},
                }
            all_results.append(result)

    # Clean up env
    os.environ.pop("DISABLE_RULES", None)

    # Generate report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = ROOT / "eval" / "results"
    results_dir.mkdir(exist_ok=True)

    # Write raw JSON
    json_path = results_dir / f"{timestamp}_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=json_serializable)
    logger.info("Raw results: %s", json_path)

    # Write markdown report
    report = build_report(all_results, timestamp)
    report_path = results_dir / f"{timestamp}_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info("Report: %s", report_path)

    # Print summary to stdout
    print("\n" + "=" * 60)
    for cond_name, _ in conditions:
        cond_results = [r for r in all_results if r["condition"] == cond_name]
        passed = sum(1 for r in cond_results if r["pass"])
        total = len(cond_results)
        print(f"  {cond_name}: {passed}/{total} ({100*passed/total:.1f}%)" if total else f"  {cond_name}: N/A")
    print("=" * 60)


if __name__ == "__main__":
    main()
