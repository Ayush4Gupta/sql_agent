#!/usr/bin/env python
"""
Validate every gold SQL in eval/questions.yaml against tpcds_sf1.duckdb.

Checks:
  1. YAML parses correctly
  2. Every gold_sql executes without error
  3. Every query returns at least one row (no vacuous queries)
  4. Prints result summaries for spot-checking
"""
import os
import sys
import yaml

# Ensure project root is on path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, ".env"))

from sql_agent.database.db import get_db


def main():
    questions_path = os.path.join(ROOT, "eval", "questions.yaml")
    print(f"Loading: {questions_path}")

    with open(questions_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    questions = data["questions"]
    print(f"Found {len(questions)} questions\n")

    db = get_db()

    passed = 0
    failed = 0
    errors = []

    # Category counters
    cat_counts = {"ambiguous": 0, "governed": 0, "baseline": 0}

    for q in questions:
        qid = q["id"]
        cat = q["category"]
        question = q["question"]
        gold_sql = q["gold_sql"].strip()
        rules = q.get("rules_exercised", [])
        ambig_rule = q.get("ambiguous_rule", "")
        cat_counts[cat] = cat_counts.get(cat, 0) + 1

        try:
            result = db.run(gold_sql)
            # Parse result -- db.run returns a string representation of list of tuples
            if not result or result == "[]":
                raise ValueError("Query returned no rows")

            passed += 1
            # Truncate result for display
            result_str = str(result)
            if len(result_str) > 120:
                result_str = result_str[:120] + "..."
            label = f"[{cat}]"
            if ambig_rule:
                label += f" [{ambig_rule}]"
            print(f"  PASS {qid} {label}")
            print(f"    Q: {question}")
            print(f"    R: {result_str}")
            print(f"    Rules: {rules}")
            print()

        except Exception as e:
            failed += 1
            errors.append((qid, str(e)))
            print(f"  FAIL {qid} [{cat}] FAILED: {e}")
            print(f"    Q: {question}")
            print(f"    SQL: {gold_sql[:100]}...")
            print()

    # Summary
    print("=" * 70)
    print(f"RESULTS: {passed} passed, {failed} failed, {len(questions)} total")
    print(f"Categories: {cat_counts}")
    print()

    if errors:
        print("FAILURES:")
        for qid, err in errors:
            print(f"  {qid}: {err}")
        print()

    # Sanity checks
    assert len(questions) == 40, f"Expected 40 questions, got {len(questions)}"
    assert cat_counts.get("ambiguous", 0) == 14, f"Expected 14 ambiguous, got {cat_counts.get('ambiguous')}"
    assert cat_counts.get("governed", 0) == 20, f"Expected 20 governed, got {cat_counts.get('governed')}"
    assert cat_counts.get("baseline", 0) == 6, f"Expected 6 baseline, got {cat_counts.get('baseline')}"

    if failed == 0:
        print("ALL GOLD SQL VALIDATED SUCCESSFULLY")
    else:
        print(f"VALIDATION FAILED -- {failed} queries had errors")
        sys.exit(1)


if __name__ == "__main__":
    main()
