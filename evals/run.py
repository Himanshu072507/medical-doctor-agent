"""
Run rule-based evals against the symptom triage agent's SYSTEM_PROMPT.

Usage:
    python -m evals.run
    # cheaper/faster:
    EVAL_MODEL=llama-3.1-8b-instant python -m evals.run
    # multiple runs to detect flake:
    EVAL_RUNS=3 python -m evals.run

Reads GROQ_API_KEY from .env or shell.
"""

import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evals._shared import (  # noqa: E402
    get_groq_client,
    load_env,
    parse_sections,
    with_retry,
)

load_env()

from prompts import SYSTEM_PROMPT  # noqa: E402
from evals.cases import CASES  # noqa: E402


MODEL = os.getenv("EVAL_MODEL", "llama-3.3-70b-versatile")
TEMPERATURE = float(os.getenv("EVAL_TEMPERATURE", "0.0"))  # deterministic by default
MAX_TOKENS = int(os.getenv("EVAL_MAX_TOKENS", "2048"))
RUNS = int(os.getenv("EVAL_RUNS", "1"))
SAVE_DIR = ROOT / "evals" / "outputs"


def call_model(client, prompt):
    def _do():
        return client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )

    return with_retry(_do).choices[0].message.content


def score(text, case):
    failures = []
    sections = parse_sections(text)

    for pattern in case.get("must_include", []):
        if not re.search(pattern, text, re.IGNORECASE):
            failures.append(f"missing: /{pattern}/")
    for pattern in case.get("must_exclude", []):
        if re.search(pattern, text, re.IGNORECASE):
            failures.append(f"unexpected: /{pattern}/")

    for sec_num, patterns in (case.get("must_include_in_section") or {}).items():
        body = sections.get(sec_num, "")
        for pattern in patterns:
            if not re.search(pattern, body, re.IGNORECASE):
                failures.append(f"section {sec_num} missing: /{pattern}/")

    for sec_num, patterns in (case.get("must_exclude_in_section") or {}).items():
        body = sections.get(sec_num, "")
        for pattern in patterns:
            if re.search(pattern, body, re.IGNORECASE):
                failures.append(f"section {sec_num} unexpected: /{pattern}/")

    return failures


def run_once(client, run_id):
    SAVE_DIR.mkdir(exist_ok=True)
    results = []
    for i, case in enumerate(CASES, 1):
        name = case["name"]
        print(f"  [{i:>2}/{len(CASES)}] {name:<38}", end=" ", flush=True)
        try:
            text = call_model(client, case["prompt"])
            failures = score(text, case)
            suffix = f"_run{run_id}" if RUNS > 1 else ""
            (SAVE_DIR / f"{name}{suffix}.md").write_text(
                f"# {name} (run {run_id})\n\n## Prompt\n{case['prompt']}\n\n"
                f"## Response\n{text}\n"
            )
            status = "PASS" if not failures else f"FAIL ({len(failures)})"
            print(status)
            results.append((name, failures))
        except Exception as e:
            print(f"ERROR: {e}")
            results.append((name, [f"exception: {e}"]))
    return results


def main():
    client = get_groq_client()
    print(f"Model: {MODEL} | temperature: {TEMPERATURE} | runs: {RUNS}\n")

    t0 = time.time()
    pass_counts = {c["name"]: 0 for c in CASES}
    last_failures = {}
    for r in range(1, RUNS + 1):
        if RUNS > 1:
            print(f"── Run {r}/{RUNS} ──")
        results = run_once(client, r)
        for name, failures in results:
            if not failures:
                pass_counts[name] += 1
            else:
                last_failures[name] = failures
        if RUNS > 1:
            print()

    elapsed = time.time() - t0
    total = len(CASES)
    fully_passed = sum(1 for n in pass_counts if pass_counts[n] == RUNS)

    print("─" * 70)
    print(f"Result: {fully_passed}/{total} passed all {RUNS} run(s) in {elapsed:.1f}s\n")
    if RUNS > 1:
        print("Pass rate per case:")
        for name in pass_counts:
            rate = pass_counts[name] / RUNS
            tag = "✓" if rate == 1.0 else "⚠" if rate >= 0.5 else "✗"
            print(f"  {tag} {name:<38}  {pass_counts[name]}/{RUNS}")
        print()

    if fully_passed < total:
        print("Failures (most recent run):")
        for name, failures in last_failures.items():
            if pass_counts[name] < RUNS:
                print(f"  ✗ {name}")
                for f in failures:
                    print(f"      {f}")
                print(f"      see: evals/outputs/{name}.md\n")

    sys.exit(0 if fully_passed == total else 1)


if __name__ == "__main__":
    main()
