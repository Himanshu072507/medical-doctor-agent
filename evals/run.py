"""
Run rule-based evals against the medical triage agent's SYSTEM_PROMPT.

Usage:
    python -m evals.run
    # or with a different model:
    EVAL_MODEL=llama-3.1-8b-instant python -m evals.run

Reads GROQ_API_KEY from .env or shell.
"""

import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evals._shared import get_groq_client, load_env, with_retry  # noqa: E402

load_env()

from prompts import SYSTEM_PROMPT  # noqa: E402
from evals.cases import CASES  # noqa: E402


MODEL = os.getenv("EVAL_MODEL", "llama-3.3-70b-versatile")
TEMPERATURE = float(os.getenv("EVAL_TEMPERATURE", "0.4"))
MAX_TOKENS = int(os.getenv("EVAL_MAX_TOKENS", "2048"))
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
    for pattern in case.get("must_include", []):
        if not re.search(pattern, text, re.IGNORECASE):
            failures.append(f"missing: /{pattern}/")
    for pattern in case.get("must_exclude", []):
        if re.search(pattern, text, re.IGNORECASE):
            failures.append(f"unexpected match: /{pattern}/")
    return failures


def main():
    client = get_groq_client()
    SAVE_DIR.mkdir(exist_ok=True)

    print(f"Running {len(CASES)} eval cases on model: {MODEL}\n")
    results = []
    t0 = time.time()

    for i, case in enumerate(CASES, 1):
        name = case["name"]
        print(f"  [{i:>2}/{len(CASES)}] {name:<38}", end=" ", flush=True)
        try:
            text = call_model(client, case["prompt"])
            failures = score(text, case)
            (SAVE_DIR / f"{name}.md").write_text(
                f"# {name}\n\n## Prompt\n{case['prompt']}\n\n## Response\n{text}\n"
            )
            status = "PASS" if not failures else f"FAIL ({len(failures)})"
            print(status)
            results.append((name, failures))
        except Exception as e:
            print(f"ERROR: {e}")
            results.append((name, [f"exception: {e}"]))

    elapsed = time.time() - t0
    passed = sum(1 for _, f in results if not f)
    total = len(results)

    print("\n" + "─" * 70)
    print(f"Result: {passed}/{total} passed in {elapsed:.1f}s\n")

    if passed < total:
        print("Failures:\n")
        for name, failures in results:
            if failures:
                print(f"  ✗ {name}")
                for f in failures:
                    print(f"      {f}")
                print(f"      see: evals/outputs/{name}.md\n")

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
