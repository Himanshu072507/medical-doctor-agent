"""
Inter-judge agreement: run BOTH the Groq judge and the Gemini judge against
the same agent responses, then report how often they agree.

Why: a single LLM judge can share blind spots with the agent (especially on
the same provider). Cross-provider agreement is a sanity check against
self-grading bias.

Usage:
    python -m evals.agreement
    # Limit to a subset:
    AGREEMENT_LIMIT=5 python -m evals.agreement

Requires GROQ_API_KEY and GEMINI_API_KEY in .env or shell.
"""

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evals._shared import (  # noqa: E402
    get_gemini_client,
    get_groq_client,
    load_env,
    with_retry,
)

load_env()

from prompts import SYSTEM_PROMPT  # noqa: E402
from evals.cases import CASES  # noqa: E402
from evals.judge import (  # noqa: E402
    CRITERIA,
    JUDGE_SYSTEM_PROMPT,
    call_judge_gemini,
    call_judge_groq,
    evaluate_verdict,
)


AGENT_MODEL = os.getenv("EVAL_MODEL", "llama-3.3-70b-versatile")
AGENT_TEMP = float(os.getenv("EVAL_TEMPERATURE", "0.0"))
GROQ_JUDGE_MODEL = os.getenv("GROQ_JUDGE_MODEL", "openai/gpt-oss-120b")
GEMINI_JUDGE_MODEL = os.getenv("GEMINI_JUDGE_MODEL", "gemini-2.5-pro")
LIMIT = int(os.getenv("AGREEMENT_LIMIT", "0"))  # 0 = all
SAVE_DIR = ROOT / "evals" / "outputs"


def call_agent(client, user_prompt):
    def _do():
        return client.chat.completions.create(
            model=AGENT_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=AGENT_TEMP,
            max_tokens=2048,
        )

    return with_retry(_do).choices[0].message.content


def call_groq_judge_with_model(client, user_prompt, agent_response):
    """Wrap call_judge_groq with our explicit GROQ_JUDGE_MODEL override."""
    judge_input = f"USER INPUT:\n{user_prompt}\n\nAGENT RESPONSE:\n{agent_response}"

    def _do():
        return client.chat.completions.create(
            model=GROQ_JUDGE_MODEL,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": judge_input},
            ],
            temperature=0.0,
            max_tokens=1024,
            response_format={"type": "json_object"},
        )

    return json.loads(with_retry(_do).choices[0].message.content)


def call_gemini_judge_with_model(client, user_prompt, agent_response):
    from google.genai import types

    judge_input = f"USER INPUT:\n{user_prompt}\n\nAGENT RESPONSE:\n{agent_response}"

    def _do():
        return client.models.generate_content(
            model=GEMINI_JUDGE_MODEL,
            contents=judge_input,
            config=types.GenerateContentConfig(
                system_instruction=JUDGE_SYSTEM_PROMPT,
                temperature=0.0,
                max_output_tokens=1024,
                response_mime_type="application/json",
            ),
        )

    return json.loads(with_retry(_do).text)


def main():
    groq_client = get_groq_client()
    gemini_client = get_gemini_client()
    SAVE_DIR.mkdir(exist_ok=True)

    cases = CASES[:LIMIT] if LIMIT > 0 else CASES

    print(f"Agent:        groq / {AGENT_MODEL}  (temp={AGENT_TEMP})")
    print(f"Groq judge:   {GROQ_JUDGE_MODEL}")
    print(f"Gemini judge: {GEMINI_JUDGE_MODEL}")
    print(f"Cases:        {len(cases)}\n")

    rows = []
    t0 = time.time()

    for i, case in enumerate(cases, 1):
        name = case["name"]
        print(f"  [{i:>2}/{len(cases)}] {name:<38}", end=" ", flush=True)
        try:
            agent_text = call_agent(groq_client, case["prompt"])
            v_groq = call_groq_judge_with_model(groq_client, case["prompt"], agent_text)
            v_gem = call_gemini_judge_with_model(gemini_client, case["prompt"], agent_text)

            p_groq, s_groq, _ = evaluate_verdict(v_groq)
            p_gem, s_gem, _ = evaluate_verdict(v_gem)

            mode_g = v_groq.get("mode", "?")
            mode_e = v_gem.get("mode", "?")
            mode_match = mode_g == mode_e
            pass_match = p_groq == p_gem

            diffs = {c: abs(s_groq.get(c, 0) - s_gem.get(c, 0)) for c in CRITERIA}
            mean_diff = sum(diffs.values()) / len(CRITERIA)

            print(
                f"groq={'P' if p_groq else 'F'} gem={'P' if p_gem else 'F'} "
                f"mode_match={'Y' if mode_match else 'N'} mean_diff={mean_diff:.2f}"
            )
            rows.append({
                "name": name,
                "p_groq": p_groq,
                "p_gem": p_gem,
                "pass_match": pass_match,
                "mode_groq": mode_g,
                "mode_gem": mode_e,
                "mode_match": mode_match,
                "scores_groq": s_groq,
                "scores_gem": s_gem,
                "diffs": diffs,
                "mean_diff": mean_diff,
            })
        except Exception as e:
            print(f"ERROR: {e}")
            rows.append({"name": name, "error": str(e)})

    elapsed = time.time() - t0
    valid = [r for r in rows if "error" not in r]
    n = len(valid)

    if not valid:
        print("\nNo valid runs to summarize.")
        sys.exit(2)

    pass_agree = sum(1 for r in valid if r["pass_match"])
    mode_agree = sum(1 for r in valid if r["mode_match"])
    overall_mean_diff = sum(r["mean_diff"] for r in valid) / n

    print("\n" + "─" * 70)
    print(f"Inter-judge agreement over {n} cases ({elapsed:.1f}s)\n")
    print(f"  Pass/fail agreement:     {pass_agree}/{n}  ({100*pass_agree/n:.0f}%)")
    print(f"  Mode (A/B/C) agreement:  {mode_agree}/{n}  ({100*mode_agree/n:.0f}%)")
    print(f"  Mean per-criterion |Δ|:  {overall_mean_diff:.2f} points (out of 4 max)")

    print("\nPer-criterion mean |Δ|:")
    for c in CRITERIA:
        vals = [r["diffs"][c] for r in valid]
        print(f"  {c:<14} {sum(vals)/len(vals):.2f}")

    disagreements = [r for r in valid if not r["pass_match"]]
    if disagreements:
        print("\nPass/fail disagreements:")
        for r in disagreements:
            print(
                f"  ✗ {r['name']:<38}  groq={'P' if r['p_groq'] else 'F'}  "
                f"gem={'P' if r['p_gem'] else 'F'}"
            )

    out_path = SAVE_DIR / "agreement_report.json"
    out_path.write_text(json.dumps(rows, indent=2))
    print(f"\nFull report saved to: {out_path}")

    sys.exit(0 if pass_agree == n else 1)


if __name__ == "__main__":
    main()
