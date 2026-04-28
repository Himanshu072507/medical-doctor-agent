# Evals — Medical Triage Agent

Rule-based regression checks for the agent's `SYSTEM_PROMPT`. Each test case sends a user prompt to the model (via Groq) and asserts that the response contains expected patterns and avoids forbidden ones.

## What's tested

| Category | Why |
|---|---|
| **Structure** | All 7 sections present; correct headings |
| **Clarification** | Vague input ("I feel unwell") triggers questions, not a definitive diagnosis |
| **Red flags** | Thunderclap headache, cardiac chest pain, stroke signs, meningitis → ER mentioned |
| **No edge-case jumping** | Common cold doesn't get diagnosed as brain tumor; mild back strain doesn't trigger cancer workup |
| **Clinical reasoning** | Migraine pattern recognized; positional vertigo → BPPV; severe headache + nosebleed → BP check |
| **Always-on** | Disclaimer + evaluation steps appear in every response |

## Run

Two runners are available:

```bash
export GROQ_API_KEY=gsk_...   # free key at https://console.groq.com/keys

# 1. Rule-based (fast, regex-based assertions)
python -m evals.run

# 2. LLM-as-judge (slower, scores 1-5 on 5 criteria)
python -m evals.judge
```

### Rule-based runner (`evals.run`)

Pattern matching via `must_include` / `must_exclude` regex per case. Pass = all patterns satisfied.

### LLM-as-judge runner (`evals.judge`)

A stronger Groq model (`openai/gpt-oss-120b` by default) grades each agent response on:

| Criterion | What it checks |
|---|---|
| `structure` | All 7 sections present and ordered |
| `safety` | Red flags surfaced; ER guidance accurate; no harm |
| `calibration` | Conservative diagnosis; clarifying questions for vague input; no edge-case jumping |
| `completeness` | Actionable next steps (tests, self-care, escalation) |
| `disclaimer` | "Not a substitute" reminder is present |

Each gets a 1–5 score with reasoning, returned as JSON. Pass = every criterion ≥ `MIN_SCORE` (default 3) AND `overall_pass` is true.

Optional environment overrides:

```bash
EVAL_MODEL=llama-3.1-8b-instant python -m evals.run        # cheaper agent
JUDGE_MODEL=llama-3.3-70b-versatile python -m evals.judge  # different judge
JUDGE_TEMPERATURE=0.0 python -m evals.judge                # deterministic
MIN_SCORE=4 python -m evals.judge                          # stricter
```

## Output

- Pass/fail summary per case in stdout
- Full model response saved to `evals/outputs/<case_name>.md` for inspection
- Exit code `0` if all pass, `1` if any fail (suitable for CI)

## Adding a new case

Edit `evals/cases.py` and append:

```python
{
    "name": "my_new_check",
    "prompt": "User message here",
    "must_include": [r"regex1", r"regex2"],
    "must_exclude": [r"regex_to_avoid"],   # optional
}
```

Patterns are matched case-insensitively. Use `\b` word boundaries to avoid false matches.

## Limitations

- Rule-based only — catches *presence/absence* of keywords, not factual correctness or nuance.
- For deeper grading (faithfulness, completeness, harm), add an LLM-as-judge pass.
- Model output is non-deterministic at temperature > 0; expect occasional flake. Lower `EVAL_TEMPERATURE` for stricter regression testing.
