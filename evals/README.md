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

```bash
export GROQ_API_KEY=gsk_...   # free key at https://console.groq.com/keys
python -m evals.run
```

Optional environment overrides:

```bash
EVAL_MODEL=llama-3.1-8b-instant python -m evals.run    # cheaper/faster
EVAL_TEMPERATURE=0.2 python -m evals.run               # less variance
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
