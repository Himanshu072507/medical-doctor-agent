# 🩺 Symptom Triage Helper

A localhost-only, structured-intake symptom triage assistant built with Streamlit. BYOK (bring your own Groq or Gemini key) — no server-side keys, nothing logged remotely.

> ⚠️ **Informational only — not medical advice.** This tool helps you organize symptoms and decide whether to seek care. It cannot diagnose you. For emergencies, call your local emergency number.

## What it does

1. Collects a structured symptom intake (5 required fields, more optional).
2. Sends a clean medical-style summary to an LLM with a careful system prompt.
3. Streams back a 7-section response: symptom understanding → differential → working diagnosis → recommended evaluation → self-care → red flags → disclaimer.
4. Lets you ask follow-up questions in chat, or edit the intake and re-submit.

## Safety features

- **Crisis pre-check**: any input matching suicide/self-harm patterns is intercepted *before* the LLM call and replaced with crisis-helpline information (India / US / UK / global).
- **No medication doses**: the system prompt forbids specific drug doses; the agent recommends consulting a doctor or pharmacist.
- **Pediatric / pregnancy redirects**: under-18 or pregnant intakes are routed toward a pediatrician or obstetrician.
- **Red-flag-first response**: severe-symptom inputs (chest pain, stroke signs, thunderclap headache, cauda equina, etc.) get an emergency instruction *first*, then the structured response.
- **Prominent disclaimer banner** at the top of every page.

## Limitations (please read)

- The agent is an LLM, not a doctor. It can hallucinate, miss context, or be confidently wrong.
- It has no access to your medical record, vitals, or prior labs.
- It does **not** replace examination, imaging, or diagnostic testing.
- Eval scores (see `evals/`) measure surface behaviours (structure, keyword presence, judge ratings) — *not* clinical accuracy. Treat all green checkmarks as "didn't crash on these 21 cases," not "is correct."
- Cross-provider judge (Gemini) is recommended over a single Groq judge to mitigate self-grading bias.
- Intake form is fixed — adaptive follow-ups are handled by the LLM via chat, but the form itself does not branch.
- Every interaction sends your symptom data to a third-party LLM provider (Groq or Google). Do not enter information you would not type into a search engine.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open <http://localhost:8501>.

### API keys

Recommended: a `.env` file in the project root.

```bash
cp .env.example .env
# edit .env and set GROQ_API_KEY=... (and optionally GEMINI_API_KEY=...)
```

Both keys are auto-loaded by the app and the eval runners. If `.env` is absent, you can also paste a key directly in the sidebar.

Free keys:
- **Groq** (recommended, fastest): <https://console.groq.com/keys>
- **Google Gemini** (cross-provider judge): <https://aistudio.google.com/apikey>

## Evaluation

Three runners under `evals/`:

| Command | What it does |
|---|---|
| `python -m evals.run` | Rule-based regex + section-scoped assertions over 21 cases (incl. adversarial) |
| `python -m evals.judge` | LLM-as-judge (Groq or Gemini) scores 5 criteria 1–5 |
| `python -m evals.agreement` | Runs Groq + Gemini judges side-by-side; reports inter-judge agreement |

Defaults:
- Agent temperature pinned to **0.0** for evals (deterministic, no flake).
- Set `EVAL_RUNS=3` to run each case 3× and report pass rate (catches residual flake).
- Set `JUDGE_PROVIDER=gemini` to use the Gemini judge instead of Groq.

See [`evals/README.md`](./evals/README.md) for details.

## Deploy

Not currently deployed anywhere — by design. Run on `localhost` only.

## License

[MIT](./LICENSE).
