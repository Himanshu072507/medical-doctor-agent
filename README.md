# 🩺 Dr. Claude — Medical Triage Agent

A structured medical triage chat agent built with Streamlit. Users bring their own Groq or Google Gemini API key (BYOK) — no server-side keys, nothing logged.

## What it does

Given a symptom description, the agent walks through:

1. **Understanding the symptom** (physiology)
2. **Differential diagnosis** (most likely → red-flag)
3. **Working diagnosis**
4. **Recommended evaluation** (bedside → labs → imaging → specialist)
5. **Self-care**
6. **Red-flag triggers**
7. **Disclaimer**

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501), paste your API key in the sidebar, and start chatting.

### Get a free API key

- **Groq** (recommended, fastest): <https://console.groq.com/keys>
- **Google Gemini**: <https://aistudio.google.com/apikey>

## Deploy to Streamlit Cloud

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Select this repository and `app.py` as the entrypoint.
4. Click **Deploy**. No secrets needed — users paste their own key.

## Disclaimer

This tool is informational only. It is **not** a substitute for an in-person clinical examination or qualified medical advice. Seek emergency care for any red-flag symptom.
