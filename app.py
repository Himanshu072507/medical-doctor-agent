"""
Symptom Triage Helper (BYOK)
A localhost-only informational triage assistant. Not a doctor, not a substitute for one.
"""

import os
import re

import streamlit as st
from dotenv import load_dotenv

from prompts import SYSTEM_PROMPT

load_dotenv()

st.set_page_config(
    page_title="Symptom Triage Helper",
    page_icon="🩺",
    layout="centered",
)

# --------------------------------------------------------------------------- #
# Form options                                                                #
# --------------------------------------------------------------------------- #
DURATION_OPTIONS = [
    "Less than 1 day",
    "1–3 days",
    "4–7 days",
    "1–4 weeks",
    "1–3 months",
    "3–6 months",
    "More than 6 months",
]

ASSOCIATED_SYMPTOMS = [
    "Fever", "Nausea", "Vomiting", "Headache", "Dizziness", "Fatigue / tiredness",
    "Cough", "Cold / runny nose", "Sore throat", "Shortness of breath",
    "Chest pain", "Stomach pain", "Diarrhea", "Constipation",
    "Joint pain", "Muscle aches", "Back pain", "Neck pain",
    "Sleep issues", "Loss of appetite", "Weight loss / gain",
    "Numbness or tingling", "Vision changes", "Hearing changes",
    "Skin rash", "Anxiety / low mood",
]

KNOWN_CONDITIONS = [
    "Diabetes", "Hypertension (high BP)", "Thyroid disorder", "Asthma / COPD",
    "Heart disease", "Kidney disease", "Liver disease", "Cancer (current or past)",
    "Autoimmune disorder", "Mental-health condition", "PCOS / PCOD",
    "Pregnancy", "Allergies (specify in notes)",
]

SEVERITY_LABELS = {
    1: "1 — barely notice",
    3: "3 — mild, doesn't disrupt activity",
    5: "5 — moderate, interferes with normal activity",
    7: "7 — severe, hard to function",
    9: "9 — very severe, can't function",
    10: "10 — worst imaginable",
}

# --------------------------------------------------------------------------- #
# Crisis guardrail — runs BEFORE any LLM call                                 #
# --------------------------------------------------------------------------- #
CRISIS_PATTERN = re.compile(
    r"(?:\bsuicid\w*\b"
    r"|\bkill\s+(?:myself|my\s*self)\b"
    r"|\bend\s+(?:my\s+)?life\b"
    r"|\bdon'?t\s+want\s+to\s+(?:live|be\s+alive)\b"
    r"|\bwant\s+to\s+die\b"
    r"|\bwish\s+(?:i\s+was|i\s+were)\s+dead\b"
    r"|\bbetter\s+off\s+dead\b"
    r"|\bhurt\s+myself\b"
    r"|\bcut(?:ting)?\s+myself\b"
    r"|\bself[\s.\-]?harm\w*\b)",
    re.IGNORECASE,
)

CRISIS_RESPONSE = """🚨 **Please reach out for help right now — you are not alone.**

If you're having thoughts of suicide or self-harm, please contact a crisis helpline immediately:

- 🇮🇳 **India**
  - **Vandrevala Foundation:** 1860-2662-345 (24/7)
  - **iCall:** +91 9152987821 (Mon–Sat, 8 AM–10 PM)
  - **AASRA:** +91 9820466726 (24/7)
- 🇺🇸 **USA:** **988** Suicide & Crisis Lifeline (call or text)
- 🇬🇧 **UK:** Samaritans **116 123** (24/7)
- 🌍 **Global directory:** https://findahelpline.com

If you are in immediate danger, please call your local emergency number now.

A trained human counselor can listen and help — please reach out. Your life matters."""


def is_crisis(text: str) -> bool:
    return bool(CRISIS_PATTERN.search(text or ""))


# --------------------------------------------------------------------------- #
# Intake → prompt                                                             #
# --------------------------------------------------------------------------- #
def compose_intake_prompt(d: dict) -> str:
    parts = [
        f"**Main complaint:** {d['main_symptom']}",
        f"**Duration:** {d['duration']}",
        f"**Severity:** {d['severity']}/10",
        f"**Patient:** {d['age']}-year-old {d['sex'].lower()}",
    ]
    if d.get("triggers"):
        parts.append(f"**Worsens with:** {d['triggers']}")
    if d.get("relievers"):
        parts.append(f"**Relieved by:** {d['relievers']}")
    associated = list(d.get("associated") or [])
    if d.get("associated_other"):
        associated.append(d["associated_other"])
    if associated:
        parts.append(f"**Associated symptoms:** {', '.join(associated)}")
    if d.get("conditions"):
        parts.append(f"**Known conditions:** {', '.join(d['conditions'])}")
    if d.get("medications"):
        parts.append(f"**Current medications:** {d['medications']}")
    if d.get("family_history"):
        parts.append(f"**Family history:** {d['family_history']}")
    if d.get("notes"):
        parts.append(f"**Additional notes:** {d['notes']}")
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Session state                                                               #
# --------------------------------------------------------------------------- #
if "messages" not in st.session_state:
    st.session_state.messages = []
if "intake" not in st.session_state:
    st.session_state.intake = None


# --------------------------------------------------------------------------- #
# Sidebar — provider, key, controls                                           #
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.title("🩺 Symptom Triage Helper")
    st.caption("Informational only — not medical advice")

    st.markdown("### 🔑 API Configuration")
    provider = st.selectbox(
        "Provider",
        ["Groq (recommended)", "Google Gemini"],
        help="Both providers offer generous free tiers.",
    )

    if provider.startswith("Groq"):
        env_key = os.getenv("GROQ_API_KEY", "")
        env_var_name = "GROQ_API_KEY"
        get_key_url = "https://console.groq.com/keys"
        models = [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "openai/gpt-oss-120b",
        ]
        placeholder = "gsk_..."
    else:
        env_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
        env_var_name = "GEMINI_API_KEY"
        get_key_url = "https://aistudio.google.com/apikey"
        models = ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"]
        placeholder = "AIza..."

    if env_key:
        st.success(f"✓ Key loaded from `.env` (`{env_var_name}`)")
        api_key = env_key
    else:
        api_key = st.text_input(
            f"{provider.split(' ')[0]} API Key",
            type="password",
            placeholder=placeholder,
            help=f"Tip: put `{env_var_name}=...` in `.env` to skip this field.",
        )

    model = st.selectbox("Model", models, index=0)
    st.markdown(f"[Get a free API key →]({get_key_url})")

    st.divider()
    if st.button("🔄 New consultation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.intake = None
        st.rerun()

    if st.session_state.intake and st.session_state.messages:
        if st.button("✏️ Edit my inputs", use_container_width=True):
            st.session_state.messages = []  # keep intake for prefill
            st.rerun()

    st.divider()
    st.caption(
        "⚠️ For informational triage only — not a substitute for a qualified physician. "
        "In a medical emergency, call your local emergency number."
    )


# --------------------------------------------------------------------------- #
# Streaming clients                                                           #
# --------------------------------------------------------------------------- #
def stream_groq(messages, key, model_name):
    from groq import Groq

    client = Groq(api_key=key)
    stream = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
        temperature=0.4,
        max_tokens=2048,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def stream_gemini(messages, key, model_name):
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=key)
    contents = [
        types.Content(
            role="user" if m["role"] == "user" else "model",
            parts=[types.Part.from_text(text=m["content"])],
        )
        for m in messages
    ]
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        temperature=0.4,
        max_output_tokens=2048,
    )
    stream = client.models.generate_content_stream(
        model=model_name, contents=contents, config=config
    )
    for chunk in stream:
        if chunk.text:
            yield chunk.text


def stream_response(messages, key, prov, model_name):
    if prov.startswith("Groq"):
        return stream_groq(messages, key, model_name)
    return stream_gemini(messages, key, model_name)


# --------------------------------------------------------------------------- #
# Prominent disclaimer banner                                                 #
# --------------------------------------------------------------------------- #
st.title("🩺 Symptom Triage Helper")
st.markdown(
    "<div style='padding:0.6rem 0.9rem;border-left:4px solid #d97706;"
    "background:#fef3c7;color:#78350f;border-radius:0.25rem;margin-bottom:1rem;'>"
    "<strong>⚠️ Informational only — not medical advice.</strong> "
    "This tool helps you organize symptoms and decide whether to seek care. "
    "It cannot diagnose you. For emergencies, call your local emergency number."
    "</div>",
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
# Intake form (shown until first assessment is generated)                     #
# --------------------------------------------------------------------------- #
if not st.session_state.messages:
    st.markdown(
        "Tell me about your symptoms below. Required fields are marked with **\\***. "
        "Open **More details** if you have additional context to share."
    )

    intake = st.session_state.intake or {}

    with st.form("intake_form", clear_on_submit=False):
        st.subheader("🩻 Symptom Intake")

        main_symptom = st.text_input(
            "Main symptom *",
            value=intake.get("main_symptom", ""),
            placeholder="e.g., headache, dizziness, chest pain, stomach ache",
        )

        col1, col2 = st.columns(2)
        with col1:
            try:
                duration_idx = DURATION_OPTIONS.index(intake.get("duration", DURATION_OPTIONS[1]))
            except ValueError:
                duration_idx = 1
            duration = st.selectbox("Duration *", DURATION_OPTIONS, index=duration_idx)
        with col2:
            severity = st.slider(
                "Severity *", 1, 10, int(intake.get("severity", 5))
            )
            st.caption(SEVERITY_LABELS.get(severity, ""))

        col3, col4 = st.columns([1, 2])
        with col3:
            age = st.number_input(
                "Age *", min_value=0, max_value=120, value=int(intake.get("age", 30))
            )
        with col4:
            sex_options = ["Male", "Female", "Other / Prefer not to say"]
            try:
                sex_idx = sex_options.index(intake.get("sex", "Male"))
            except ValueError:
                sex_idx = 0
            sex = st.radio("Sex *", sex_options, index=sex_idx, horizontal=True)

        with st.expander("➕ More details (optional but helpful)"):
            col5, col6 = st.columns(2)
            with col5:
                triggers = st.text_input(
                    "What makes it worse?",
                    value=intake.get("triggers", ""),
                    placeholder="e.g., sitting at desk, after meals",
                )
            with col6:
                relievers = st.text_input(
                    "What makes it better?",
                    value=intake.get("relievers", ""),
                    placeholder="e.g., rest, hot water, lying down",
                )

            associated = st.multiselect(
                "Other symptoms you're experiencing",
                ASSOCIATED_SYMPTOMS,
                default=[s for s in intake.get("associated", []) if s in ASSOCIATED_SYMPTOMS],
            )
            associated_other = st.text_input(
                "Other symptoms (free text — describe anything not in the list above)",
                value=intake.get("associated_other", ""),
                placeholder="e.g., burning sensation in chest after eating, ringing in ears",
            )

            conditions = st.multiselect(
                "Known medical conditions",
                KNOWN_CONDITIONS,
                default=[c for c in intake.get("conditions", []) if c in KNOWN_CONDITIONS],
            )
            medications = st.text_input(
                "Current medications",
                value=intake.get("medications", ""),
                placeholder="e.g., metformin, atorvastatin",
            )
            family_history = st.text_input(
                "Relevant family history",
                value=intake.get("family_history", ""),
                placeholder="e.g., mother has hypertension; father had diabetes",
            )
            notes = st.text_area(
                "Anything else worth mentioning?",
                value=intake.get("notes", ""),
                placeholder="Lifestyle, recent travel, stress, sleep, diet…",
                height=80,
            )

        submitted = st.form_submit_button(
            "🩺 Get assessment", type="primary", use_container_width=True
        )

    if submitted:
        if not main_symptom.strip():
            st.error("Please describe your main symptom to continue.")
            st.stop()
        if not api_key:
            st.error(f"Please add your {provider.split(' ')[0]} API key in the sidebar.")
            st.stop()

        new_intake = {
            "main_symptom": main_symptom.strip(),
            "duration": duration,
            "severity": severity,
            "age": age,
            "sex": sex,
            "triggers": triggers.strip(),
            "relievers": relievers.strip(),
            "associated": associated,
            "associated_other": associated_other.strip(),
            "conditions": conditions,
            "medications": medications.strip(),
            "family_history": family_history.strip(),
            "notes": notes.strip(),
        }
        st.session_state.intake = new_intake
        prompt = compose_intake_prompt(new_intake)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Crisis pre-check on the composed intake (covers main symptom + notes etc.)
        if is_crisis(prompt):
            st.session_state.messages.append(
                {"role": "assistant", "content": CRISIS_RESPONSE}
            )
        st.rerun()

# --------------------------------------------------------------------------- #
# Conversation view (after intake submitted)                                  #
# --------------------------------------------------------------------------- #
else:
    for msg in st.session_state.messages:
        avatar = "🩺" if msg["role"] == "assistant" else "🧑"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    # If the last message is from the user, the assistant still needs to reply.
    if st.session_state.messages[-1]["role"] == "user":
        if not api_key:
            st.error(f"Please add your {provider.split(' ')[0]} API key in the sidebar.")
            st.stop()
        with st.chat_message("assistant", avatar="🩺"):
            try:
                stream = stream_response(
                    st.session_state.messages, api_key, provider, model
                )
                full_response = st.write_stream(stream)
                st.session_state.messages.append(
                    {"role": "assistant", "content": full_response}
                )
                st.rerun()
            except Exception as e:
                err = str(e)
                st.error(f"⚠️ {provider} error: {err}")
                if "api_key" in err.lower() or "auth" in err.lower() or "401" in err:
                    st.info("Double-check your API key in the sidebar.")
                st.session_state.messages.pop()

    follow_up = st.chat_input("Ask a follow-up question…")
    if follow_up:
        # Crisis pre-check on follow-up text
        if is_crisis(follow_up):
            st.session_state.messages.append({"role": "user", "content": follow_up})
            st.session_state.messages.append(
                {"role": "assistant", "content": CRISIS_RESPONSE}
            )
        else:
            st.session_state.messages.append({"role": "user", "content": follow_up})
        st.rerun()
