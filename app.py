"""
Dr. Claude — Medical Triage Agent (BYOK)
Structured intake form → assessment → conversational follow-up.
"""

import streamlit as st

from prompts import SYSTEM_PROMPT

st.set_page_config(
    page_title="Dr. Claude — Medical Triage Agent",
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
    "Fever",
    "Nausea",
    "Vomiting",
    "Headache",
    "Dizziness",
    "Fatigue / tiredness",
    "Cough",
    "Cold / runny nose",
    "Sore throat",
    "Shortness of breath",
    "Chest pain",
    "Stomach pain",
    "Diarrhea",
    "Constipation",
    "Joint pain",
    "Muscle aches",
    "Back pain",
    "Neck pain",
    "Sleep issues",
    "Loss of appetite",
    "Weight loss / gain",
    "Numbness or tingling",
    "Vision changes",
    "Hearing changes",
    "Skin rash",
    "Anxiety / low mood",
]

KNOWN_CONDITIONS = [
    "Diabetes",
    "Hypertension (high BP)",
    "Thyroid disorder",
    "Asthma / COPD",
    "Heart disease",
    "Kidney disease",
    "Liver disease",
    "Cancer (current or past)",
    "Autoimmune disorder",
    "Mental-health condition",
    "PCOS / PCOD",
    "Pregnancy",
    "Allergies (specify in notes)",
]


def compose_intake_prompt(d):
    """Turn the form's structured data into a medical-style intake summary."""
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
    if d.get("associated"):
        parts.append(f"**Associated symptoms:** {', '.join(d['associated'])}")
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
# Sidebar — BYOK + reset                                                      #
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.title("🩺 Dr. Claude")
    st.caption("Medical triage agent — BYOK")

    st.markdown("### 🔑 API Configuration")
    provider = st.selectbox(
        "Provider",
        ["Groq (recommended)", "Google Gemini"],
        help="Both providers offer generous free tiers.",
    )

    if provider.startswith("Groq"):
        api_key = st.text_input(
            "Groq API Key",
            type="password",
            placeholder="gsk_...",
            help="Get a free key at console.groq.com/keys",
        )
        model = st.selectbox(
            "Model",
            [
                "llama-3.3-70b-versatile",
                "llama-3.1-8b-instant",
                "openai/gpt-oss-120b",
            ],
            index=0,
        )
        get_key_url = "https://console.groq.com/keys"
    else:
        api_key = st.text_input(
            "Google Gemini API Key",
            type="password",
            placeholder="AIza...",
            help="Get a free key at aistudio.google.com/apikey",
        )
        model = st.selectbox(
            "Model",
            ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"],
            index=0,
        )
        get_key_url = "https://aistudio.google.com/apikey"

    st.markdown(f"[Get a free API key →]({get_key_url})")

    st.divider()
    if st.button("🔄 New consultation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.intake = None
        st.rerun()

    st.divider()
    st.caption(
        "⚠️ For informational triage only — not a substitute for a qualified "
        "physician. In an emergency, call your local emergency number."
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
# Main UI                                                                     #
# --------------------------------------------------------------------------- #
st.title("🩺 Dr. Claude")
st.caption("Structured medical triage — symptom intake → differential & next steps")


# ── Intake form (shown until first assessment is generated) ───────────────── #
if not st.session_state.messages:
    st.markdown(
        "Tell me about your symptoms below. Required fields are marked with **\\***."
    )

    with st.form("intake_form", clear_on_submit=False):
        st.subheader("🩻 Symptom Intake")

        main_symptom = st.text_input(
            "Main symptom *",
            placeholder="e.g., headache, dizziness, chest pain, stomach ache",
        )

        col1, col2 = st.columns(2)
        with col1:
            duration = st.selectbox("Duration *", DURATION_OPTIONS, index=1)
        with col2:
            severity = st.slider(
                "Severity (1 = mild, 10 = unbearable) *", 1, 10, 5
            )

        col3, col4 = st.columns([1, 2])
        with col3:
            age = st.number_input("Age *", min_value=0, max_value=120, value=30)
        with col4:
            sex = st.radio(
                "Sex *",
                ["Male", "Female", "Other / Prefer not to say"],
                horizontal=True,
            )

        col5, col6 = st.columns(2)
        with col5:
            triggers = st.text_input(
                "What makes it worse?",
                placeholder="e.g., sitting at desk, after meals",
            )
        with col6:
            relievers = st.text_input(
                "What makes it better?",
                placeholder="e.g., rest, hot water, lying down",
            )

        associated = st.multiselect(
            "Other symptoms you're experiencing",
            ASSOCIATED_SYMPTOMS,
            help="Pick any that apply — leave empty if none.",
        )

        with st.expander("➕ More details (optional but helpful)"):
            conditions = st.multiselect(
                "Known medical conditions", KNOWN_CONDITIONS
            )
            medications = st.text_input(
                "Current medications",
                placeholder="e.g., metformin 500mg, atorvastatin",
            )
            family_history = st.text_input(
                "Relevant family history",
                placeholder="e.g., mother has hypertension; father had diabetes",
            )
            notes = st.text_area(
                "Anything else worth mentioning?",
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
            st.error(
                f"Please add your {provider.split(' ')[0]} API key in the sidebar."
            )
            st.stop()

        intake = {
            "main_symptom": main_symptom.strip(),
            "duration": duration,
            "severity": severity,
            "age": age,
            "sex": sex,
            "triggers": triggers.strip(),
            "relievers": relievers.strip(),
            "associated": associated,
            "conditions": conditions,
            "medications": medications.strip(),
            "family_history": family_history.strip(),
            "notes": notes.strip(),
        }
        st.session_state.intake = intake
        st.session_state.messages.append(
            {"role": "user", "content": compose_intake_prompt(intake)}
        )
        st.rerun()

# ── Conversation view (after intake submitted) ────────────────────────────── #
else:
    for msg in st.session_state.messages:
        avatar = "🩺" if msg["role"] == "assistant" else "🧑"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    # If the last message is from the user, the assistant still needs to reply.
    if st.session_state.messages[-1]["role"] == "user":
        if not api_key:
            st.error(
                f"Please add your {provider.split(' ')[0]} API key in the sidebar."
            )
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
        st.session_state.messages.append({"role": "user", "content": follow_up})
        st.rerun()
