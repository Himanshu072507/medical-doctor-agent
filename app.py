"""
Dr. Claude — Medical Triage Agent (BYOK)
A structured medical-reasoning chat agent. Users bring their own Groq or Gemini API key.
"""

import streamlit as st

from prompts import SYSTEM_PROMPT

st.set_page_config(
    page_title="Dr. Claude — Medical Triage Agent",
    page_icon="🩺",
    layout="centered",
)

WELCOME_MESSAGE = """👋 Hello! I'm **Dr. Claude**, your AI medical triage assistant.

To help me assess you well, please share:

1. **Main symptom** — what are you feeling?
2. **Duration** — when did it start?
3. **Severity** — on a scale of 1–10
4. **Triggers / relievers** — what makes it worse or better?
5. **Associated symptoms** — fever, nausea, weakness, etc.
6. **Background** — age, sex, known conditions, family history

> ⚠️ This is informational guidance only and does not replace an in-person consultation. For emergencies (chest pain, breathlessness, sudden weakness, severe bleeding, loss of consciousness), seek immediate medical attention.
"""

# --------------------------------------------------------------------------- #
# Sidebar — BYOK configuration                                                #
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
        st.rerun()

    st.divider()
    st.caption(
        "⚠️ This agent is for informational triage only and does not replace "
        "a qualified physician. In an emergency, call your local emergency number."
    )


# --------------------------------------------------------------------------- #
# Provider clients                                                            #
# --------------------------------------------------------------------------- #
def stream_groq(messages, key, model_name):
    from groq import Groq

    client = Groq(api_key=key)
    full_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
    stream = client.chat.completions.create(
        model=model_name,
        messages=full_messages,
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


# --------------------------------------------------------------------------- #
# Main chat UI                                                                #
# --------------------------------------------------------------------------- #
st.title("🩺 Dr. Claude")
st.caption("Structured medical triage — symptoms in, differential & next steps out")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Welcome card
if not st.session_state.messages:
    with st.chat_message("assistant", avatar="🩺"):
        st.markdown(WELCOME_MESSAGE)

# Replay history
for msg in st.session_state.messages:
    avatar = "🩺" if msg["role"] == "assistant" else "🧑"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# Chat input
prompt = st.chat_input(
    "Describe your symptoms, e.g. 'headache for 3 days, worsens at desk, nausea'"
)

if prompt:
    if not api_key:
        st.error(
            f"Please add your {provider.split(' ')[0]} API key in the sidebar to continue."
        )
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🩺"):
        try:
            if provider.startswith("Groq"):
                stream = stream_groq(st.session_state.messages, api_key, model)
            else:
                stream = stream_gemini(st.session_state.messages, api_key, model)
            full_response = st.write_stream(stream)
            st.session_state.messages.append(
                {"role": "assistant", "content": full_response}
            )
        except Exception as e:
            err = str(e)
            st.error(f"⚠️ {provider} error: {err}")
            if "api_key" in err.lower() or "auth" in err.lower() or "401" in err:
                st.info("Double-check your API key in the sidebar.")
            st.session_state.messages.pop()  # drop the user msg so they can retry
