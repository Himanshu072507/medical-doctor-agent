"""Shared utilities for the eval runners (rule-based, LLM-judge, agreement)."""

import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Section-heading pattern, e.g. "## 1. Understanding Your Symptoms"
_HEADING_RE = re.compile(r"^##\s+(\d+)\.\s+(.+?)\s*$", re.MULTILINE)


def load_env():
    """Load .env if python-dotenv is available; otherwise rely on shell env."""
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except ImportError:
        pass


def get_groq_client():
    """Return a Groq client, or exit(2) with a friendly error if missing."""
    key = os.getenv("GROQ_API_KEY")
    if not key:
        print("ERROR: GROQ_API_KEY not set.")
        print(f"Add it to {ROOT}/.env or export it in your shell.")
        print("Get a free key at https://console.groq.com/keys")
        sys.exit(2)
    try:
        from groq import Groq
    except ImportError:
        print("ERROR: groq SDK not installed. Run: pip install -r requirements.txt")
        sys.exit(2)
    return Groq(api_key=key)


def get_gemini_client():
    """Return a google-genai client, or exit(2) on missing key/SDK."""
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        print("ERROR: GEMINI_API_KEY not set.")
        print(f"Add it to {ROOT}/.env or export it in your shell.")
        print("Get a free key at https://aistudio.google.com/apikey")
        sys.exit(2)
    try:
        from google import genai
    except ImportError:
        print("ERROR: google-genai SDK not installed. Run: pip install -r requirements.txt")
        sys.exit(2)
    return genai.Client(api_key=key)


def get_ollama_client():
    """Return an ollama Client. Assumes a local Ollama server is running."""
    try:
        import ollama
    except ImportError:
        print("ERROR: ollama SDK not installed. Run: pip install -r requirements.txt")
        sys.exit(2)
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    return ollama.Client(host=host)


def with_retry(fn, max_attempts=4):
    """Retry an LLM call on Groq 429 (rate limit) with exponential backoff."""
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as e:
            err = str(e)
            is_rate_limit = "429" in err or "rate_limit" in err.lower() or "RESOURCE_EXHAUSTED" in err
            if not is_rate_limit or attempt == max_attempts - 1:
                raise
            wait = 2 ** attempt * 5  # 5s, 10s, 20s, 40s
            print(f"\n      ↳ rate limit, retrying in {wait}s…", end="", flush=True)
            time.sleep(wait)


def parse_sections(text: str) -> dict:
    """
    Parse a markdown response into {section_number: content}.
    Recognizes headings of the form '## N. Title'. Missing sections return ''.
    """
    if not text:
        return {}
    matches = list(_HEADING_RE.finditer(text))
    sections: dict = {}
    for i, m in enumerate(matches):
        num = int(m.group(1))
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections[num] = text[start:end].strip()
    return sections


def section(text: str, n: int) -> str:
    """Return the content of section N, or empty string if missing."""
    return parse_sections(text).get(n, "")
