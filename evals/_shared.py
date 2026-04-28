"""Shared utilities for the eval runners (rule-based and LLM-judge)."""

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


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


def with_retry(fn, max_attempts=4):
    """Retry an LLM call on Groq 429 (rate limit) with exponential backoff."""
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as e:
            err = str(e)
            is_rate_limit = "429" in err or "rate_limit" in err.lower()
            if not is_rate_limit or attempt == max_attempts - 1:
                raise
            wait = 2 ** attempt * 5  # 5s, 10s, 20s, 40s
            print(f"\n      ↳ rate limit, retrying in {wait}s…", end="", flush=True)
            time.sleep(wait)
