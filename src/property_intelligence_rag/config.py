"""Runtime configuration, read from the environment (and `.env`)."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _env(name: str, default=None):
    """Environment value, falling back to `default` when unset or blank."""
    value = os.getenv(name, "").strip()
    return value or default


ROOT = Path(__file__).resolve().parents[2]
# Vercel functions can only write to /tmp; the index and cache are rebuilt per instance there.
WRITABLE_DIR = Path("/tmp/agent-container") if os.getenv("VERCEL") else ROOT / "data"
RAW_DIR = Path(_env("PI_RAW_DIR", ROOT / "data" / "raw"))
INDEX_PATH = Path(_env("PI_INDEX_PATH", WRITABLE_DIR / "index" / "properties.json"))
PROPERTY_CACHE_PATH = Path(
    _env("PI_PROPERTY_CACHE_PATH", WRITABLE_DIR / "property-cache.sqlite3")
)
DEMO_FALLBACK_PATH = Path(
    _env("PI_DEMO_FALLBACK_PATH", ROOT / "data" / "demo_property.json")
)

# Chat and embeddings both go through OpenRouter's OpenAI-compatible API.
DEFAULT_CHAT_MODEL = "anthropic/claude-opus-5.5"
DEFAULT_EMBEDDING_MODEL = "openai/text-embedding-3-small"
OPENROUTER_API_KEY = _env("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = _env("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
# Any OpenRouter model id that supports tool calling, e.g. a ":free" variant.
CHAT_MODEL = _env("PI_CHAT_MODEL", DEFAULT_CHAT_MODEL)
# OpenRouter embedding model id, or "fake" for tests.
EMBEDDING_MODEL = _env("PI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
RETRIEVER_K = int(_env("PI_RETRIEVER_K", "6"))

# Live property data.
APIFY_TOKEN = _env("APIFY_TOKEN")
ZILLOW_ACTOR = _env("PI_ZILLOW_ACTOR", "igolaizola/zillow-scraper-ppe")
PROPERTY_CACHE_TTL_SECONDS = int(_env("PI_PROPERTY_CACHE_TTL_SECONDS", "86400"))
APIFY_TIMEOUT_SECONDS = int(_env("PI_APIFY_TIMEOUT_SECONDS", "120"))
DEMO_FALLBACK = _env("PI_DEMO_FALLBACK", "false").lower() in {"1", "true", "yes"}
