"""Settings: environment (+ headstart/.env) with defaults tuned for a laptop demo."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv():
    p = ROOT / ".env"
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()

ARC_DIR = Path(os.environ.get("ARC_DIR", "/Users/karan/Desktop/work/whissle/models/arc-v1"))
ARC_THREADS = int(os.environ.get("ARC_THREADS", "6"))
EMBED_MODEL = os.environ.get("EMBED_MODEL", "google/embeddinggemma-2")
EMBED_DEVICE = os.environ.get("EMBED_DEVICE", "mps")
EMBED_DIM = int(os.environ.get("EMBED_DIM", "768"))
ANSWER_MODEL = os.environ.get("ANSWER_MODEL", "claude-haiku-5-5")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")
WEB_BACKEND = os.environ.get("WEB_BACKEND", "tavily" if TAVILY_API_KEY else "wikipedia")
INDEX_DIR = Path(os.environ.get("INDEX_DIR", ROOT / "index/data"))

# Tavily web lane (docs.tavily.com/documentation/api-reference/endpoint/search). Credits: ultra-fast/fast/basic = 1,
# advanced = 2. Speculative fires are hidden behind speech, so they take a fast depth; the final (post-endpoint) query
# is on the user's critical path, so it defaults to the same fast depth (and baseline uses it too: same lanes).
TAVILY_BASE_URL = os.environ.get("TAVILY_BASE_URL", "https://api.tavily.com").rstrip("/")
TAVILY_SPEC_DEPTH = os.environ.get("TAVILY_SPEC_DEPTH", "fast")
TAVILY_FINAL_DEPTH = os.environ.get("TAVILY_FINAL_DEPTH", "fast")
TAVILY_MAX_RESULTS = int(os.environ.get("TAVILY_MAX_RESULTS", "5"))
TAVILY_CHUNKS = int(os.environ.get("TAVILY_CHUNKS", "2"))              # 1-3; ignored by ultra-fast
TAVILY_TIMEOUT_S = float(os.environ.get("TAVILY_TIMEOUT_S", "4.0"))
TAVILY_WEATHER_MAX_AGE_H = int(os.environ.get("TAVILY_WEATHER_MAX_AGE_H", "6"))   # 0 = don't send max_age_hours
TAVILY_FINANCE_MAX_AGE_H = int(os.environ.get("TAVILY_FINANCE_MAX_AGE_H", "1"))
TAVILY_SESSION_CREDITS = float(os.environ.get("TAVILY_SESSION_CREDITS", "120"))   # per Session; 0 = unlimited
TAVILY_FINAL_RESERVE = float(os.environ.get("TAVILY_FINAL_RESERVE", "2"))        # speculation can't touch these
TAVILY_PROCESS_CREDITS = float(os.environ.get("TAVILY_PROCESS_CREDITS", "800"))   # whole server; 0 = unlimited
TAVILY_CACHE_TTL_S = float(os.environ.get("TAVILY_CACHE_TTL_S", "900"))
TAVILY_FRESH_TTL_S = float(os.environ.get("TAVILY_FRESH_TTL_S", "120"))         # news / finance / weather
# session: each Session has its own cache (A/B-honest: baseline never reads Head Start's results). global: one cache
# for the server (saves credits on a demo booth replaying the same questions; do NOT use for A/B timing).
TAVILY_CACHE_SCOPE = os.environ.get("TAVILY_CACHE_SCOPE", "session")
# Race Tavily's own include_answer against Haiku on the after-endpoint path (miss + baseline). Off by default.
TAVILY_ANSWER = os.environ.get("TAVILY_ANSWER", "0") == "1"
TAVILY_ANSWER_DEPTH = os.environ.get("TAVILY_ANSWER_DEPTH", "basic")              # include_answer: basic | advanced

# Speculation policy (see speculator.py)
MIN_WORDS = int(os.environ.get("SPEC_MIN_WORDS", "3"))
STABLE_UPDATES = int(os.environ.get("SPEC_STABLE_UPDATES", "1"))
WEB_BUDGET = int(os.environ.get("SPEC_WEB_BUDGET", "6"))
LOCAL_BUDGET = int(os.environ.get("SPEC_LOCAL_BUDGET", "8"))
DRAFT_BUDGET = int(os.environ.get("SPEC_DRAFT_BUDGET", "6"))
DRAFT_STABLE_MS = int(os.environ.get("SPEC_DRAFT_STABLE_MS", "0"))
AUDIO_LANE = os.environ.get("SPEC_AUDIO_LANE", "1") == "1"
COMMIT_TAU = float(os.environ.get("SPEC_COMMIT_TAU", "0.55"))

# Semantic endpointing (Head Start only): answer when the tagger says the command is complete and the speaker has been
# silent this long, instead of waiting for the ear's own pause endpoint (~450 ms on Arc).
SEMANTIC_ENDPOINT = os.environ.get("SEMANTIC_ENDPOINT", "1") == "1"
SEM_SILENCE_MS = int(os.environ.get("SEM_SILENCE_MS", "250"))
SEM_STABLE_MS = int(os.environ.get("SEM_STABLE_MS", "420"))   # one Arc partial refresh (~0.4 s) must confirm the words
SEM_COVER_S = float(os.environ.get("SEM_COVER_S", "0.3"))     # an ASR refresh this long after silence began covers the speech
PENDING_CAP_S = float(os.environ.get("PENDING_CAP_S", "1.0"))   # max wait at the endpoint for an in-flight speculative search
HEDGE_MS = int(os.environ.get("HEDGE_MS", "800"))   # duplicate the answer request if no token by then (both modes)
