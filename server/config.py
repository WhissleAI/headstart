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
TAVILY_FINANCE_DEPTH = os.environ.get("TAVILY_FINANCE_DEPTH", "basic")   # topic=finance is refused on fast / ultra-fast
TAGGER_HINTS = os.environ.get("TAGGER_HINTS", "1") == "1"   # Tavily topic / freshness from the full Arc read-out
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
# Per-backend defaults from the policy simulator (sim/, validated ±4% vs the real A/B). Env vars always win.
# Wikipedia (~230 ms lane): current settings are within 10 ms of the best found. Tavily (~0.9 s lane): "T2" is
# 20 ms faster at p50, cuts ungrounded answers 4.0% -> 3.3% and answer-model calls by 17% (runs: sim_*; PROGRESS.md).
_TAVILY = WEB_BACKEND == "tavily"
DRAFT_BUDGET = int(os.environ.get("SPEC_DRAFT_BUDGET", "2" if _TAVILY else "6"))
DRAFT_STABLE_MS = int(os.environ.get("SPEC_DRAFT_STABLE_MS", "0"))
AUDIO_LANE = os.environ.get("SPEC_AUDIO_LANE", "1") == "1"
COMMIT_TAU = float(os.environ.get("SPEC_COMMIT_TAU", "0.55"))
# Local-routed questions: rank moments with server/rerank.py (text cosine + the audio lane's spoken-question
# vector + lexical/tags) instead of pool cosine alone. Same ranker in both modes (baseline has no audio vector).
RERANK_LOCAL = os.environ.get("RERANK_LOCAL", "1") == "1"
RERANK_AUDIO = os.environ.get("RERANK_AUDIO", "1") == "1"

# Semantic endpointing (Head Start only): answer when the tagger says the command is complete and the speaker has been
# silent this long, instead of waiting for the ear's own pause endpoint (~450 ms on Arc).
SEMANTIC_ENDPOINT = os.environ.get("SEMANTIC_ENDPOINT", "1") == "1"
SEM_SILENCE_MS = int(os.environ.get("SEM_SILENCE_MS", "200" if _TAVILY else "250"))
SEM_STABLE_MS = int(os.environ.get("SEM_STABLE_MS", "550" if _TAVILY else "420"))   # one Arc partial refresh (~0.4 s) must confirm the words
SEM_COVER_S = float(os.environ.get("SEM_COVER_S", "0.4" if _TAVILY else "0.3"))     # an ASR refresh this long after silence began covers the speech
PENDING_CAP_S = float(os.environ.get("PENDING_CAP_S", "1.0"))   # max wait at the endpoint for an in-flight speculative search
HEDGE_MS = int(os.environ.get("HEDGE_MS", "1000" if _TAVILY else "800"))   # duplicate the answer request if no token by then (both modes)

# Ear (ASR words for the sessions): EAR=arc (Arc v1 Mini built-in transcript, local) | deepgram (Deepgram live websocket;
# needs DEEPGRAM_API_KEY, falls back to Arc without it or when the connection drops). See server/ear.py.
EAR = os.environ.get("EAR", "arc").strip().lower()
DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY", "")
DG_URL = os.environ.get("DG_URL", "wss://api.deepgram.com/v1/listen")
DG_MODEL = os.environ.get("DG_MODEL", "nova-3")
DG_LANGUAGE = os.environ.get("DG_LANGUAGE", "en")
DG_ENDPOINTING_MS = int(os.environ.get("DG_ENDPOINTING_MS", "300"))      # pause that closes an utterance (speech_final)
DG_UTTERANCE_END_MS = int(os.environ.get("DG_UTTERANCE_END_MS", "1000"))  # word-gap backstop (UtteranceEnd); min 1000; 0 = off
DG_WORD_FIELD = os.environ.get("DG_WORD_FIELD", "word")   # "word" (lowercase, like Arc's words) | "punctuated_word"
DG_KEEPALIVE_S = float(os.environ.get("DG_KEEPALIVE_S", "3"))            # Deepgram closes after 10 s without audio/KeepAlive
DG_CONNECT_TIMEOUT_S = float(os.environ.get("DG_CONNECT_TIMEOUT_S", "4"))
DG_FLUSH_TIMEOUT_S = float(os.environ.get("DG_FLUSH_TIMEOUT_S", "1.5"))  # wait for Finalize's result at the end of a run
DG_FALLBACK_REPLAY_S = float(os.environ.get("DG_FALLBACK_REPLAY_S", "10"))   # audio of the open utterance re-fed to Arc
DG_ARC_EVENTS = os.environ.get("DG_ARC_EVENTS", "0") == "1"  # also run Arc on Deepgram's words for its own events (CPU)
SEM_WORDGAP_MS = int(os.environ.get("SEM_WORDGAP_MS", "600"))   # noisy rooms: ASR word gap that counts as a pause
CHATTER_RATIO = float(os.environ.get("CHATTER_RATIO", "0.25"))   # -12 dB vs the last answered question = chatter

# Degraded / offline demo mode (see DEMO_CHECKLIST.md). WEB_BACKEND=cache serves the web lane only from the recorded
# cache; with a live backend the cache is the fallback when a live call fails (flagged "cached" in events, metric, UI).
WEB_CACHE_PATH = Path(os.environ.get("WEB_CACHE_PATH", ROOT / "runs/web_cache.json"))
WEB_CACHE_FALLBACK = os.environ.get("WEB_CACHE_FALLBACK", "1") == "1"   # live failure -> recorded cache
WEB_CACHE_RECORD = os.environ.get("WEB_CACHE_RECORD", "0") == "1"       # write every live result into the cache file
WEB_CACHE_MIN_COVER = float(os.environ.get("WEB_CACHE_MIN_COVER", "0.75"))   # fuzzy hit: share of query words covered
WEB_LIVE_TIMEOUT_S = float(os.environ.get("WEB_LIVE_TIMEOUT_S", "3.0"))  # hard deadline on one Wikipedia call
WEB_BREAKER_S = float(os.environ.get("WEB_BREAKER_S", "30"))     # after a live failure, go straight to the cache this long
# Answer model unreachable -> extractive answer (best-matching sentence of the top source, flagged), both modes.
ANSWER_FALLBACK = os.environ.get("ANSWER_FALLBACK", "1") == "1"
ANSWER_DEADLINE_MS = int(os.environ.get("ANSWER_DEADLINE_MS", "4000"))   # no first token by then -> extractive
ANSWER_BREAKER_S = float(os.environ.get("ANSWER_BREAKER_S", "30"))
ANSWER_MAX_RETRIES = int(os.environ.get("ANSWER_MAX_RETRIES", "2"))     # SDK retries per request (SDK default; the hedge is another)
STALE_CANCEL = os.environ.get("STALE_CANCEL", "0") == "1"   # cancel in-flight searches the speaker revised away
WEB_BREAKER_FAILS = int(os.environ.get("WEB_BREAKER_FAILS", "2"))   # consecutive primary-backend failures before the cache takes over
