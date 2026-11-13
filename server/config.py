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
