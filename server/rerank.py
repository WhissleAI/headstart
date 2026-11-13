"""Second-stage ranking for questions about the local moments corpus (route == "local").

The first stage is cosine similarity in EmbeddingGemma's space. On this corpus its scores are compressed (rank 1 and
rank 2 often differ by < 0.02, index/README.md), and the moments' words are noisy ASR ("Ponto ball" for Lonzo Ball), so
a cheap second stage fixes most rank-1 misses. Three signals, added on the cosine scale:

  score = text + W_LEX * min(lexical, LEX_CAP) + W_AUDIO * audio

  text   : cosine of the question's text vector with each moment's text vector (what rank() always did)
  audio  : cosine of the spoken question's AUDIO vector (the audio lane) with max(moment text, moment audio) vectors
  lexical: IDF-weighted overlap of the question's content words with the moment's words + its tags (Arc / gold
           entities: the tag "Lonzo Ball" survives the ASR's "Ponto ball"), plus small boosts when the question's cue
           words name the moment's role ("the judge" -> role judge) or event ("objection" -> legal.objection.*).

The routing cue itself ("in the hearing", "the announcer") is stripped from the lexical query: it names the recording,
not the moment, and every moment of that recording would match it.

The weights are small on purpose: they break the near-ties of the compressed cosine, they don't overrule a clear text
match. Chosen on index/eval_local.py (20 questions x {clean text, Arc's ASR text}); reciprocal-rank fusion was tried
first and lost: it let a full lexical match outvote a confident text rank 1 (local11, local15).

Everything is numpy over <= a few hundred moments: well under 1 ms per question.
"""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

from .frame import LOCAL_CUES, STOP, norm

W_LEX = 0.05          # lexical (share of the query's IDF found in the moment's words + tags, 0..1, + bonuses)
LEX_CAP = 1.3
W_AUDIO = 0.2         # the audio lane's spoken-question cosine (Head Start only: baseline has no audio vector)
ROLE_BONUS = 0.15     # added to the lexical score for a role / event match

_NUM = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7",
        "eight": "8", "nine": "9", "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14",
        "fifteen": "15", "sixteen": "16", "seventeen": "17", "eighteen": "18", "nineteen": "19", "twenty": "20",
        "thirty": "30", "forty": "40", "fifty": "50", "hundred": "100", "first": "1", "second": "2", "third": "3",
        "fifth": "5", "ninth": "9", "twelfth": "12", "fifteenth": "15"}
# words of the question that only say WHICH recording ("the commentator", "in the game"): not moment content
CUE_WORDS = {"game", "announcer", "announcers", "commentator", "commentators", "clip", "recording", "video",
             "meeting", "call", "hearing", "replay", "broadcast", "commentary", "podcast", "moment", "host", "panel"}
# question word -> moment roles it points at (meta.role)
ROLE_CUES = {"judge": {"judge"}, "justice": {"judge"}, "court": set(), "lawyer": {"counsel"}, "attorney": {"counsel"},
             "counsel": {"counsel"}, "announcer": {"play_by_play"}, "commentator": {"play_by_play", "analyst"},
             "analyst": {"analyst"}}
# question word -> event tag prefix (meta.tags)
EVENT_CUES = {"objection": "legal.objection", "objected": "legal.objection", "object": "legal.objection",
              "three": "sports.score", "triple": "sports.score", "basket": "sports.score", "bucket": "sports.score",
              "scored": "sports.score", "score": "sports.score", "dunk": "sports.score",
              "agree": "meeting.decision", "agreed": "meeting.decision", "decided": "meeting.decision",
              "decision": "meeting.decision", "deal": "meeting.decision"}
_SPLIT = re.compile(r"[^\w%']+")
EXTRA_STOP = {"many", "much", "percent", "talk", "talked", "talking", "mention", "mentioned"}


def stem(w: str) -> str:
    w = w.replace("'", "")
    w = _NUM.get(w, w)
    if w.isdigit() or len(w) <= 3:
        return w
    for suf in ("ing", "ed", "es", "s"):
        if suf == "s" and w[-2:] in ("ss", "is", "us"):
            continue
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: -len(suf)]
    return w


def toks(s: str) -> list[str]:
    out = []
    for w in _SPLIT.split((s or "").lower().replace("’", "'")):
        w = w.strip("'_")
        if w.endswith("'s"):
            w = w[:-2]
        if not w:
            continue
        if w.endswith("%") and w[:-1].isdigit():      # "6%" -> "6" (and "percent" is a stopword-ish filler)
            w = w[:-1]
        out.append(w)
    return out


def strip_cues(text: str) -> str:
    """The question without its routing cue phrases ("in the hearing", "the announcer")."""
    low = " " + norm(text) + " "
    for c in sorted(LOCAL_CUES, key=len, reverse=True):
        low = low.replace(" " + c + " ", " ")
    return " ".join(low.split())


def query_terms(text: str) -> list[str]:
    seen, out = set(), []
    for w in toks(strip_cues(text)):
        if w in STOP or w in CUE_WORDS or w in EXTRA_STOP:
            continue
        s = stem(w)
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


class Lexicon:
    """Per-moment term sets (words + tags) and IDF over the corpus."""

    def __init__(self, docs):
        self.terms: dict[str, set] = {}
        self.roles: dict[str, str] = {}
        self.events: dict[str, set] = {}
        df = Counter()
        for d in docs:
            meta = d.meta or {}
            tags = [t for t in meta.get("tags", []) if isinstance(t, str)]
            words = toks(d.text) + [w for t in tags if "." not in t or " " in t for w in toks(t)]
            ts = {stem(w) for w in words if w not in STOP}
            self.terms[d.id] = ts
            self.roles[d.id] = meta.get("role") or ""
            self.events[d.id] = {t.rsplit(".", 1)[0] for t in tags if re.fullmatch(r"[a-z_]+(\.[a-z_]+)+", t)}
            df.update(ts)
        n = max(1, len(docs))
        self.idf = {w: math.log(1 + n / c) for w, c in df.items()}
        self.idf_unseen = math.log(1 + n)

    def idf_of(self, w: str) -> float:
        return self.idf.get(w, self.idf_unseen)

    def score(self, doc_id: str, terms: list[str], roles: set, events: set) -> float:
        ts = self.terms.get(doc_id)
        if ts is None or not terms:
            return 0.0
        tot = sum(self.idf_of(w) for w in terms)
        s = sum(self.idf_of(w) for w in terms if w in ts) / tot
        if roles and self.roles.get(doc_id) in roles:
            s += ROLE_BONUS
        if events and any(e.startswith(p) for e in self.events.get(doc_id, ()) for p in events):
            s += ROLE_BONUS
        return s


def cues(text: str) -> tuple[set, set]:
    """Roles and event prefixes the raw question points at."""
    ws = set(toks(text))
    roles, events = set(), set()
    for w in ws:
        roles |= ROLE_CUES.get(w, set())
        if w in EVENT_CUES:
            events.add(EVENT_CUES[w])
    return roles, events


_lex_cache: dict = {}


def lexicon(index) -> Lexicon:
    key = (id(index), len(index.docs))
    lx = _lex_cache.get(key)
    if lx is None:
        _lex_cache.clear()
        lx = _lex_cache[key] = Lexicon(index.docs)
    return lx


def fuse(ids: list[str], text_s, question: str, lex: Lexicon, audio_s=None,
         w_audio: float | None = None, w_lex: float | None = None) -> list[tuple]:
    """[(id, fused score, parts)] best first. text_s / audio_s are cosines aligned with ids (audio_s may be None)."""
    if not ids:
        return []
    w_audio = W_AUDIO if w_audio is None else w_audio
    w_lex = W_LEX if w_lex is None else w_lex
    text_s = np.asarray(text_s, dtype=np.float64)
    terms = query_terms(question)
    roles, events = cues(question)
    lx = np.array([lex.score(i, terms, roles, events) for i in ids]) if w_lex > 0 else np.zeros(len(ids))
    s = text_s + w_lex * np.minimum(lx, LEX_CAP)
    if audio_s is not None and w_audio > 0:
        s = s + w_audio * np.asarray(audio_s, dtype=np.float64)
    order = np.argsort(-s, kind="stable")
    rt = {int(j): r + 1 for r, j in enumerate(np.argsort(-text_s, kind="stable"))}
    out = []
    for j in order:
        j = int(j)
        parts = {"text": float(text_s[j]), "lex": round(float(lx[j]), 3), "text_rank": rt[j]}
        if audio_s is not None:
            parts["audio"] = float(audio_s[j])
        out.append((ids[j], float(s[j]), parts))
    return out


def audio_scores(index, ids: list[str], qa: np.ndarray) -> np.ndarray:
    """max(text, audio) cosine of an audio query vector for moments of the index (as LocalIndex.search_vec)."""
    out = np.full(len(ids), -1.0)
    for j, i in enumerate(ids):
        r = index.row_of(i)
        if r is None:
            continue
        s = float(index.tv[r] @ qa)
        ar = index.audio_row_of(r) if hasattr(index, "audio_row_of") else None
        if ar is not None:
            s = max(s, float(index.av[ar] @ qa))
        out[j] = s
    return out
