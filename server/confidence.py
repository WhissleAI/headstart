"""Confidence fusion: one calibrated confidence per word, per slot and per query, and a learned end-of-turn score.

Inputs (what a Session knows at each ear update):
  ear       per-word confidence from the ear (Deepgram Flux / nova-3; Arc's builtin transcript has none)
  history   how often the word at this position changed, and how many updates it has held (WordTrack)
  tagger    Arc Tagger spans covering the word: score vs its thr, ver (span verification), alt (the runner-up type)
  arc       Arc audio names / numbers (link state, span_p) when the audio model reports them
Outputs:
  word_conf[i]      P(word i survives unchanged into the final transcript); logistic model fitted on the 80 real Flux
                    traces (eval/fit_confidence.py -> server/conf_model.json)
  slot_conf         per typed slot: min of its words' confidence x its type confidence (score vs alt)
  key_conf          the query's decisive words (slots + content words): drives speculation (server/speculator.py)
  eot_score(...)    P(the speaker is done) from Flux end_of_turn_confidence, tagger command_complete / act / slot-end,
                    Arc audio cc_p / for-me, pause since the last word, word / slot completeness (FLUX_SOFT=fused)
All functions are cheap (< 0.1 ms per update) and pure; the model weights live in server/conf_model.json.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent / "conf_model.json"
_model: dict | None = None

# Fallback weights (used only if conf_model.json is missing): a hand rule that orders words sensibly.
_DEFAULT = {
    "word": {"features": ["bias", "ear", "ear_missing", "revs", "age", "is_last", "in_span", "span_margin", "span_ver"],
             "w": [-1.0, 3.0, 2.0, -0.6, 0.4, -1.0, 0.0, 0.5, 0.5], "mu": None, "sd": None},
    "eot": None,
}


def model() -> dict:
    global _model
    if _model is None:
        try:
            _model = json.loads(MODEL_PATH.read_text())
        except (OSError, ValueError):
            _model = dict(_DEFAULT)
    return _model


def reload(m: dict | None = None):
    """Tests / fitting: install a model dict (or re-read the JSON)."""
    global _model
    _model = m
    if m is None:
        model()


def _sig(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _lin(spec: dict, feats: dict) -> float:
    names, w = spec["features"], spec["w"]
    mu, sd = spec.get("mu"), spec.get("sd")
    z = 0.0
    for k, name in enumerate(names):
        x = 1.0 if name == "bias" else float(feats.get(name, 0.0) or 0.0)
        if mu is not None and name != "bias":
            x = (x - mu[k]) / (sd[k] or 1.0)
        z += w[k] * x
    return z


# ---------------------------------------------------------------------------------------------- revision history
class WordTrack:
    """Revision history of one utterance's words: per position, how many times the word changed and how many updates
    it has held since it last changed. Words are compared normalized (Flux re-cases / re-punctuates freely)."""

    def __init__(self):
        self.words: list[str] = []
        self.revs: list[int] = []
        self.age: list[int] = []

    def reset(self):
        self.__init__()

    def update(self, words: list[str]) -> dict:
        from .frame import norm
        new = [norm(w) for w in words]
        revs, age = [], []
        for i, w in enumerate(new):
            if i < len(self.words):
                if w == self.words[i]:
                    revs.append(self.revs[i])
                    age.append(self.age[i] + 1)
                else:
                    revs.append(self.revs[i] + 1)
                    age.append(0)
            else:
                revs.append(0)
                age.append(0)
        # a shrunk utterance (words dropped at the end) keeps nothing of the dropped positions
        self.words, self.revs, self.age = new, revs, age
        return {"revs": list(revs), "age": list(age)}


# ---------------------------------------------------------------------------------------------- per-word / per-slot
_SLOT_FAMILIES = ("entity", "number")


def _covering(spans: list[dict], i: int) -> dict | None:
    """The most confident entity / number span covering word i (generic entity.numeral last)."""
    best = None
    for s in spans:
        if s.get("family") not in _SLOT_FAMILIES or not (s["start"] <= i < s["end"]):
            continue
        key = (s["type"] != "entity.numeral", float(s.get("score") or 0))
        if best is None or key > best[0]:
            best = (key, s)
    return best[1] if best else None


def type_conf(s: dict) -> float:
    """How sure the tagger is of a span's TYPE: its score against the runner-up type (alt), in [0, 1]."""
    sc = float(s.get("score") or 0.0)
    alt = float(((s.get("alt") or {}).get("p")) or 0.0)
    return sc / (sc + alt) if sc + alt > 0 else 0.0


def word_features(i: int, n: int, ear: float | None, revs: int, age: int, spans: list[dict]) -> dict:
    s = _covering(spans, i)
    margin = 0.0
    ver = 0.0
    if s is not None:
        margin = float(s.get("score") or 0) - float(s.get("thr") or 0)
        ver = float(s.get("ver") or 0)
    return {"ear": 0.0 if ear is None else float(ear), "ear_missing": 1.0 if ear is None else 0.0,
            "revs": float(min(revs, 4)), "age": float(min(age, 6)), "is_last": 1.0 if i == n - 1 else 0.0,
            "in_span": 1.0 if s is not None else 0.0, "span_margin": margin, "span_ver": ver}


def word_confs(words: list[str], ears: list, revs: list[int], age: list[int], spans: list[dict]) -> list[float]:
    spec = model()["word"]
    n = len(words)
    out = []
    for i in range(n):
        e = ears[i] if i < len(ears) else None
        r = revs[i] if i < len(revs) else 0
        a = age[i] if i < len(age) else 0
        out.append(_sig(_lin(spec, word_features(i, n, e, r, a, spans))))
    return out


def arc_name_boost(arc: dict | None, word_t: tuple | None) -> float:
    """Arc audio's own name / number read (ARC_META) for a word, when it has one: a LINKED name (matched to a known
    entity) is strong evidence the word is right; 'not_in_list' / no read is neutral. On this setup Arc reports names
    only when the utterance closes (eval/conf_features.py, PROGRESS.md), so mid-utterance this is almost always 0."""
    if not arc or not word_t or word_t[0] is None:
        return 0.0
    for n in (arc.get("names") or []) + (arc.get("numbers") or []):
        if n.get("start") is None or n.get("end") is None:
            continue
        if n["start"] - 0.05 <= word_t[0] < n["end"] + 0.05 and n.get("state") in ("linked", "in_list", "match"):
            return 0.5
    return 0.0


def slot_confs(f, wc: list[float]) -> dict:
    """{(type, text): conf} for the frame's typed slots: min word conf over the span x type confidence."""
    out = {}
    for s in f.spans:
        if s.get("family") not in _SLOT_FAMILIES or s["type"] == "entity.numeral" or s["end"] > len(wc):
            continue
        from .frame import norm
        key = (s["type"].split(".")[-1], norm(s.get("text") or ""))
        if key not in set(f.slots):
            continue
        c = min(wc[s["start"]:s["end"]]) * (0.5 + 0.5 * type_conf(s))
        out[key] = max(out.get(key, 0.0), c)
    return out


def key_conf(f, wc: list[float]) -> tuple[float, int | None]:
    """Confidence of the query's decisive words: the lowest fused confidence among its content words (the words that
    go into the search). Returns (conf, index of the weakest word)."""
    idx = [i for i in getattr(f, "content_idx", ()) if i < len(wc)]
    if not idx:
        return (min(wc) if wc else 0.0), (int(min(range(len(wc)), key=lambda i: wc[i])) if wc else None)
    j = min(idx, key=lambda i: wc[i])
    return wc[j], j


# ---------------------------------------------------------------------------------------------- end of turn
def eot_features(eot: float, pause_s: float, n_words: int, cc_last: bool, cc_score: float, gate: bool,
                 act_question: bool, slot_end: bool, has_content: bool, last_content: bool,
                 arc_cc_p: float | None, arc_cc_recent: bool, arc_for_me: float | None, is_eager: bool,
                 last_conf: float | None = None) -> dict:
    return {"eot": eot, "pause": min(max(pause_s, 0.0), 1.5), "nw": min(n_words, 15) / 15.0,
            "cc_last": float(cc_last), "cc_score": cc_score, "gate": float(gate), "act_q": float(act_question),
            "slot_end": float(slot_end), "content": float(has_content), "last_content": float(last_content),
            "arc_cc_p": 0.0 if arc_cc_p is None else float(arc_cc_p), "arc_cc_missing": float(arc_cc_p is None),
            "arc_cc_recent": float(arc_cc_recent), "arc_for_me": 0.0 if arc_for_me is None else float(arc_for_me),
            "eager": float(is_eager), "last_conf": 1.0 if last_conf is None else float(last_conf)}


def eot_score(feats: dict) -> float | None:
    """P(complete) from the fitted end-of-turn model, or None when no model is installed."""
    spec = model().get("eot")
    if not spec:
        return None
    return _sig(_lin(spec, feats))


def eot_threshold(on: str = "eager") -> float:
    """The fitted operating threshold: on eager events, and (optionally) on plain updates."""
    spec = model().get("eot") or {}
    return float(spec.get("thr_" + on, spec.get("thr", 0.5)))
