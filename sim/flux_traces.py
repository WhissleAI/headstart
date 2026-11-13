"""Flux-trace replay for the policy simulator: the 80 REAL Deepgram Flux traces (runs/flux_traces.jsonl, every TurnInfo
message with its arrival time) as simulator traces, emulating server/ear.FluxEar exactly (words filtered past the last
closed turn, partials only on changed words, on_turn_info / on_turn_tick / on_eager / on_final), plus the audio RMS of
the real wav every 100 ms (the silence detector) and Arc Mini's audio metadata (cc_p, cc event, for-me) as it stood
when each message arrived (runs/conf_features.jsonl, eval/conf_features.py: Arc replayed on the real audio + words).

  .venv/bin/python -m sim.simulate --flux [--prefetch] --set FLUX_SOFT=fused --profile tavily --seeds 3
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FLUX = ROOT / "runs" / "flux_traces.jsonl"
FEAT = ROOT / "runs" / "conf_features.jsonl"


def _meta():
    q = {}
    for p in (ROOT / "eval" / "questions.jsonl", ROOT / "eval" / "showcase" / "questions.jsonl"):
        for line in p.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                q[d["id"]] = d
    return q


def load(ids=None, include_current: bool = True) -> list[dict]:
    import soundfile as sf
    from server.ear import flux_word
    meta = _meta()
    arc = {}
    if FEAT.exists():
        for line in FEAT.read_text().splitlines():
            r = json.loads(line)
            arc[(r["id"], r["i"])] = {"cc_p": r.get("arc_cc_p"), "for_me": r.get("arc_for_me"),
                                      "cc_events": [{"t": r["arc_cc_t"]}] if r.get("arc_cc_t") is not None else []}
    out = []
    for line in FLUX.read_text().splitlines():
        tr = json.loads(line)
        if ids and tr["id"] not in ids:
            continue
        m = meta.get(tr["id"], {})
        if not include_current and m.get("kind") == "current":
            continue
        ev = []
        x, _ = sf.read(str(ROOT / "eval" / "audio" / tr["file"]), dtype="float32")
        if x.ndim > 1:
            x = x.mean(1)
        x = np.concatenate([x, np.zeros(int(1.6 * 16000), np.float32)])
        for i in range(0, len(x), 1600):
            c = x[i:i + 1600]
            ev.append({"k": "audio", "w": (i + 1600) / 16000, "at": (i + 1600) / 16000, "n": len(c),
                       "rms": float(np.sqrt(np.mean(c * c)))})
        for i, msg in enumerate(tr["msgs"]):
            if msg.get("type") != "TurnInfo":
                continue
            ws = [{"text": flux_word(w.get("word") or ""), "start": float(w["start"]), "end": float(w["end"]),
                   "conf": w.get("confidence")} for w in msg.get("words") or [] if w.get("start") is not None]
            ws = [w for w in ws if w["text"]]
            ev.append({"k": "flux", "w": msg["_t"], "ev": msg["event"], "turn": msg["turn_index"],
                       "eot": msg.get("end_of_turn_confidence"), "words": ws, "arc": arc.get((tr["id"], i))})
        ev.sort(key=lambda e: (e["w"], e["k"] != "audio"))
        out.append({"id": tr["id"], "text": tr["text"], "kind": m.get("kind"), "file": tr["file"],
                    "expect": m.get("expect"), "ear": "flux", "events": ev})
    return out


class FluxEmu:
    """server/ear.FluxEar's mapping of TurnInfo messages onto a session (sim side)."""

    def __init__(self, sess, finals: list, create_task):
        self.s, self.finals, self.task = sess, finals, create_task
        self.turn, self.interim, self.boundary, self.key, self.arc = None, [], 0.0, None, None

    def _final(self):
        ws, self.interim, self.key = self.interim, [], None
        if not ws:
            return
        self.boundary = max(self.boundary, ws[-1]["end"])
        self.finals.append(self.task(self.s.on_final(ws)))

    def on(self, e: dict, now: float):
        s = self.s
        if e.get("arc") and e["arc"] != self.arc:
            self.arc = e["arc"]
            s.on_meta(dict(e["arc"]))
        if self.turn is not None and e["turn"] != self.turn and self.interim:
            self._final()
        self.turn = e["turn"]
        self.interim = [w for w in e["words"] if w["end"] > self.boundary + 1e-3]
        s.on_turn_info(e["ev"], e.get("eot"))
        if e["ev"] == "EndOfTurn":
            self._final()
            return
        if self.interim:
            key = tuple(w["text"] for w in self.interim)
            if key != self.key:
                self.key = key
                s.on_partial(list(self.interim), now)
        s.on_turn_tick(e["ev"])
        if e["ev"] == "EagerEndOfTurn" and self.interim:
            s.on_eager(list(self.interim), now)

    def flush(self):
        if self.interim:
            self._final()
