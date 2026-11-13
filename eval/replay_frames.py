"""Offline frame replay: ASR text -> Arc Tagger (word-by-word, as live) -> frame. No audio, no ear."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from server import frame as framemod  # noqa: E402
from server.session import clean  # noqa: E402
from server.tagger import Tagger  # noqa: E402

_T: Tagger | None = None


def tagger() -> Tagger:
    global _T
    if _T is None:
        _T = Tagger(domain="assistant")
    return _T


def frame_for(text: str, carry: dict | None = None) -> framemod.Frame:
    """Feed the words one at a time (the live path pushes partials), then build the final frame."""
    t = tagger()
    t.reset()
    words = [w for w in (clean(x) for x in text.split()) if w]
    spans, current, events, prev_intent = [], None, set(), None
    for i in range(len(words)):
        t.push(words[i:i + 1])
        r = t.poll()
        a0 = r.get("a0", 0)
        spans = [s for s in spans if s["end"] <= a0] + r.get("spans", [])
        if r.get("current") is not None:
            current = r["current"]
        for u in r.get("utterances") or []:
            current = u
        for e in r.get("events") or []:
            events.add(e["type"])
        f = framemod.build(words[:i + 1], spans, current, events, prev_intent, carry=carry)
        if f.intent.startswith("discourse.intent."):
            prev_intent = (f.intent, f.intent_p)
    return framemod.build(words, spans, current, events, prev_intent, carry=carry)


if __name__ == "__main__":
    for line in sys.argv[1:]:
        f = frame_for(line)
        print(repr(line), "->", repr(f.query()), "| ent:", repr(f.entity_query()), "| clean:", repr(f.resolved_text))
