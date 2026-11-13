"""Flux eager end-of-turn as the soft endpoint: which extra gate (if any) earns its keep? Measured on REAL Flux traces.

Input: runs/flux_traces.jsonl (eval/record_flux_traces.py: every TurnInfo message from the live Flux API with its
arrival time, eager_eot_threshold 0.3). Each trace is replayed in arrival order through
  - the Arc Tagger exactly as Session feeds it (normalized words, revisions pushed from the first changed word), and
  - optionally Arc Mini audio metadata in external-transcript mode (--arc: real audio + the same words, ~10 ms/chunk).
At every EagerEndOfTurn we ask each policy whether to answer now (soft). An eager is COMPLETE if its words equal the
words of the question's last turn at EndOfTurn (and no later turn adds content); otherwise it is PREMATURE (the speaker
went on, or Flux split the question): a premature soft answer is a wasted answer call and, shown first, a wrong answer.
Per policy:
  trigger_ms   p50 / mean of (time the answer starts - last word end); a question whose complete eager is rejected
               falls back to EndOfTurn (the ear's final)
  premature    soft answers started on premature eagers (per 100 questions)
Policies also emulate higher eager thresholds from end_of_turn_confidence (the eager at 0.3 + conf >= x).

Run: .venv/bin/python eval/flux_gate_eval.py [--arc] [traces.jsonl]
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from server import config, frame  # noqa: E402
from server.ear import flux_word  # noqa: E402
from server.session import CC, clean, gate_open, meaning  # noqa: E402
from server.tagger import Tagger  # noqa: E402


class TagState:
    """Session._sync_words / _apply, minus the speculator."""

    def __init__(self, t: Tagger):
        self.t = t
        self.reset()

    def reset(self):
        self.t.reset()
        self.words, self.spans, self.current, self.cc_at = [], [], None, None

    def sync(self, ws):
        new = [w for w in (clean(x) for x in ws) if w]
        k = 0
        while k < min(len(new), len(self.words)) and new[k] == self.words[k]:
            k += 1
        changed = False
        if k < len(self.words):
            self.t.push(new[k:], from_=k)
            changed = True
        elif len(new) > len(self.words):
            self.t.push(new[k:])
            changed = True
        self.words = new
        if changed:
            if self.cc_at is not None and self.cc_at >= k:
                self.cc_at = None
            r = self.t.poll()
            a0 = r.get("a0", 0)
            self.spans = [s for s in self.spans if s["end"] <= a0] + r.get("spans", [])
            if r.get("current") is not None:
                self.current = r["current"]
            for u in r.get("utterances") or []:
                self.current = u
            for e in r.get("events") or []:
                if e["type"] == CC:
                    w = e.get("word")
                    self.cc_at = w if w is not None else len(self.words) - 1

    def frame(self):
        ev = {CC} if (self.cc_at is not None and self.cc_at == len(self.words) - 1) else set()
        return frame.build(self.words, self.spans, self.current, ev, None, local_ok=False)


def norm_words(m):
    return [w for w in (flux_word(x.get("word") or "") for x in m.get("words") or []) if w]


class ArcReplay:
    def __init__(self):
        from server.ear import arc
        self.a = arc()

    def run(self, tr):
        """Arc Mini metadata state at each message's arrival: [(cc_p, cc_event_t or None, for_me)] by message index."""
        import soundfile as sf
        q = tr["file"]
        x, _ = sf.read(str(Path(getattr(self, "audio_dir", ROOT / "eval" / "audio")) / q), dtype="float32")
        if x.ndim > 1:
            x = x.mean(1)
        x = np.concatenate([x, np.zeros(int(1.6 * 16000), np.float32)])
        s = self.a.stream(domain="assistant", transcript="external", name_source="word", interim_source="deepgram",
                          partials=False, fallback=None)
        msgs = tr["msgs"]
        state = {"cc_p": None, "cc_t": None, "for_me": None}
        snap = {}
        self.cc_emits = []                     # (audio time Arc said it, event t)
        k = 0
        for i in range(0, len(x), 1600):
            now = (i + 1600) / 16000
            while k < len(msgs) and msgs[k]["_t"] <= now:
                m = msgs[k]
                snap[k] = dict(state)          # what Arc had said before this message arrived
                if m.get("type") == "TurnInfo" and m.get("words"):
                    s.add_words([dict(text=w["word"], start=w["start"], end=w["end"],
                                      is_final=m["event"] == "EndOfTurn") for w in m["words"]])
                k += 1
            out = s.feed(x[i:i + 1600])
            u = (out.get("live") or {}).get("utt") or {}
            if u:
                state["cc_p"] = u.get("command_complete")
                state["for_me"] = u.get("device_directed")
            for e in out.get("events") or []:
                if e.get("tag") == CC:
                    state["cc_t"] = e.get("t")
                    self.cc_emits.append((now, e.get("t")))
        s.flush()
        return snap


POLICIES = {
    "eager (shipped default)": lambda c: True,
    "eager + tagger gate": lambda c: c["gate"],
    "eager + conf>=0.4 (~eager_eot 0.4)": lambda c: c["conf"] >= 0.4,
    "eager + conf>=0.5 (~eager_eot 0.5)": lambda c: c["conf"] >= 0.5,
    "eager + (gate OR conf>=0.5)": lambda c: c["gate"] or c["conf"] >= 0.5,
    "eager + last word is content": lambda c: c["last_content"],
}
ARC_POLICIES = {
    "eager + arc cc_p>=0.3": lambda c: (c["arc_cc_p"] or 0) >= 0.3,
    "eager + arc cc event within 0.6 s of last word": lambda c: c["arc_cc_recent"],
    "eager + (gate OR arc cc_p>=0.3)": lambda c: c["gate"] or (c["arc_cc_p"] or 0) >= 0.3,
    "eager + arc for_me>=0.5": lambda c: (c["arc_for_me"] or 0) >= 0.5,
    "eager, then wait for arc cc event (live: on_meta)": lambda c: True if c["arc_cc_recent"] else "wait",
    "eager + gate, else wait for arc cc event": lambda c: True if (c["gate"] or c["arc_cc_recent"]) else "wait",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="?", default=str(ROOT / "runs" / "flux_traces.jsonl"))
    ap.add_argument("--arc", action="store_true", help="also replay Arc Mini audio metadata (external mode)")
    a = ap.parse_args()
    rows = [json.loads(x) for x in Path(a.traces).read_text().splitlines() if x.strip()]
    pol = dict(POLICIES, **(ARC_POLICIES if a.arc else {}))
    t = Tagger(domain="assistant")
    ts = TagState(t)
    arc = ArcReplay() if a.arc else None
    res = {k: {"lat": [], "prem": 0} for k in pol}
    n_eager = n_prem = n_split = 0
    cases = []
    for tr in rows:
        msgs = tr["msgs"]
        tis = [(i, m) for i, m in enumerate(msgs) if m.get("type") == "TurnInfo"]
        eots = [m for _, m in tis if m["event"] == "EndOfTurn" and norm_words(m)]
        if not eots:
            continue
        final = eots[-1]
        n_split += len(eots) > 1
        final_words = norm_words(final)
        last_end = float(final["words"][-1]["end"])
        snap = arc.run(tr) if arc else {}
        ts.reset()
        cur_turn = None
        cands = []
        for i, m in tis:
            if m["turn_index"] != cur_turn:
                ts.reset()
                cur_turn = m["turn_index"]
            ws = norm_words(m)
            ts.sync(ws)
            if m["event"] != "EagerEndOfTurn" or not ws:
                continue
            f = ts.frame()
            complete = m["turn_index"] == final["turn_index"] and ws == final_words
            sm = snap.get(i, {})
            w_end = float(m["words"][-1]["end"])
            c = {"t": m["_t"], "complete": complete, "gate": gate_open(f, ts.cc_at, ts.spans) and f.has_content(),
                 "conf": float(m.get("end_of_turn_confidence") or 0), "last_content": bool(ws) and ws[-1] not in frame.STOP,
                 "arc_cc_p": sm.get("cc_p"), "arc_for_me": sm.get("for_me"),
                 "arc_cc_recent": sm.get("cc_t") is not None and sm["cc_t"] >= w_end - 0.6,
                 "words": " ".join(ws), "sig": meaning(f)}
            # waiting variant: rejected now, but Arc's command_complete (for these words) arrives before Flux says more
            nxt = next((mm["_t"] for _, mm in tis if mm["_t"] > m["_t"] and mm["event"] in
                        ("TurnResumed", "EndOfTurn", "EagerEndOfTurn")), 1e9)
            c["arc_wait_t"] = next((now for now, et in (arc.cc_emits if arc else [])
                                    if m["_t"] <= now < nxt and et is not None and et >= w_end - 0.6), None)
            cands.append(c)
            n_eager += 1
            n_prem += not complete
        cases.append((tr["id"], tr["text"], cands, final["_t"] - last_end))
        for k, p in pol.items():
            fired = None
            for c in cands:
                ok = p(c)
                at = c["t"]
                if ok == "wait":
                    ok, at = c["arc_wait_t"] is not None, c["arc_wait_t"]
                if ok:
                    if c["complete"]:
                        fired = at
                        break
                    res[k]["prem"] += 1
            res[k]["lat"].append(((fired if fired is not None else final["_t"]) - last_end) * 1000)
    t.close()
    n = len(cases)
    print(f"{n} questions (real Flux traces, eager_eot_threshold {rows[0].get('eager')}, eot {rows[0].get('eot')}); "
          f"{n_eager} EagerEndOfTurn events, {n_prem} premature; {n_split} questions split into >1 Flux turn\n")
    eot = np.array([c[3] for c in cases]) * 1000
    print(f"{'policy':52s} trig p50  mean   p90  premature/100q")
    print(f"{'(no soft answer: EndOfTurn)':52s} {np.median(eot):8.0f} {eot.mean():5.0f} {np.percentile(eot, 90):5.0f}  {0:6.1f}")
    for k, r in res.items():
        x = np.array(r["lat"])
        print(f"{k:52s} {np.median(x):8.0f} {x.mean():5.0f} {np.percentile(x, 90):5.0f}  {100 * r['prem'] / n:6.1f}")
    print("\npremature eagers (words at the eager | question):")
    for qid, txt, cands, _ in cases:
        for c in cands:
            if not c["complete"]:
                arcs = f" arc_cc_p={c['arc_cc_p']}" if a.arc else ""
                print(f"  {qid}: '{c['words']}' gate={int(c['gate'])} conf={c['conf']:.2f}{arcs} | {txt}")
    print("\ncomplete eagers the tagger gate rejects:")
    for qid, txt, cands, _ in cases:
        for c in cands:
            if c["complete"] and not c["gate"]:
                print(f"  {qid}: '{c['words']}' conf={c['conf']:.2f}")
                break


if __name__ == "__main__":
    main()
