"""Per-message feature snapshots of the 80 real Flux traces, for the confidence-fusion fits and evals.

Every TurnInfo message of runs/flux_traces.jsonl is replayed in arrival order through
  - the Arc Tagger exactly as Session feeds it (eval/flux_gate_eval.TagState), and
  - Arc Mini audio metadata in external-transcript mode on the real audio + the same words (eval/flux_gate_eval.ArcReplay)
and we save what a live Session would know when that message arrives: the words with Flux confidence, per-word revision
history (server/confidence.WordTrack), the tagger's spans (score / thr / ver / alt), intents / acts, command_complete,
the gate, Arc audio cc_p / cc events / for-me, Flux's end_of_turn_confidence and the pause since the last word ended.
Labels: `complete` (the message's words are the question's final words, same turn) and, per word, `final_ok` (this
word survives unchanged into the final transcript).

Run: .venv/bin/python eval/conf_features.py [--no-arc]   -> runs/conf_features.jsonl (~1 min with Arc)
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from flux_gate_eval import ArcReplay, TagState, norm_words  # noqa: E402
from server import confidence, frame  # noqa: E402
from server.session import gate_open  # noqa: E402
from server.tagger import Tagger  # noqa: E402

OUT = ROOT / "runs" / "conf_features.jsonl"


def span_rec(s):
    return {"type": s["type"], "family": s.get("family"), "start": s["start"], "end": s["end"], "text": s.get("text"),
            "score": s.get("score"), "thr": s.get("thr"), "ver": s.get("ver"), "alt": s.get("alt")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default=str(ROOT / "runs" / "flux_traces.jsonl"))
    ap.add_argument("--no-arc", action="store_true")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    rows = [json.loads(x) for x in Path(a.traces).read_text().splitlines() if x.strip()]
    t = Tagger(domain="assistant")
    cc_scores = {}                                  # word index -> score of the newest command_complete event there
    _poll = t.poll

    def poll():
        r = _poll()
        for e in r.get("events") or []:
            if e["type"] == "discourse.assistant.command_complete" and e.get("word") is not None:
                cc_scores[e["word"]] = float(e.get("score") or 0)
        return r
    t.poll = poll
    ts = TagState(t)
    arc = None if a.no_arc else ArcReplay()
    out = open(a.out, "w")
    for tr in rows:
        msgs = tr["msgs"]
        tis = [(i, m) for i, m in enumerate(msgs) if m.get("type") == "TurnInfo"]
        eots = [m for _, m in tis if m["event"] == "EndOfTurn" and norm_words(m)]
        if not eots:
            continue
        final = eots[-1]
        final_words = norm_words(final)
        last_end = float(final["words"][-1]["end"])
        snap = arc.run(tr) if arc else {}
        cc_emits = list(arc.cc_emits) if arc else []
        ts.reset()
        cc_scores.clear()
        track = confidence.WordTrack()
        cur_turn = None
        recs = []
        for i, m in tis:
            if m["turn_index"] != cur_turn:
                ts.reset()
                cc_scores.clear()
                track = confidence.WordTrack()
                cur_turn = m["turn_index"]
            ws = norm_words(m)
            if not ws:
                continue
            ts.sync(ws)
            confs = [float(w.get("confidence") or 0) for w in m["words"] if norm_words({"words": [w]})]
            ends = [float(w["end"]) for w in m["words"] if norm_words({"words": [w]})]
            hist = track.update(ws)
            f = ts.frame()
            cc_score = cc_scores.get(ts.cc_at) if ts.cc_at is not None else None
            sm = snap.get(i, {})
            w_end = ends[-1]
            same_turn = m["turn_index"] == final["turn_index"]
            fin_toks = [frame.norm(w) for w in final_words]
            toks = [frame.norm(w) for w in ws]
            final_ok = [same_turn and k < len(fin_toks) and toks[k] == fin_toks[k] for k in range(len(toks))]
            rec = {
                "id": tr["id"], "text": tr["text"], "i": i, "t": m["_t"], "event": m["event"], "turn": m["turn_index"],
                "eot": float(m.get("end_of_turn_confidence") or 0), "words": ws, "conf": confs, "ends": ends,
                "revs": hist["revs"], "age": hist["age"], "pause": round(m["_t"] - w_end, 3),
                "complete": same_turn and ws == final_words, "final_ok": final_ok,
                "last_end": last_end, "final_t": final["_t"], "final_turn": final["turn_index"],
                "spans": [span_rec(s) for s in ts.spans],
                "intent": [{"tag": x["tag"], "p": x["p"], "thr": x.get("thr"), "on": x.get("on")}
                           for x in ((ts.current or {}).get("intent") or [])],
                "act": [{"tag": x["tag"], "p": x["p"], "thr": x.get("thr"), "on": x.get("on")}
                        for x in ((ts.current or {}).get("act") or [])],
                "cc_at": ts.cc_at, "cc_score": cc_score, "gate": bool(gate_open(f, ts.cc_at, ts.spans) and f.has_content()),
                "has_content": f.has_content(), "act_question": f.act_question, "content_idx": list(f.content_idx),
                "slots": [list(s) for s in f.slots], "query": f.query(), "fintent": f.intent, "fintent_p": f.intent_p,
                "arc_cc_p": sm.get("cc_p"), "arc_cc_t": sm.get("cc_t"), "arc_for_me": sm.get("for_me"),
            }
            # Arc cc event for exactly these words arriving before Flux's next message (the eager+arc wait path)
            nxt = next((mm["_t"] for _, mm in tis if mm["_t"] > m["_t"]), 1e9)
            rec["arc_wait_t"] = next((now for now, et in cc_emits
                                      if m["_t"] <= now < nxt and et is not None and et >= w_end - 0.6), None)
            recs.append(rec)
        for r in recs:
            out.write(json.dumps(r) + "\n")
        print(tr["id"], len(recs), flush=True)
    out.close()
    t.close()


if __name__ == "__main__":
    main()
