"""Semantic-endpoint gate rules, measured on eval/questions.jsonl (in-process Arc Tagger, no audio, ~1 s of CPU).

For every question the words are streamed into the tagger one at a time, as Session does. At each prefix (>= 2 words,
with content) we ask each rule whether the soft endpoint gate would be open. Reported per rule:
  complete  fraction of full questions whose gate is open at the final word (higher = more soft endpoints)
  early     fraction of proper prefixes whose gate is open (lower = fewer premature answers on a mid-question pause)

Run: .venv/bin/python eval/gate_eval.py [questions.jsonl]
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from server import frame  # noqa: E402
from server.session import gate_open  # noqa: E402
from server.tagger import Tagger  # noqa: E402

CC = "discourse.assistant.command_complete"


def rules(f, cc_at, n, spans):
    slot_end = any(s.get("family") in ("entity", "number") and s["type"] != "entity.numeral" and s["end"] == n
                   for s in spans)
    cc_last = cc_at == n - 1
    return {
        "old: sticky cc OR act_question": bool(f.act_question or cc_at is not None),
        "sticky cc only": cc_at is not None,
        "cc at last word": cc_last,
        "cc at last word OR (question AND slot ends at last word)": cc_last or (f.act_question and slot_end),
        "shipped (session.gate_open)": gate_open(f, cc_at, spans),
    }


def main(path):
    rows = [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    t = Tagger(domain="assistant")
    comp, early, n_pref = {}, {}, 0
    examples = []
    for row in rows:
        ws = row["text"].split()
        t.reset()
        spans, cc_at = [], None
        for i, w in enumerate(ws):
            t.push([w])
            r = t.poll()
            a0 = r.get("a0", 0)
            spans = [s for s in spans if s["end"] <= a0] + r.get("spans", [])
            for e in r.get("events") or []:
                if e["type"] == CC:
                    cc_at = e.get("word", i)
            n = i + 1
            f = frame.build(ws[:n], spans, r.get("current"), {CC} if cc_at is not None else set(), local_ok=False)
            if n < 2 or not f.has_content():
                continue
            res = rules(f, cc_at, n, spans)
            if n == len(ws):
                for k, v in res.items():
                    comp[k] = comp.get(k, 0) + v
                examples.append((row["text"], res["shipped (session.gate_open)"]))
            else:
                n_pref += 1
                for k, v in res.items():
                    early[k] = early.get(k, 0) + v
    t.close()
    print(f"{len(rows)} questions, {n_pref} proper prefixes (>=2 words, with content)\n")
    print(f"{'rule':62s} complete  early")
    for k in comp:
        print(f"{k:62s} {comp[k] / len(rows):8.2f} {early.get(k, 0) / max(1, n_pref):6.2f}")
    print("\nfull questions whose gate stays closed (shipped rule):")
    for txt, ok in examples:
        if not ok:
            print("  ", txt)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ROOT / "eval" / "questions.jsonl")
