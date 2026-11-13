"""Retrieval eval for the local moments index: does each spoken question find its gold moment?

    .venv/bin/python index/eval_local.py      # -> index/data/retrieval_eval.json

Per question (index/questions_local.jsonl, audio in eval/audio/local_*.wav), recall@1 / @4 of gold_ids for:
  text_text   : question text  -> text vectors only                     (search_vec use_audio=False)
  text_max    : question text  -> max(text, audio vectors) per moment    (LocalIndex.search_vec default)
  audio_full  : question AUDIO -> max(text, audio)                       (the audio lane, whole question)
  audio_1p5   : first 1.5 s of speech (+0.4 s lead silence) -> max(text, audio)   (the audio lane, early)
  audio_full_textonly / audio_1p5_textonly : same audio queries against text vectors only
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from server import embed  # noqa: E402
from server.lanes import LocalIndex  # noqa: E402

DATA = ROOT / "index" / "data"
QFILE = ROOT / "index" / "questions_local.jsonl"
LEAD = 0.4


def load_wav(p):
    x, sr = sf.read(str(p), dtype="float32")
    if x.ndim > 1:
        x = x.mean(1)
    assert sr == 16000, sr
    return x


def main():
    qs = [json.loads(l) for l in QFILE.read_text().splitlines() if l.strip()]
    idx = LocalIndex(DATA)
    ids = [d.id for d in idx.docs]
    t = time.perf_counter()
    embed.model()
    load_s = time.perf_counter() - t

    qt = embed.encode_queries([q["text"] for q in qs])
    wavs = [load_wav(ROOT / "eval" / "audio" / q["file"]) for q in qs]
    t = time.perf_counter()
    # one clip per call, as the server's audio lane does (padded multi-clip batches hit an MPS indexing fault)
    qa = np.concatenate([embed.encode_audio([w]) for w in wavs])
    audio_ms = (time.perf_counter() - t) * 1000 / len(qs)
    qa15 = np.concatenate([embed.encode_audio([w[: int(16000 * (LEAD + 1.5))]]) for w in wavs])

    modes = {"text_text": (qt, False), "text_max": (qt, True), "audio_full": (qa, True), "audio_1p5": (qa15, True),
             "audio_full_textonly": (qa, False), "audio_1p5_textonly": (qa15, False)}
    per_q, summary = [], {m: {"r@1": 0.0, "r@4": 0.0, "mrr@10": 0.0} for m in modes}
    for i, q in enumerate(qs):
        gold = set(q["gold_ids"])
        row = {"id": q["id"], "text": q["text"], "gold_ids": q["gold_ids"]}
        for m, (Q, use_audio) in modes.items():
            hits = idx.search_vec(Q[i], k=10, use_audio=use_audio)
            hid = [h.id for h in hits]
            r1 = float(hid[0] in gold)
            r4 = float(bool(gold & set(hid[:4])))
            rr = next((1.0 / (j + 1) for j, h in enumerate(hid) if h in gold), 0.0)
            summary[m]["r@1"] += r1 / len(qs)
            summary[m]["r@4"] += r4 / len(qs)
            summary[m]["mrr@10"] += rr / len(qs)
            gold_rank = next((j + 1 for j, h in enumerate(hid) if h in gold), None)
            row[m] = {"top4": [(h.id, round(h.score, 3)) for h in hits[:4]], "gold_rank": gold_rank,
                      "top1_title": hits[0].title}
        # expect-keyword check on the top-1 text for the default lane
        top = idx.search_vec(qt[i], k=1)[0]
        row["text_max_top1_has_expect"] = any(k in (top.text + " " + top.title).lower() for k in q["expect"])
        per_q.append(row)
    for m in summary:
        summary[m] = {k: round(v, 3) for k, v in summary[m].items()}
    out = {"n_questions": len(qs), "n_docs": len(ids), "n_audio_vectors": 0 if idx.av is None else len(idx.av),
           "summary": summary, "embed_load_s": round(load_s, 1), "audio_query_ms": round(audio_ms, 1),
           "per_question": per_q, "date": time.strftime("%Y-%m-%d %H:%M")}
    (DATA / "retrieval_eval.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(summary, indent=1))
    for r in per_q:
        print(r["id"], {m: r[m]["gold_rank"] for m in modes}, "|", r["text_max"]["top1_title"])


if __name__ == "__main__":
    main()
