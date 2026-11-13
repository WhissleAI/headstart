"""Retrieval eval for the local moments index: does each spoken question find its gold moment?

    .venv/bin/python index/eval_local.py      # -> index/data/retrieval_eval.json

Per question (index/questions_local.jsonl, audio in eval/audio/local_*.wav), recall@1 / @4 / MRR@10 of gold_ids for:
  text_text   : question text  -> text vectors only                     (search_vec use_audio=False)
  text_max    : question text  -> max(text, audio vectors) per moment    (LocalIndex.search_vec default)
  audio_full  : question AUDIO -> max(text, audio)                       (the audio lane, whole question)
  audio_1p5   : first 1.5 s of speech (+0.4 s lead silence) -> max(text, audio)   (the audio lane, early)
  audio_full_textonly / audio_1p5_textonly : same audio queries against text vectors only
Second stage (server/rerank.py), over the whole index:
  strip_text  : cue-stripped question text ("in the hearing" removed) -> text vectors
  lex_only    : lexical + tag/role score alone
  fused_tl    : text + lexical                        (what baseline gets: no audio lane)
  fused_ta    : text + audio
  fused_tal   : text + audio + lexical                (Head Start with the audio lane: server default)
Every mode is also run on the ASR text Arc actually heard (sim/data/ear_traces/<id>.json final words, when recorded):
columns with the suffix "@asr". Query vectors are cached in runs/local_eval_vecs.pkl (git-ignored).
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from server import rerank  # noqa: E402
from server.lanes import LocalIndex  # noqa: E402

DATA = ROOT / "index" / "data"
QFILE = ROOT / "index" / "questions_local.jsonl"
TRACES = ROOT / "sim" / "data" / "ear_traces"
CACHE = ROOT / "runs" / "local_eval_vecs.pkl"
LEAD = 0.4


def load_wav(p):
    x, sr = sf.read(str(p), dtype="float32")
    if x.ndim > 1:
        x = x.mean(1)
    assert sr == 16000, sr
    return x


def asr_text(qid: str) -> str | None:
    p = TRACES / f"{qid}.json"
    if not p.exists():
        return None
    t = json.loads(p.read_text())
    fin = [e for e in t["events"] if e["k"] == "final" and e["words"]]
    ws = [w for e in fin for w in e["words"]] if fin else []
    return " ".join(w["text"] for w in ws) or None


class Vecs:
    """Query vectors, cached on disk (the model loads only on a miss)."""

    def __init__(self):
        self.c = pickle.loads(CACHE.read_bytes()) if CACHE.exists() else {}
        self.dirty = False
        self.load_s = 0.0

    def _embed(self):
        from server import embed
        if self.load_s == 0.0:
            t = time.perf_counter()
            embed.model()
            embed.qmodel()
            self.load_s = time.perf_counter() - t
        return embed

    def text(self, s: str) -> np.ndarray:
        k = ("q", s)
        if k not in self.c:
            self.c[k] = self._embed().encode_queries([s])[0]
            self.dirty = True
        return self.c[k]

    def audio(self, name: str, w: np.ndarray) -> np.ndarray:
        k = ("a", name, len(w))
        if k not in self.c:
            # one clip per call, as the server's audio lane does (padded multi-clip batches hit an MPS indexing fault)
            self.c[k] = self._embed().encode_audio([w])[0]
            self.dirty = True
        return self.c[k]

    def save(self):
        if self.dirty:
            CACHE.parent.mkdir(exist_ok=True)
            CACHE.write_bytes(pickle.dumps(self.c))


def full_rank(idx: LocalIndex, q: np.ndarray, use_audio: bool) -> np.ndarray:
    s = idx.tv @ q
    if use_audio and idx.av is not None:
        sa = idx.av @ q
        for row, v in zip(idx.av_rows, sa):
            s[row] = max(s[row], v)
    return s


def metrics(hid: list[str], gold: set) -> tuple[float, float, float, int | None]:
    r1 = float(hid[0] in gold)
    r4 = float(bool(gold & set(hid[:4])))
    rank = next((j + 1 for j, h in enumerate(hid[:10]) if h in gold), None)
    return r1, r4, (1.0 / rank if rank else 0.0), rank


def main():
    qs = [json.loads(l) for l in QFILE.read_text().splitlines() if l.strip()]
    idx = LocalIndex(DATA)
    ids = [d.id for d in idx.docs]
    lex = rerank.lexicon(idx)
    V = Vecs()
    wavs = {q["id"]: load_wav(ROOT / "eval" / "audio" / q["file"]) for q in qs}
    t = time.perf_counter()
    qa = {q["id"]: V.audio(q["file"], wavs[q["id"]]) for q in qs}
    audio_ms = (time.perf_counter() - t) * 1000 / len(qs)
    qa15 = {q["id"]: V.audio(q["file"], wavs[q["id"]][: int(16000 * (LEAD + 1.5))]) for q in qs}

    fuse_ms = []

    def run_modes(qtext: str, a: np.ndarray, a15: np.ndarray) -> dict[str, list[str]]:
        qt = V.text(qtext)
        qs_ = V.text(rerank.strip_cues(qtext))
        st = full_rank(idx, qt, False)
        sa = full_rank(idx, a, True)
        out = {
            "text_text": st, "text_max": full_rank(idx, qt, True), "audio_full": sa,
            "audio_1p5": full_rank(idx, a15, True), "audio_full_textonly": full_rank(idx, a, False),
            "audio_1p5_textonly": full_rank(idx, a15, False), "strip_text": full_rank(idx, qs_, False),
        }
        hits = {m: [ids[j] for j in np.argsort(-s, kind="stable")[:10]] for m, s in out.items()}
        terms, (roles, events) = rerank.query_terms(qtext), rerank.cues(qtext)
        lx = np.array([lex.score(i, terms, roles, events) for i in ids])
        hits["lex_only"] = [ids[j] for j in np.argsort(-lx, kind="stable")[:10]]
        t0 = time.perf_counter()
        for m, kw in {"fused_tl": dict(audio_s=None), "fused_ta": dict(audio_s=sa, w_lex=0.0),
                      "fused_tal": dict(audio_s=sa)}.items():
            hits[m] = [i for i, _, _ in rerank.fuse(ids, st, qtext, lex, **kw)[:10]]
        fuse_ms.append((time.perf_counter() - t0) * 1000 / 3)
        return hits

    per_q, sums = [], {}
    for q in qs:
        gold = set(q["gold_ids"])
        row = {"id": q["id"], "text": q["text"], "gold_ids": q["gold_ids"]}
        heard = asr_text(q["id"])
        variants = [("", q["text"])] + ([("@asr", heard)] if heard else [])
        if heard:
            row["asr"] = heard
        for suf, qtext in variants:
            for m, hid in run_modes(qtext, qa[q["id"]], qa15[q["id"]]).items():
                r1, r4, rr, rank = metrics(hid, gold)
                s = sums.setdefault(m + suf, {"r@1": 0.0, "r@4": 0.0, "mrr@10": 0.0, "n": 0})
                s["r@1"] += r1
                s["r@4"] += r4
                s["mrr@10"] += rr
                s["n"] += 1
                row[m + suf] = {"top4": hid[:4], "gold_rank": rank}
        per_q.append(row)
    V.save()
    summary = {m: {k: round(v / s["n"], 3) for k, v in s.items() if k != "n"} | {"n": s["n"]} for m, s in sums.items()}

    # held-out view: questions added after the reranker's weights were chosen (local11+)
    def sub(pred):
        out = {}
        for m in ("text_text", "fused_tl", "fused_tal", "text_text@asr", "fused_tl@asr", "fused_tal@asr"):
            rs = [r[m] for r in per_q if pred(r["id"]) and m in r]
            if rs:
                out[m] = {"r@1": round(sum(r["gold_rank"] == 1 for r in rs) / len(rs), 3),
                          "r@4": round(sum(bool(r["gold_rank"] and r["gold_rank"] <= 4) for r in rs) / len(rs), 3),
                          "mrr@10": round(sum(1 / r["gold_rank"] for r in rs if r["gold_rank"]) / len(rs), 3),
                          "n": len(rs)}
        return out

    out = {"n_questions": len(qs), "n_docs": len(ids), "n_audio_vectors": 0 if idx.av is None else len(idx.av),
           "summary": summary, "summary_q01_10": sub(lambda i: int(i[5:]) <= 10),
           "summary_q11_plus": sub(lambda i: int(i[5:]) > 10),
           "rerank": {"w_audio": rerank.W_AUDIO, "w_lex": rerank.W_LEX, "lex_cap": rerank.LEX_CAP,
                      "role_bonus": rerank.ROLE_BONUS},
           "fuse_ms_222_docs": round(float(np.mean(fuse_ms)), 2),
           "embed_load_s": round(V.load_s, 1), "audio_query_ms": round(audio_ms, 1),
           "per_question": per_q, "date": time.strftime("%Y-%m-%d %H:%M")}
    (DATA / "retrieval_eval.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    show = ["text_text", "strip_text", "text_max", "audio_full", "audio_full_textonly", "lex_only", "fused_tl",
            "fused_ta", "fused_tal"]
    print(f"{'mode':22s} {'r@1':>5s} {'r@4':>5s} {'mrr':>5s} | {'@asr r@1':>8s} {'r@4':>5s} {'mrr':>5s}")
    for m in show:
        a, b = summary.get(m, {}), summary.get(m + "@asr", {})
        print(f"{m:22s} {a.get('r@1', 0):5.2f} {a.get('r@4', 0):5.2f} {a.get('mrr@10', 0):5.2f} | "
              f"{b.get('r@1', float('nan')):8.2f} {b.get('r@4', float('nan')):5.2f} {b.get('mrr@10', float('nan')):5.2f}")
    print("q01-10:", json.dumps(out["summary_q01_10"]))
    print("q11+  :", json.dumps(out["summary_q11_plus"]))
    print(f"fuse: {out['fuse_ms_222_docs']} ms per question over {len(ids)} moments")
    for r in per_q:
        print(r["id"], {m: r[m]["gold_rank"] for m in ("text_text", "audio_full", "fused_tl", "fused_tal",
                                                       "text_text@asr", "fused_tal@asr") if m in r})


if __name__ == "__main__":
    main()
