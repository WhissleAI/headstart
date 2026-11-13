"""Build the local "moments" index for Head Start: real recordings -> utterance-level moments -> EmbeddingGemma 2 vectors.

    .venv/bin/python index/build_index.py              # all stages (Arc transcripts are cached in index/data/arc/)
    .venv/bin/python index/build_index.py --rearc      # re-run Arc even when a cached transcript exists
    .venv/bin/python index/build_index.py --no-audio   # text vectors only

Output (index/data/, the format server/lanes.py LocalIndex loads):
    docs.jsonl   one moment per line {"id","title","text","url","t","meta":{source,end,speaker,role,tags,domain,...}}
    vectors.npz  text [N,768] (embed.encode_docs with titles), audio [M,768] (embed.encode_audio on the moment's own
                 16 kHz slice), audio_rows [M] (docs.jsonl row of each audio vector). All L2-normalized.

Stages run one model at a time (Arc, then Arc Tagger, then EmbeddingGemma) and free each before the next.
"""
from __future__ import annotations

import argparse
import gc
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from server import config  # noqa: E402

DATA = ROOT / "index" / "data"
AUDIO = DATA / "audio"
ARC_CACHE = DATA / "arc"
SR = 16000

HOME = Path.home()
YT1 = HOME / "Documents/downloaded_audios_1"
REPLAY = Path("/Users/karan/Desktop/work/whissle/wt-studio-general/kits/arc_v1/media/replay")

# Each source: id, label (title prefix, read aloud in a citation), kind (replay | file), Arc domain, Tagger domain.
#   replay: an Arc Live Replay export (90 s of held-out test audio + Arc/Deepgram words + gold names/roles/events)
#   file:   a real recording (YouTube download), cut to [a, b] s, transcribed here by Arc v1 (built-in decoder)
SOURCES = [
    # ---- game broadcasts / sports commentary
    {"id": "nba_cavs_heat", "label": "NBA broadcast", "kind": "replay", "ex": "ex1", "words": "deepgram",
     "domain": "nba", "about": "Cavaliers vs Heat, TV booth (play-by-play + analyst)"},
    {"id": "cricket_greig", "label": "Cricket commentary", "kind": "file", "domain": "cricket",
     "glob": "Sports_commentary_interviews/Tony Greig - Commentary Gold*", "about": "Tony Greig calls cricket"},
    {"id": "cricket_box", "label": "Cricket commentary box", "kind": "file", "domain": "cricket",
     "glob": "Sports_commentary_interviews/Commentary box sledging*",
     "about": "Sourav Ganguly and Rahul Dravid banter in the commentary box"},
    {"id": "olympics", "label": "Olympics commentary", "kind": "file", "domain": "unknown", "tdomain": "talk",
     "glob": "Sports_commentary_interviews/Hilarious Olympic Commentary*", "b": 180,
     "about": "Kevin Hart and Kenan Thompson commentate Olympic events"},
    {"id": "drury", "label": "Football commentator interview", "kind": "file", "domain": "talk",
     "glob": "Sports_commentary_interviews/Peter Drury talks*", "about": "Peter Drury on his favourite commentary"},
    # ---- court hearings
    {"id": "scotus", "label": "Supreme Court argument", "kind": "replay", "ex": "ex3", "words": "builtin",
     "domain": "court_appellate", "about": "U.S. Supreme Court oral argument, justices question counsel"},
    {"id": "court_sham", "label": "Court hearing", "kind": "file", "domain": "court_trial",
     "glob": "Judge_and_defendant_courtroom_exchanges/'THIS IS A SHAM!'*",
     "about": "a prosecutor storms out after arguing with the judge"},
    {"id": "court_heated", "label": "Court hearing", "kind": "file", "domain": "court_trial",
     "glob": "Judge_and_defendant_courtroom_exchanges/Heated exchange in courtroom*",
     "about": "heated exchange in a courtroom"},
    {"id": "court_dismissed", "label": "Court hearing", "kind": "file", "domain": "court_trial",
     "glob": "Judge_and_defendant_courtroom_exchanges/Case DISMISSED*", "about": "a case dismissed in 34 seconds"},
    {"id": "court_hill", "label": "Court hearing", "kind": "file", "domain": "court_trial",
     "glob": "Judge_and_defendant_courtroom_exchanges/Tense courtroom exchange in Mylik Hill*",
     "about": "Mylik Hill trial"},
    {"id": "court_cueto", "label": "Court hearing", "kind": "file", "domain": "court_trial",
     "glob": "Judge_and_defendant_courtroom_exchanges/Courtroom exchange with Asad and Judge Cueto*",
     "about": "Asad and Judge Cueto"},
    # ---- meetings / calls
    {"id": "mtg_parent_teacher", "label": "Parent-teacher meeting", "kind": "file", "domain": "meeting",
     "glob": "Teacher_and_parent_meetings/Parent Teacher Meetings - English Meeting Conversation*",
     "about": "a parent-teacher meeting"},
    {"id": "mtg_feedback", "label": "Product feedback meeting", "kind": "file", "domain": "meeting",
     "glob": "Customer_feedback_sessions_with_product_managers/Recruitee presents*",
     "about": "a product feedback session"},
    {"id": "mtg_negotiation", "label": "Negotiation call", "kind": "file", "domain": "meeting",
     "glob": "Business_negotiations_roleplays/Business English Conversation*", "about": "a business negotiation"},
    # ---- podcasts / panels
    {"id": "pod_who_won", "label": "Debate analysis podcast", "kind": "file", "domain": "talk",
     "glob": "Podcast_debates_between_experts/Who won the Harris-Trump debate*",
     "about": "political experts weigh in on who won the Harris-Trump debate"},
    {"id": "pod_christie", "label": "Debate analysis panel", "kind": "file", "domain": "talk", "b": 240,
     "glob": "Podcast_debates_between_experts/Chris Christie and Donna Brazile*",
     "about": "Chris Christie and Donna Brazile on the Harris-Trump debate"},
    # ---- voice-assistant session
    {"id": "assistant", "label": "Voice assistant session", "kind": "replay", "ex": "ex5", "words": "builtin",
     "domain": "assistant", "about": "29 spoken commands to a voice assistant (SLURP)", "seg": "assistant", "split_on_speaker": True},
]

# segmentation (seconds): close a moment at a pause >= GAP once it is >= MIN long; at a sentence end once >= SOFT;
# always at HARD or at a pause >= BREAK. Moments with < MIN_WORDS words are merged into a neighbour.
SEG = {"default": dict(GAP=0.55, MIN=3.0, SOFT=9.0, HARD=16.0, BREAK=1.8, MIN_WORDS=5),
       "assistant": dict(GAP=0.6, MIN=0.0, SOFT=9.0, HARD=16.0, BREAK=0.6, MIN_WORDS=1)}
AUDIO_MIN, AUDIO_MAX = 0.6, 30.0
CTX_WORDS = 14  # neighbour words kept in meta.context_before / context_after (not embedded)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def mmss(t):
    t = int(round(t))
    return f"{t // 60:02d}:{t % 60:02d}"


# ------------------------------------------------------------------ audio

def source_path(s):
    if s["kind"] == "replay":
        return REPLAY / f"{s['ex']}.ogg"
    hits = sorted(YT1.glob(s["glob"]))
    if not hits:
        raise FileNotFoundError(s["glob"])
    return hits[0]


def prep_audio(s) -> Path:
    out = AUDIO / f"{s['id']}.wav"
    if out.exists():
        return out
    src = source_path(s)
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(src), "-ac", "1", "-ar", str(SR)]
    if s.get("a"):
        cmd += ["-ss", str(s["a"])]
    if s.get("b"):
        cmd += ["-t", str(s["b"] - s.get("a", 0))]
    subprocess.run(cmd + ["-c:a", "pcm_s16le", str(out)], check=True)
    return out


# ------------------------------------------------------------------ transcripts

def _plain(x):
    """Arc output -> JSON-safe (drop numpy arrays)."""
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items() if not isinstance(v, np.ndarray)}
    if isinstance(x, list):
        return [_plain(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


def arc_transcribe(todo, rearc=False):
    """Arc v1 offline predict on each file source -> index/data/arc/<id>.json (cached)."""
    need = [s for s in todo if rearc or not (ARC_CACHE / f"{s['id']}.json").exists()]
    if not need:
        return {}
    sys.path.insert(0, str(config.ARC_DIR))
    from arc_audio import Arc
    t0 = time.perf_counter()
    arc = Arc.from_pretrained(str(config.ARC_DIR), device="cpu", threads=config.ARC_THREADS)
    log(f"Arc loaded in {time.perf_counter() - t0:.1f}s")
    timing = {}
    for s in need:
        wav = AUDIO / f"{s['id']}.wav"
        dur = sf.info(str(wav)).duration
        t = time.perf_counter()
        out = arc.predict(str(wav), domain=s["domain"], transcript="builtin")
        dt = time.perf_counter() - t
        keep = {k: out.get(k) for k in ("domain", "transcript", "names", "numbers", "events", "speakers",
                                         "diarization", "utterances")}
        (ARC_CACHE / f"{s['id']}.json").write_text(json.dumps(_plain(keep)))
        timing[s["id"]] = {"audio_s": round(dur, 1), "arc_s": round(dt, 2)}
        log(f"Arc {s['id']}: {dur:.0f}s audio in {dt:.1f}s ({dur / dt:.0f}x real time)")
    del arc
    gc.collect()
    return timing


def load_words_and_labels(s):
    """-> words [(text, start, end)], labels {names:[(s,e,text,kind)], numbers, events:[(t,tag)], roles:[(s,e,role)],
    spk:[(s,e,speaker)]} on the clip's own clock."""
    if s["kind"] == "replay":
        d = json.loads((REPLAY / f"{s['ex']}.json").read_text())
        words = [(w[0], float(w[1]), float(w[2])) for w in d["sources"][s["words"]]["words"]]
        g = d["gold"]
        lab = {"names": [(a, b, t, k) for a, b, t, k in g["names"] if t],
               "numbers": [(a, b, v, k) for a, b, k, v in g["numbers"]],
               "events": [(t, tag) for t, tag in g["events"]],
               "roles": [(a, b, r) for a, b, r in g["roles"]],
               "spk": [(a, b, sp) for a, b, sp in g["speakers"] if sp != "overlap"],
               "phases": [(a, b, p) for a, b, p in g.get("phases") or []],
               "words_source": f"{d['sources'][s['words']]['label']} (Arc Live Replay export)",
               "labels_source": "gold (replay export)"}
        return words, lab
    a = json.loads((ARC_CACHE / f"{s['id']}.json").read_text())
    words = []
    for u in a["transcript"]:
        for w in u.get("words") or []:
            if w.get("text", "").strip():
                words.append((w["text"].strip(), float(w["start"]), float(w["end"])))
    role_of = {}
    for k, v in (a.get("speakers") or {}).items():
        r = (((v.get("groups") or {}).get("speaker.role") or {}).get("label") or "").replace("speaker.role.", "")
        if r and s["domain"] not in ("unknown", "talk") and not (r == "device_user" and not s["domain"].startswith("assistant")):
            role_of[k] = r
    lab = {"names": [(n["start"], n["end"], (n.get("entity") or n.get("heard") or "").strip(" .,?!"),
                      (n.get("tag") or "").split(".")[-1]) for n in a.get("names") or []],
           "numbers": [(n["start"], n["end"], n.get("heard"), n.get("kind") or n.get("kind_head"))
                       for n in a.get("numbers") or []],
           "events": [(e["t"], e["tag"]) for e in a.get("events") or []
                      if not e["tag"].startswith(("speaker.", "meta."))],
           "roles": [(d["start"], d["end"], role_of.get(d["speaker"])) for d in a.get("diarization") or []
                     if role_of.get(d["speaker"])],
           "spk": [(d["start"], d["end"], d["speaker"]) for d in a.get("diarization") or []],
           "phases": [],
           "words_source": "Arc v1 built-in decoder (offline predict)", "labels_source": "Arc v1"}
    return words, lab


# ------------------------------------------------------------------ moments

_END = re.compile(r"[.?!]['\"]?$")


def segment(words, p, bounds=()):
    """bounds: extra cut times (e.g. gold speaker-turn starts); a moment never spans one."""
    bounds = sorted(bounds)
    segs, cur = [], []
    for w in words:
        if cur and any(cur[-1][2] - 0.25 <= b <= w[1] + 0.25 and b > cur[0][1] + 0.3 for b in bounds):
            segs.append(cur)
            cur = []
        if cur:
            gap = w[1] - cur[-1][2]
            dur = cur[-1][2] - cur[0][1]
            if (gap >= p["BREAK"] or (gap >= p["GAP"] and dur >= p["MIN"]) or (dur >= p["SOFT"] and _END.search(cur[-1][0]))
                    or dur >= p["HARD"]):
                segs.append(cur)
                cur = []
        cur.append(w)
    if cur:
        segs.append(cur)
    # merge tiny moments into the nearer neighbour
    changed = True
    while changed and len(segs) > 1:
        changed = False
        for i, sg in enumerate(segs):
            if len(sg) >= p["MIN_WORDS"]:
                continue
            gp = sg[0][1] - segs[i - 1][-1][2] if i > 0 else 1e9
            gn = segs[i + 1][0][1] - sg[-1][2] if i + 1 < len(segs) else 1e9
            j = i - 1 if gp <= gn else i + 1
            if min(gp, gn) > 3.0:
                continue
            a, b = sorted((i, j))
            segs[a:b + 1] = [segs[a] + segs[b]]
            changed = True
            break
    return segs


def _overlap_pick(items, a, b):
    """items [(s,e,label)] -> label with most overlap with [a,b]."""
    acc = {}
    for s, e, lab in items:
        ov = min(b, e) - max(a, s)
        if ov > 0 and lab:
            acc[lab] = acc.get(lab, 0) + ov
    return max(acc, key=acc.get) if acc else ""


def build_moments(sources):
    docs = []
    for s in sources:
        words, lab = load_words_and_labels(s)
        bounds = [a for a, _, _ in lab["spk"]] if s.get("split_on_speaker") else ()
        segs = segment(words, SEG[s.get("seg", "default")], bounds)
        texts = [" ".join(w[0] for w in sg) for sg in segs]
        for i, sg in enumerate(segs):
            a, b = sg[0][1], sg[-1][2]
            text = texts[i][:600]
            prev_tail = " ".join(w[0] for w in segs[i - 1][-CTX_WORDS:]) if i > 0 else ""
            next_head = " ".join(w[0] for w in segs[i + 1][:CTX_WORDS]) if i + 1 < len(segs) else ""
            tags = []
            for ns, ne, t, k in lab["names"]:
                if ns < b + 0.2 and ne > a - 0.2 and t:
                    tags.append(t)
            for ns, ne, t, k in lab["numbers"]:
                if ns < b + 0.2 and ne > a - 0.2 and t:
                    tags.append(str(t))
            for t, tag in lab["events"]:
                if a - 0.3 <= t <= b + 0.6:
                    tags.append(tag.replace("event.", "").replace("discourse.", ""))
            docs.append({
                "id": f"{s['id']}-{i:03d}",
                "title": f"{s['label']} · {mmss(a)}",
                "text": text,
                "url": "",
                "t": round(a, 2),
                "meta": {"source": s["id"], "source_label": s["label"], "about": s.get("about", ""),
                         "end": round(b, 2), "speaker": _overlap_pick(lab["spk"], a, b),
                         "role": _overlap_pick(lab["roles"], a, b) or "",
                         "phase": _overlap_pick(lab["phases"], a, b) or "",
                         "tags": tags, "domain": s["domain"], "words_source": lab["words_source"],
                         "context_before": prev_tail, "context_after": next_head},
            })
    return docs


def tag_moments(docs, sources):
    """Arc Tagger (native int8) on each moment's text: entities, numbers, events, on-intents -> meta.tags."""
    from server.tagger import Tagger
    tdom = {s["id"]: s.get("tdomain", s["domain"]) for s in sources}
    taggers = {}
    t0 = time.perf_counter()
    for d in docs:
        dom = tdom[d["meta"]["source"]]
        if dom not in taggers:
            taggers[dom] = Tagger(domain=dom)
        r = taggers[dom].tag_text(d["text"])
        tags = list(d["meta"]["tags"])
        for sp in r.get("spans") or []:
            fam, typ = sp.get("family"), sp.get("type", "")
            if fam in ("entity", "number") and sp.get("score", 0) >= sp.get("thr", 1) and typ != "entity.numeral":
                tags.append(sp["text"].strip(" .,?!"))
        for ev in r.get("events") or []:
            tags.append(ev["type"].replace("event.", ""))
        for u in r.get("utterances") or []:
            for it in u.get("intent") or []:
                if it.get("on"):
                    tags.append(it["tag"].replace("discourse.", ""))
        seen, out = set(), []
        for t in tags:
            k = t.lower()
            if t and k not in seen:
                seen.add(k)
                out.append(t)
        d["meta"]["tags"] = out[:14]
    n = len(docs)
    for t in taggers.values():
        t.close()
    return time.perf_counter() - t0, n


def embed_text_of(d):
    """What the text vector sees: the moment's own words. (Adding neighbour context was tried and lowered recall@1 on
    the local question set, 0.7 -> 0.5, so the neighbours only go in meta for the answer step.)"""
    return d["text"][:600]


def embed_title_of(d):
    m = d["meta"]
    role = f" · {m['role'].replace('_', ' ')}" if m.get("role") else ""
    return f"{d['title']}{role} · {m['about']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rearc", action="store_true")
    ap.add_argument("--no-audio", action="store_true")
    args = ap.parse_args()
    for p in (DATA, AUDIO, ARC_CACHE):
        p.mkdir(parents=True, exist_ok=True)
    stats = {"t_start": time.strftime("%Y-%m-%d %H:%M:%S")}

    t = time.perf_counter()
    for s in SOURCES:
        prep_audio(s)
    stats["prep_s"] = round(time.perf_counter() - t, 1)
    stats["sources"] = {s["id"]: {"label": s["label"], "kind": s["kind"], "domain": s["domain"],
                                  "file": str(source_path(s)), "about": s.get("about", ""),
                                  "dur_s": round(sf.info(str(AUDIO / f"{s['id']}.wav")).duration, 1)}
                        for s in SOURCES}

    stats["arc"] = arc_transcribe([s for s in SOURCES if s["kind"] == "file"], rearc=args.rearc)

    docs = build_moments(SOURCES)
    dt, n = tag_moments(docs, SOURCES)
    stats["tagger_s"] = round(dt, 2)
    log(f"{len(docs)} moments; Arc Tagger {dt * 1000 / max(n, 1):.1f} ms/moment")

    # ---- EmbeddingGemma 2 (bf16 on MPS via server.embed)
    from server import embed
    t = time.perf_counter()
    embed.model()
    stats["embed_load_s"] = round(time.perf_counter() - t, 1)
    t = time.perf_counter()
    tv = embed.encode_docs([embed_text_of(d) for d in docs], [embed_title_of(d) for d in docs])
    stats["embed_text_s"] = round(time.perf_counter() - t, 1)
    log(f"text vectors {tv.shape} in {stats['embed_text_s']}s")

    av, rows = np.zeros((0, config.EMBED_DIM), np.float32), []
    if not args.no_audio:
        t = time.perf_counter()
        wavs, cache = [], {}
        for i, d in enumerate(docs):
            src = d["meta"]["source"]
            if src not in cache:
                cache[src], _ = sf.read(str(AUDIO / f"{src}.wav"), dtype="float32")
            a, b = d["t"] - 0.15, d["meta"]["end"] + 0.15
            if not (AUDIO_MIN <= b - a <= AUDIO_MAX):
                continue
            x = cache[src][max(0, int(a * SR)): int(b * SR)]
            if len(x) < AUDIO_MIN * SR:
                continue
            wavs.append(x)
            rows.append(i)
        # one clip per call: matches query-time encoding (no padding) and avoids an MPS indexing fault seen with
        # padded batches of mixed-length audio in EmbeddingGemma 2
        chunks, keep = [], []
        for x, r in zip(wavs, rows):
            try:
                chunks.append(embed.encode_audio([x]))
                keep.append(r)
            except Exception as e:  # noqa: BLE001
                log(f"audio embed failed for row {r} ({len(x) / SR:.1f}s): {str(e)[:120]}")
        rows = keep
        av = np.concatenate(chunks) if chunks else av
        stats["embed_audio_s"] = round(time.perf_counter() - t, 1)
        log(f"audio vectors {av.shape} in {stats['embed_audio_s']}s")

    with open(DATA / "docs.jsonl", "w") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    np.savez(DATA / "vectors.npz", text=tv.astype(np.float32), audio=av.astype(np.float32),
             audio_rows=np.asarray(rows, np.int64))
    stats.update({"n_docs": len(docs), "n_audio": len(rows),
                  "total_audio_s": round(sum(v["dur_s"] for v in stats["sources"].values()), 1),
                  "per_source": {s["id"]: sum(1 for d in docs if d["meta"]["source"] == s["id"]) for s in SOURCES}})
    (DATA / "build_stats.json").write_text(json.dumps(stats, indent=1))
    log(f"wrote {DATA / 'docs.jsonl'} ({len(docs)}) and vectors.npz (audio {len(rows)})")


if __name__ == "__main__":
    main()
