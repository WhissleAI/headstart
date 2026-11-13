"""Spoken questions about the local moments corpus -> index/questions_local.jsonl + eval/audio/local_*.wav.

    .venv/bin/python index/make_questions.py           # only synthesizes wavs that are missing (--redo: all)

Audio: macOS `say` -> 16 kHz mono 16-bit wav with 0.4 s leading and 0.3 s trailing silence. Each question carries a
local cue from server/frame.py LOCAL_CUES so the router sends it to the local lane; gold_ids are checked to exist.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from server.frame import LOCAL_CUES, norm  # noqa: E402

OUT = ROOT / "index" / "questions_local.jsonl"
AUD = ROOT / "eval" / "audio"
SR = 16000

Q = [
    ("When did the announcer call Hunter's three in the game?", "Samantha",
     ["hunter", "nails|nailed"], ["nba_cavs_heat-003"]),
    ("What did the judge say about the objection in the hearing?", "Daniel",
     ["objection"], ["court_sham-001"]),
    ("In the call, where did they agree to meet halfway on the discount?", "Karen",
     ["halfway", "6%"], ["mtg_negotiation-012"]),
    ("What did the commentator say about Lonzo Ball in the game?", "Samantha",
     ["ball", "triple"], ["nba_cavs_heat-007"]),
    ("In the hearing, where was the gun found?", "Daniel",
     ["chair"], ["scotus-001"]),
    ("What did the judge ask about the officers waiting fifteen seconds?", "Karen",
     ["15|fifteen", "seconds"], ["scotus-002"]),
    ("In the meeting, what did the teacher say Samuel won?", "Samantha",
     ["science", "exhibition"], ["mtg_parent_teacher-017"]),
    ("What did the commentator say about who won the debate?", "Daniel",
     ["winner|won"], ["pod_who_won-001", "pod_who_won-000", "pod_christie-011"]),
    ("In the recording, what did Chris Christie say about debate prep with Trump?", "Karen",
     ["prep|prepping|preparation", "wrestling", "boxing"], ["pod_christie-001", "pod_christie-002"]),
    ("In the clip, when did they say team USA won bronze?", "Samantha",
     ["bronze"], ["olympics-005", "olympics-003"]),
    # local_11+ (added with the reranker; written before tuning it, kept as a check on overfitting)
    ("What did the judge say about where in the chair the gun was?", "Daniel",
     ["chair", "cushion|stuff|stuffed"], ["scotus-006", "scotus-007"]),
    ("In the call, what discount did they offer first?", "Karen",
     ["3%|three percent"], ["mtg_negotiation-009"]),
    ("In the meeting, what did the teacher say about Aaron's workbook?", "Samantha",
     ["blank|sketches"], ["mtg_parent_teacher-009", "mtg_parent_teacher-011"]),
    ("What did the announcer say about Kel'el Ware's triple?", "Daniel",
     ["twelfth|12th|12"], ["nba_cavs_heat-006"]),
    ("In the hearing, what did the judge tell the defendant about getting an attorney?", "Karen",
     ["attorney|lawyer|public defender"], ["court_cueto-008", "court_cueto-009", "court_cueto-010"]),
    ("In the clip, who did France beat in overtime?", "Samantha",
     ["japan"], ["olympics-012"]),
    ("In the commentary, what was England's score when they talked about declaring?", "Daniel",
     ["490"], ["cricket_greig-001"]),
    ("In the video, what did the commentator say was his favourite goal?", "Karen",
     ["shabalala|south africa"], ["drury-001", "drury-003"]),
    ("In the hearing, how many cues are taught in the training?", "Samantha",
     ["24|twenty four|twenty-four"], ["court_dismissed-000"]),
    ("In the broadcast, how many points is Jarrett Allen averaging?", "Daniel",
     ["15|fifteen"], ["nba_cavs_heat-002"]),
]

def synth(text, voice, path):
    with tempfile.TemporaryDirectory() as td:
        aiff = Path(td) / "q.aiff"
        subprocess.run(["say", "-v", voice, "-o", str(aiff), text], check=True)
        x, sr = sf.read(str(aiff), dtype="float32")
    if x.ndim > 1:
        x = x.mean(1)
    if sr != SR:
        x = librosa.resample(x, orig_sr=sr, target_sr=SR)
    x = np.concatenate([np.zeros(int(0.4 * SR), np.float32), x.astype(np.float32), np.zeros(int(0.3 * SR), np.float32)])
    sf.write(str(path), x, SR, subtype="PCM_16")
    return len(x) / SR


def main():
    redo = "--redo" in sys.argv
    ids = {json.loads(l)["id"] for l in (ROOT / "index/data/docs.jsonl").read_text().splitlines()}
    AUD.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (text, voice, expect, gold) in enumerate(Q, 1):
        missing = [g for g in gold if g not in ids]
        assert not missing, (text, missing)
        low = " " + norm(text) + " "
        cue = next((c for c in LOCAL_CUES if " " + c + " " in low), None)
        assert cue, f"no local cue: {text}"
        f = f"local_{i:02d}.wav"
        if (AUD / f).exists() and not redo:     # keep existing audio (other runs may be reading it)
            x, _ = sf.read(str(AUD / f))
            dur = len(x) / SR
        else:
            dur = synth(text, voice, AUD / f)
        rows.append({"id": f"local{i:02d}", "file": f, "text": text, "voice": f"say:{voice}", "kind": "local",
                     "expect": expect, "gold_ids": gold, "cue": cue, "dur_s": round(dur, 2)})
        print(rows[-1]["id"], f"{dur:.2f}s", cue, "|", text)
    OUT.write_text("".join(json.dumps(r) + "\n" for r in rows))


if __name__ == "__main__":
    main()
