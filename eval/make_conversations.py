"""Two-turn spoken conversations (follow-up uses a pronoun) → eval/audio/conv_*.wav + eval/conversations.jsonl."""
import json, subprocess, sys, tempfile
from pathlib import Path
import numpy as np, soundfile as sf, librosa
ROOT = Path(__file__).resolve().parent
CONVS = [
    ("conv01", "Samantha", ["Who founded the company Nvidia?", "When was it founded?"], [["jensen", "huang"], ["1993"]]),
    ("conv02", "Daniel", ["Who painted the Mona Lisa?", "Which museum is it in?"], [["leonardo"], ["louvre"]]),
    ("conv03", "Samantha", ["Who wrote Pride and Prejudice?", "When was she born?"], [["austen"], ["1775"]]),
    ("conv04", "Daniel", ["What is the capital of Australia?", "What is the population there?"], [["canberra"], []]),
    ("conv05", "Samantha", ["Who developed the theory of general relativity?", "Where was he born?"], [["einstein"], ["ulm"]]),
    ("conv06", "Daniel", ["Which company makes the PlayStation?", "Where is its headquarters?"], [["sony"], ["tokyo"]]),
]
def say(text, voice):
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "x.aiff"
        subprocess.run(["say", "-v", voice, "-o", str(p), text], check=True)
        w, sr = sf.read(str(p), dtype="float32")
    w = w.mean(1) if w.ndim > 1 else w
    return librosa.resample(w, orig_sr=sr, target_sr=16000)
rows = []
for cid, voice, turns, expect in CONVS:
    files = []
    for i, t in enumerate(turns):
        w = say(t, voice)
        w = np.concatenate([np.zeros(6400, np.float32), w, np.zeros(4800, np.float32)])
        f = f"{cid}_{i+1}.wav"
        sf.write(ROOT / "audio" / f, w, 16000, subtype="PCM_16")
        files.append(f)
    rows.append({"id": cid, "voice": voice, "turns": turns, "files": files, "expect": expect, "kind": "conversation"})
(ROOT / "conversations.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
print(len(rows), "conversations")
