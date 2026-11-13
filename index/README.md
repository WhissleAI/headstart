# Local moments index

The `local` lane searches this index. It holds real recordings (game broadcasts, court hearings, meetings, podcasts and a
voice-assistant session) cut into utterance-level **moments**. Each moment has a timestamp, so an answer can cite
"NBA broadcast · 00:26". Each moment is embedded twice with EmbeddingGemma 2: once from its **text** and once from its
**raw audio**. Both land in the same 768-d space as the query, so a spoken question's audio can search the index
before its words settle.

## Build

```bash
.venv/bin/python index/build_index.py        # source audio -> index/data/{docs.jsonl, vectors.npz, build_stats.json}
.venv/bin/python index/make_questions.py     # index/questions_local.jsonl + eval/audio/local_*.wav (macOS `say`)
.venv/bin/python index/eval_local.py         # -> index/data/retrieval_eval.json
```

`build_index.py` runs one model at a time and frees each before loading the next: Arc v1, then Arc Tagger, then
EmbeddingGemma 2 (bf16 on MPS, through `server/embed.py`, so the vectors match what the server computes at query
time). Peak memory is about 4 GB. Arc transcripts are cached in `index/data/arc/`. Use `--rearc` to redo them and
`--no-audio` to build text vectors only. Everything under `index/data/` is git-ignored, including the 16 kHz copies
of the sources in `index/data/audio/`.

Timings on an M3 Pro, full rebuild from the source audio:

| step | time |
|---|---|
| Arc v1 offline transcription, CPU, 6 threads | 2.6 min for 31 min of audio (9–17x real time) |
| Arc Tagger | 4 ms per moment |
| Text embedding, 222 moments | 3.5 s |
| Audio embedding, 222 moments, one clip per call | 96 s (about 430 ms per moment) |

## Sources (35 min of real recordings, 17 clips, 222 moments)

| group | clips | words from | tags from |
|---|---|---|---|
| NBA broadcast (Cavaliers vs Heat TV booth, 90 s) | `nba_cavs_heat` | Deepgram words from the Arc Live Replay export (`wt-studio-general/kits/arc_v1/media/replay/ex1`) | gold names, numbers, score events and play-by-play/analyst roles from the replay, plus Arc Tagger |
| Supreme Court oral argument (90 s) | `scotus` | Arc v1.4 words from the replay export (`ex3`) | gold judge/counsel roles and argument phase, plus Arc Tagger |
| Voice-assistant session (SLURP commands, 90 s) | `assistant` | Arc words from the replay export (`ex5`), split at gold speaker turns | gold command-complete events, plus Arc Tagger |
| Sports commentary (cricket ×2, Olympics, a football commentator interview) | `cricket_greig`, `cricket_box`, `olympics`, `drury` | Arc v1 built-in decoder, offline `predict` | Arc names, numbers and events, plus Arc Tagger |
| Court hearings (5 clips) | `court_sham`, `court_heated`, `court_dismissed`, `court_hill`, `court_cueto` | Arc v1 | Arc and Arc Tagger (`court_trial`) |
| Meetings and calls | `mtg_parent_teacher`, `mtg_feedback`, `mtg_negotiation` | Arc v1 | Arc and Arc Tagger (`meeting`) |
| Podcast and panel (debate analysis) | `pod_who_won`, `pod_christie` (first 240 s) | Arc v1 | Arc and Arc Tagger (`talk`) |

Notes on the sources:

- **YouTube downloads.** The non-replay clips are downloads in `~/Documents/downloaded_audios_1/<category>/`. They are
  third-party content, used locally for the prototype; clear the rights before showing them publicly.
- **Dropped clip.** `Judge kicks defendant out of courtroom` was dropped: Arc's built-in decoder produced unrelated
  assistant-style text for it.
- **Transcript quality.** Arc's built-in decoder text for the YouTube clips is noisy (Moonshine tiny). Examples:
  "Ponto ball" for Lonzo Ball, and "team USA one broadcast" for "won bronze".
- **Counts.** Moments per clip are listed in `index/data/build_stats.json`.

## Format (what `server/lanes.py` `LocalIndex` loads)

- **`docs.jsonl`** has one moment per line:
  `{"id": "nba_cavs_heat-003", "title": "NBA broadcast · 00:26", "text": <the moment's words>, "url": "", "t": 26.28, "meta": {...}}`.
  `meta` holds:
  - `source`, `source_label`, `about`, `domain` and `words_source`
  - `end` (seconds)
  - `speaker`, `role` (play_by_play / analyst / judge / counsel / device_user when known) and `phase`
  - `tags`: up to 14 short strings (entity and number texts, events like `sports.score.recent`, on-intents)
  - `context_before` / `context_after`: up to 14 neighbour words on each side, for the answer step
- **`vectors.npz`**:
  - `text` [222, 768] = `embed.encode_docs(texts, titles)`. The title is `"<title> · <role> · <about>"` and the text is
    the moment's own words.
  - `audio` [222, 768] = `embed.encode_audio` on the moment's 16 kHz slice (±0.15 s), one clip per call. Slices shorter
    than 0.6 s or longer than 30 s are skipped; none were in this build.
  - `audio_rows` [222] maps each audio vector to its row in `docs.jsonl`.
  - All vectors are L2-normalized.

Segmentation works on word timestamps:

- A moment ends at a pause of at least 0.55 s once it is at least 3 s long.
- It also ends at a sentence end once it is at least 9 s long, at 16 s, or at any pause of at least 1.8 s.
- Moments with fewer than 5 words are merged into the nearer neighbour.

Moments run 1–16 s, most of them 4–12 s.

Neighbour context is not embedded. Adding it lowered recall@1 on the question set from 0.7 to 0.5, and dropping the
`about` part of the title lowered it to 0.6.

## Retrieval eval (`index/data/retrieval_eval.json`)

There are 10 spoken questions (`index/questions_local.jsonl`). Each has a cue from `server/frame.py` `LOCAL_CUES` and 1–3
gold moments. The audio was made with macOS `say` (Samantha, Daniel, Karen), padded with 0.4 s of silence before and
0.3 s after. Each column gives recall@1 / recall@4 / MRR@10:

| query → index | R@1 | R@4 | MRR |
|---|---|---|---|
| question text → text vectors | 0.70 | 0.90 | 0.78 |
| question text → max(text, audio), as in `search_vec` | 0.70 | 0.90 | 0.79 |
| **question audio (whole) → max(text, audio)** (audio lane) | 0.50 | **1.00** | 0.71 |
| question audio (whole) → text vectors only | 0.70 | 0.80 | 0.75 |
| first 1.5 s of question audio → max(text, audio) | 0.10 | 0.20 | 0.13 |
| first 1.5 s of question audio → text vectors only | 0.00 | 0.10 | 0.03 |

Encoding one question's audio takes about 370 ms on a busy machine; the server's warm figure is about 100 ms for 1.5–4 s.

What this says:

- **The whole-question audio embedding is a real retrieval lane.** It finds the gold moment in the top 4 for every
  question, and its top 1 matches the text query's top 1 when audio is scored against text vectors only. It gives a
  usable local candidate list as soon as the speaker stops, with no ASR.
- **The first 1.5 s of audio is not useful.** It is mostly the cue ("In the hearing…", "What did the judge…") and
  carries no content.
- **Moment-audio vectors barely help text queries.** Text-to-audio similarity sits under text-to-text, so the max
  rarely changes the ranking.
- **Moment-audio vectors slightly hurt audio queries at rank 1.** Audio-to-audio matching pulls in acoustically similar
  moments. They do raise recall@4.
- **Scores are compressed.** Top hits score 0.73–0.84, and the gap between rank 1 and rank 2 is often under 0.02, so an
  absolute threshold cannot separate a hit from a miss.
