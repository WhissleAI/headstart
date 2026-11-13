# Head Start: progress log

Speculative voice search: retrieval and answer drafting start from the third spoken word, so the answer is ready
when the speaker stops. Built for SearchHack (Sunnyvale, Telem AI × Tavily). Architecture: README.md.

## Layout
| path | what |
|---|---|
| `server/` | tagger binding (ctypes → vendor/arctag native C), frames, speculator, lanes, answer, session, ear (Arc), app (FastAPI WS) |
| `web/` | browser UI: mic + replay, live transcript, frame card, speculation timeline, A/B race panes, results |
| `eval/` | 42 spoken questions (`questions.jsonl`, `audio/q*.wav`, Kokoro + macOS voices), `run_ab.py` harness |
| `index/` | local moments corpus (35 min real audio, 222 moments, text+audio vectors), local questions, retrieval eval |
| `tests/` | 11 deterministic speculation tests (fake lanes + LLM) |

Run: `.venv/bin/python -m uvicorn server.app:app --port 8770` → http://localhost:8770.
Keys in `.env` (git-ignored): `ANTHROPIC_API_KEY` (set), `TAVILY_API_KEY` (missing → Wikipedia web lane).

## Measured (2026-11-12, M3 Pro)
- Arc Tagger native int8: 0.7 ms/word, revision 3 ms; weather_query at word 3.
- Arc v1 live: 9 ms per 100 ms chunk; words 0.1–0.4 s behind speech; own pause endpoint ~470 ms after last word.
- EmbeddingGemma 2: MPS bf16 docs ~30–160 ms; CPU text-only copy (271M) for queries 43 ms, identical vectors.
- Haiku 5.5: default config thinks (can burn max_tokens with no text). thinking=disabled + effort=low → TTFT ~620 ms.
- Local retrieval (10 q): text r@1 0.7 / r@4 0.9; whole spoken-question AUDIO → index r@1 0.7 (text vecs) / r@4 1.0
  (max) — no ASR needed; first 1.5 s of audio useless (r@1 ≤ 0.1).
- Real-audio spot checks (quiet machine): HS 0.70–1.07 s vs BL 1.5–2.4 s on Nvidia/Eiffel/Apollo; local q05 0.75 vs 1.12 s.

## Real-audio A/B (eval/run_ab.py, 48 spoken questions, 13 voices, Arc ear, Wikipedia web lane, Haiku 5.5)
| run | baseline p50 / mean / p90 | Head Start p50 / mean / p90 | acc BL / HS | HS faster | notes |
|---|---|---|---|---|---|
| full1 (17:00) | 1600 / 1677 / 2104 | 1473 / 1663 / 2663 | 85.4 / 81.2 % | 26/48 | semantic endpoint threw away the speculator when the speaker paused → 13 re-answers from cold |
| full2 (17:25) | 1546 / 1586 / 1866 | **1254 / 1216 / 1831** | 83.3 / **85.4 %** | **37/48** | soft endpoint keeps speculator warm; paired median saved **435 ms**; drafts reused 52 %; sources p50 568 vs 926 ms; 2.9 vs 0.8 web calls |
Chart: runs/ab_latest.png. Remaining slow cases: API TTFT outliers (q26 1.3 s) → hedged requests added (both modes).

## Fixed along the way (from tests, traces and agent reports)
- Draft built from stale or local-only results → drafts wait for all lanes of their own query, fresh docs preferred.
- Fire marked done before its docs were embedded → done only after vectors.
- Semantic endpoint on misheard partial → words must hold one ASR refresh or be refreshed after silence began.
- Noise floor learned speech as "silence" → floor clamped ≤ 0.01.
- Same utterance answered twice after semantic endpoint → partials of an answered utterance ignored.
- Every web question also searched local moments → route is web | local only.
- Filler words in queries ("wondering") → stopwords extended. Final arriving while busy → queued.
- Audio lane hogged the GPU queue → only for local-routed questions, ≥1.8 s of speech.

## Conversation memory (17:45) — eval/run_conv.py, 6 two-turn spoken conversations, follow-up uses a pronoun
- Follow-up "it/its/they/there" → the previous question's thing; "he/she" → a person, usually named only in the
  previous ANSWER (tagged with a second Arc Tagger handle). Query, ranking text and answer prompt use the resolved
  question; the previous turn's docs (already embedded) seed the next speculator.
- With memory, HS follow-ups: 4/5 scored correct and grounded (queries "nvidia founded", "albert einstein born"),
  ready 0.82–1.34 s; baseline 1.15–3.0 s. Without memory the searches are bare ("founded", "museum") and the
  model falls back to unsourced knowledge ("the sources don't say...") at 1.4–1.7 s.
- conv03 failed in both: Arc heard turn 1 as "Coron pride and prejudice" (ASR, not memory).

## Verified end to end (17:40)
- Live server, Race mode, WS replay: q01 HS 1139 vs BL 1755 ms; q27 (correction) 1110 vs 1409.
- Mic path (server side; Float32 frames over /ws, 3 questions in one session): 1021/457/626 vs 1361/1114/1259 ms.
- Browser mic capture itself (AudioWorklet) NOT yet exercised: needs a human to grant mic permission.

## Agent wave 2 (launched 17:40)
- policy-sim (branch policy-sim): Arc trace recorder + virtual-time simulator + parameter sweep (wikipedia/tavily profiles)
- ear-deepgram (branch ear-deepgram): pluggable ear, Deepgram live ear ready for a key, fake-server tests
- tavily-lane (branch tavily-lane): API verification, intent→topic routing, credits, cache, fake server
- robustness (main, eval/ + runs/): noise 20/10 dB, speaking rate, current-events set; owns CPU via runs/.run_ab.lock
- reviewer (read-only): independent correctness review of server/
- pitch (docs/): PITCH.md, JUDGE_QA.md, demo GIF + screenshots

## Open / next
- [ ] Tavily key from user (sponsor); optional Deepgram key (Arc Mini mishears some voices)
- [ ] Human check of the browser mic path
- [x] Correction cues as ASR hears them ("Sari, I mean", "no weight", "I met")
- [x] Demo script (DEMO.md)
