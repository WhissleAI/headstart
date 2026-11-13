# Head Start: progress log

Speculative voice search: retrieval and answer drafting start from the third spoken word, so the answer is ready
when the speaker stops. Built for SearchHack (Sunnyvale, Telem AI × Tavily).

## Layout (ownership while parallel agents run)
| path | what | owner |
|---|---|---|
| `server/` | tagger binding, frames, speculator, lanes, answer, session, ear, app | main |
| `web/` | browser UI (mic + replay, live transcript, speculation timeline, A/B race) | UI agent |
| `eval/` | spoken question set (`eval/audio/*.wav`, `eval/questions.jsonl`), A/B harness, charts | eval agent |
| `index/` | local "moments" corpus builder → `index/data/{docs.jsonl,vectors.npz}` | index agent |

Run: `.venv/bin/uvicorn server.app:app --port 8770` (open http://localhost:8770). Keys in `.env` (git-ignored):
`ANTHROPIC_API_KEY` (set), `TAVILY_API_KEY` (missing → web lane uses Wikipedia).

## Measured so far (2026-11-12)
- Arc Tagger, native C int8 via ctypes: 0.7 ms/word, revision 3 ms. weather_query at word 3, Paris→Texas revision OK.
- Arc v1 (Mini) live on M3 Pro: 9 ms per 100 ms chunk; words arrive 0.1–0.4 s after spoken, self-revises.
- EmbeddingGemma 2 (MPS, bf16, warmed): query 30 ms, 5 docs 32 ms, 1.5–4 s audio ~100 ms. Audio↔transcript cos 0.87.
- Wikipedia lane 200–270 ms (tail up to 2.8 s). Haiku 5.5 with thinking disabled + effort low: TTFT ~620 ms
  (default config thinks and can burn the whole max_tokens budget → no text).
- Simulated speech A/B (4 questions): Head Start ready 1.0–1.3 s after last word vs baseline 1.6–2.0 s.

## Open / next
- [ ] Real-audio A/B over the spoken set (Arc ear) → runs/ab_latest.json + chart
- [ ] Local moments index + audio lane demo
- [ ] UI
- [ ] Semantic endpointing (tagger command_complete + short pause) vs Arc pause
- [ ] Tavily key from user
