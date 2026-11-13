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
| `tests/` | deterministic speculation tests (fake lanes + LLM) + Tavily lane tests against `fake_tavily.py` |

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
- With memory, HS follow-ups: 3/5 scored correct (runs/conv.json; baseline 2/5; conv06 expects Tokyo, answer named Sony Interactive) and grounded (queries "nvidia founded", "albert einstein born"),
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

## Pluggable ear (ear-deepgram branch)
- `server/ear.py`: `make_ear()` picks `EAR=arc|deepgram`. DeepgramEar = raw websocket to `/v1/listen` (nova-3,
  interim_results, endpointing 300 ms, utterance_end_ms 1000, smart_format, punctuate), KeepAlive every 3 s, Finalize on
  flush, CloseStream on close; fallback to Arc on no key / connect failure / drop (re-feeds audio since the last final).
- `tests/test_ear_deepgram.py`: fake Deepgram websocket server (documented message shapes) -> DeepgramEar -> two race
  Sessions; revisions, finals, timestamps, identical words, keepalive, Finalize, fallback. Not yet run against real Deepgram.

## Tavily web lane (tavily-lane branch) — built against the docs + a local fake, no key yet
Docs verified 2026-11-12: docs.tavily.com/documentation/api-reference/endpoint/search, /rate-limits, /api-credits,
/best-practices/best-practices-search.
- Request: `search_depth` speculative `fast` (1 credit, reranked ~500-char chunks, `chunks_per_source` 2), final `fast`
  too (on the critical path; baseline uses the same final depth, same lanes). `advanced` = 2 credits + highest latency:
  never by default. `auto_parameters` false (it can silently raise depth to advanced), `include_usage` true (billed
  credits come back in `usage.credits` and reconcile our estimate), `max_results` 5, query cut to 400 chars.
- Topic by intent: news_query → `news` (+ `time_range` day for today/tonight/breaking, week for latest/yesterday/this
  week); qa_stock/qa_currency → `finance` + `max_age_hours` 1; weather → `general` + `max_age_hours` 6 (docs: time_range
  drops weather/price pages because they keep old dates). 434 (flex capacity, only with max_age_hours) → retried once
  without it.
- Credits: per fire (meter), per question in the metric (`credits`, `credits_spec`, `credits_final`, `web_cached`,
  `web_backend`, `session_credits`), per mode in run_ab summaries (`mean_credits*`, `total_credits`). Session budget
  120 (speculation stops 2 credits early so the final query always has credit), process cap 800 (free plan = 1000/mo).
  Out of budget / breaker → final query falls back to Wikipedia, speculative fire skipped.
- Errors: 429 (Retry-After) / 432 / 433 / 401 trip a process breaker → Wikipedia until it clears; 5xx/400/timeout → that
  fire is a lane error, a failed final falls back to Wikipedia. Failed calls are refunded in the ledger.
- Cache: TTL (900 s; 120 s for news/finance/weather), key = sorted content words + params, per session by default
  (A/B-honest: baseline never sees Head Start's results; TAVILY_CACHE_SCOPE=global for a demo booth). In-flight
  requests are shared; a stale fire's request (already paid) is not aborted, it lands in the cache.
- HTTP/2 (h2 installed) + 120 s keep-alive on the shared httpx client; hard total deadline TAVILY_TIMEOUT_S=4.
- Fake: tests/fake_tavily.py (MockTransport in-process, or `uvicorn --app-dir tests fake_tavily:app` +
  TAVILY_BASE_URL); lognormal latency per depth (placeholder medians: ultra-fast 320, fast 480, basic 800,
  advanced 1600 ms), paired per query so A/B modes see the same latency.
- Slow-lane sim (fake Tavily 600-1500 ms, 6 questions x 3 seeds, FakeStream TTFT 300 ms): Head Start faster 18/18,
  ~300-340 ms saved (vs ~435 ms real with Wikipedia): the final query needs the last word, so a slow lane can't be
  hidden fully; HS wins by firing it when the word is visible instead of at the endpoint. 2.8 credits/q vs 1.
- include_answer race (TAVILY_ANSWER=1, off): on the after-endpoint path (miss + baseline, both modes for fairness) a
  parallel `include_answer: basic` call races Haiku; first text wins. Wins only if Tavily's answer overhead < our
  embed + Haiku TTFT (~620 ms) — unmeasured; costs +1 credit per miss. Answer has no [n] citations.
- When a key lands: `eval/tavily_probe.py --yes` (~54 credits) → p50 per depth, include_answer overhead, response keys;
  then `eval/run_ab.py` with the Tavily lane.

## Post-fix real A/B (full3, 18:55, all merges + review fixes, Wikipedia lane, 48 spoken questions)
| | baseline | Head Start |
|---|---|---|
| p50 / mean / p90 ready | 1478 / 1523 / 1830 ms | **1152 / 1144 / 1662 ms** |
| sources p50 | 880 ms | **644 ms** |
| accuracy | 91.7 % | 91.7 % |
Paired: median saved **398 ms**, mean 379 ms, faster on **37/48**; drafts reused 52 %; 0 answered twice; endpoint
semantic 16 / ear 32 (stricter gate). runs/ab_full3.json.

## Policy simulator (merged; sim/)
- Replays 51 recorded Arc traces into the real Session/Speculator on a virtual clock (~6 ms/question).
- Post-fix validation vs full3: HS p50 err 45 ms (3.9 %), per-question corr 0.82, path agreement 85 %, paired
  saved sim 412 vs real 398 ms.
- Sweep → Wikipedia: current config is within 10 ms of best. Tavily: T2 (DRAFT 2, SEM_SILENCE 200, SEM_STABLE 550,
  SEM_COVER 0.4, HEDGE 1000) → −20 ms p50, ungrounded 4.0→3.3 %, −17 % answer-model calls; now the default when
  WEB_BACKEND=tavily. Aggressive W1-on-Tavily would be −100 ms but 17 % ungrounded → rejected.

## Merged (18:00)
- ear-deepgram → main (21 tests then). tavily-lane → main (42 tests). UI glitch round committed (paired-median tile,
  label collision avoidance, settled pipeline strip). Pitch docs + demo media in docs/.
- Deferred: tavily-lane's eval/run_ab.py credit columns (runs/deferred/run_ab.tavily.py) until the robustness agent
  commits its run_ab.py edits.
- In flight: review-fixes (10 confirmed bugs incl. hedge busy-spin, cancel-before-start hang, session wedge on
  exception, overlapping soft answers, open semantic gate), policy-sim, robustness (snr20/snr10/rate).

## Local moments: second-stage ranker + moment citations (branch local-moments)
Diagnosis of the local rows in ab_full2/ab_full3 (0.6–0.7 accuracy): mostly **scoring**, then retrieval, then ASR.
- Scoring (6 of the misses): right answer, strict `expect` — "nailed" vs `nails`, "fifteen" vs `15`, "clearly the
  winner" vs `won`, "prepping" vs `prep`. Gold amended with alternatives (`nails|nailed`, `15|fifteen`, `winner|won`,
  `prep|prepping|preparation`); numbers below give both.
- Ranking: gold at cosine rank 3–5 (local01 Hunter: four adjacent Hunter moments; local04 Lonzo Ball: the ASR wrote
  "Ponto ball"; local02 objection: rank 3). ASR: local06 "office's wedding" (baseline missed scotus-002 entirely),
  local09 "Debate with Trump" (Arc dropped "prep": gold not retrievable from the words), "Samuel one".
- Answer: right moment, model said "the transcript is garbled"/"sources don't say".
Changes: `server/rerank.py` (text cosine + 0.05·lexical/tags/role/event + 0.2·audio-lane cosine over the whole index,
0.2 ms), used for the final ranking and drafts of local-routed questions in both modes (`RERANK_LOCAL`, `RERANK_AUDIO`);
the audio lane keeps its spoken-question vector for it. Answer prompt: moment lines read
`[1] NBA broadcast at 00:26, play-by-play announcer: …before [moment] after…`, and the system prompt asks for when +
who + a verbatim quote (drafts use the same prompt, so draft reuse keeps it). 10 more questions (local11–20), traces.
Retrieval (index/eval_local.py, 20 q, Arc ASR text): r@1 0.65 → 0.80, r@4 0.90 → 1.00, MRR 0.78 → 0.88 (Head Start);
clean text r@1 0.80 → 0.90 (baseline ranker). E2E replay (index/replay_local.py: real traces + real audio lane in the
simulator, real Haiku, one sample each, 100 calls): gold moment at source #1 HS 13 → 16 / 20, BL 11 → 15 / 20, gold
in sources BL 18 → 20; answers citing a mm:ss HS 4 → 13, BL 3 → 14 / 20 (q11–20 measured with the first prompt wording); accuracy (amended expect) HS 20 → 19, BL 17 → 19 of 20;
strict expect q01–10 HS 8 → 9, BL 7 → 9. Single-sample LLM noise is ±1–2 here: accuracy is at ceiling on the
amended gold; the ranker's win shows in which moment is cited first. Remaining: local09 (ASR drops "prep").

## Open / next
- [ ] Tavily key from user (sponsor); optional Deepgram key (Arc Mini mishears some voices)
- [ ] Human check of the browser mic path
- [x] Correction cues as ASR hears them ("Sari, I mean", "no weight", "I met")
- [x] Demo script (DEMO.md)
