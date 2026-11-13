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

## HEADLINE ON TAVILY (21:55; real key; 70 spoken questions = 58 scored + 12 current events) — runs/ab_tavily.json = runs/ab_latest.*
| | baseline | Head Start |
|---|---|---|
| p50 / mean / p90 ready | 1726 / 1780 / 2399 ms | **1320 / 1357 / 2036 ms** |
| sources p50 | 926 ms | **703 ms** |
| accuracy (58 scored) | 93.1 % | 93.1 % |
| Tavily credits / question | 0.69 | 2.43 |
Paired: median saved 288 ms, faster on 51/70. By kind (paired median saved, faster): current events 744 ms 10/12
(HS 1074 vs BL 2178 p50), filler 618 ms 7/8, correction 556 ms 5/8, factoid 372 ms 20/22, local −110 ms 9/20 (local
regressed with the Tavily-tuned knobs — DRAFT_BUDGET 2 / SEM_STABLE 550 — not yet investigated).
Probe (eval/tavily_probe.py): fast p50 210 ms / p90 514, 1 credit; include_answer +760 ms → kept off. Tavily caches
repeat queries server-side (~0.1 s): the harness alternates mode order, which balances it on average.

## FINAL HEADLINE (Wikipedia lane): full5 (20:00; all 8 agent branches merged; 108 tests; 58 spoken questions = 38 web (22 factoid, 8 filler, 8 correction) + 20 local) — runs/ab_latest.*
| | baseline | Head Start |
|---|---|---|
| p50 / mean / p90 ready | 1538 / 1622 / 2126 ms | **1160 / 1247 / 1763 ms** |
| sources p50 | 880 ms | **650 ms** |
| accuracy | 96.6 % | 96.6 % (web 100/100 %, local 90/90 %) |
HS slower on 14/58 (8 local: local p50 1316 vs 1348 — little gain on that lane).
Paired: median saved **348 ms**, mean 375 ms, faster on **44/58**; fillers 1094 vs 1854 ms; corrections 1228 vs 1599 ms.
run_demo.sh preflight: GO (q01 1189 vs 1646 ms; local_05 686 vs 1351 ms).

## Earlier headline: full4 (19:40; review fixes + answer quality; Wikipedia lane; 48 spoken questions) — runs/ab_latest.*
| | baseline | Head Start |
|---|---|---|
| p50 / mean / p90 ready | 1558 / 1561 / 1806 ms | **1176 / 1171 / 1582 ms** |
| sources p50 | 918 ms | **626 ms** |
| accuracy | 93.8 % | **97.9 %** |
Paired: median saved **381 ms**, mean 390 ms, faster on **38/48**. 83 tests pass.

## Robustness (post-fix code; runs/robustness.md)
| condition | BL p50 | HS p50 | acc BL/HS | Arc WER | HS faster (median saved) |
|---|---|---|---|---|---|
| clean | 1478 | 1152 | 93.8/93.8 % | 4.1 % | 37/48 (398 ms) |
| 20 dB babble | 1546 | 1174 | 77.1/77.1 % | 15 % | 35/48 (440 ms) |
| 10 dB babble | 1567 | 1505 | 54.2/60.4 % | 41 % | 32/48 (173 ms) |
| speech 1.21× (20 q) | 1440 | 1106 | 85/85 % | 5.5 % | 16/20 (388 ms) |
| speech 0.85× (20 q) | 1559 | 1148 | 95/95 % | 5.8 % | 15/20 (486 ms) |
- 10 dB loses the semantic endpoint (energy VAD never hears silence) → fixed 19:30: ASR word-gap silence when the room
  is noisy; chatter guard (no early answer for utterances 12 dB quieter than the last question). Not yet re-measured.
- 10 dB re-run after the fix (runs/ab_snr10_fix.json): HS 1437 vs BL 1618 ms p50, saved 212 ms (was 173), faster
  32/48; accuracy 72.9 / 77.1 %. Semantic endpoint still 2/48: the noise is BABBLE (other voices), Arc transcribes
  stray words from it, so the last-word time keeps advancing and no pause is seen. Speech-vs-speech separation at
  10 dB is out of scope; the demo uses a close mic.
- Demo advice: close mic (headset/phone) keeps the room ≥ 20 dB; push-to-talk avoids neighbours' chatter.

## Answer quality (merged 19:20; runs/error_analysis.md)
- 225 wrong/ungrounded rows across all runs: ASR 107, query construction 48, answer model 30, retrieval 21, scoring 19.
- Fixes: fillers/contractions, mangled correction cues, entity phrase queries + Wikipedia entity lane, zero-hit retry,
  no-disclaimer prompt. Offline replay: previously-wrong 9→29/48 right+grounded; regression 20/20.

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

## Arc Tagger depth (branch tagger-depth)
The frame now reads the rest of Arc Tagger's per-word read-out (acts, device-directed, typed number/time spans, entity
types, span-level PII, jailbreak) and the UI shows it in a "read-out" row of the frame card.
- **Finance bug fixed (main):** Tavily answers 400 "Finance topic is not supported for fast or ultra-fast" — every
  qa_stock / qa_currency question failed over to Wikipedia. web_plan now sends topic=finance at `TAVILY_FINANCE_DEPTH`
  (basic, same 1 credit). runs/replay_finance.json (5 q): grounded 4/5 → 5/5, S&P/Bitcoin/Nvidia answers dated today;
  cost ~+1.3 s per finance call (speculation hides it on the Arc side; the baseline pays it).
- **Freshness from typed time spans:** a result question (game/race/score/won) said with a relative time ("last night",
  "this past weekend", "right now") → topic news + time_range day|week; money/crypto/market questions → finance;
  absolute years stay general. Arc's weather intent with no weather word (ibuprofen, Japan visa) no longer gets the
  6 h max_age. runs/replay_tagger.json: 11/42 questions change plan; sources published ≤7 days 0/40 → 25/30 (news
  ones); grounded 10/11 → 6/11 (3 finance were the 400 above; 3 news answers now say "sources don't give the score");
  blind judge after 4 / before 7 — mixed, no gold for current events. `TAGGER_HINTS=0` turns it off.
- **Privacy:** phone / email / street address / card / gov id / dob / handle spans never reach the web query; the
  answer model sees "[private]". private_name is NOT used (fires on Nvidia, Mount, Toyota, Mozart); utterance-level
  pii.request / pii.leak fire on most ordinary questions.
- **Jailbreak:** safety.jailbreak.attempt (1.0 on the injection probe, off on all 80 others) → no speculative search.
- **Not used (measured):** device_directed is 0.82–0.92 on chatter vs 0.95–1.0 on questions (no margin); entity
  types are noisy (Mona Lisa / Jaws = person, penicillin = org). Gate: six act/intent-completeness rules in
  eval/gate_eval.py; none dominates the shipped gate (each +complete costs +early), so the gate is unchanged.
- Sim (tavily profile, 58 traces × 5 seeds): identical to main (p50 1598 / p90 2295 ms, paths hit 53 % / miss 26 % /
  draft 21 %, grounded 98.6 %): the eval traces contain no PII / money / injection, and the sim's web lane ignores the
  Tavily plan, so freshness/topic effects are measured only by the replay.

## Search narrowing from the Arc read-out (branch search-narrowing; server/narrow.py, eval/narrow_eval.py)
Offline, real Tavily (fast, 1 credit), 80 questions (30 showcase + 12 current + 38 web), heard text replayed word by
word through the native tagger; the speculator's fire policy replayed on the partial frames. Blind judge (Haiku 5.5,
fixed 0/1/2 rubric, every distinct result graded once, shuffled, no variant labels): runs/narrow_judge.json.
283 Tavily credits, judge calls est. 150-190 (one per question per 25-result chunk, 3 passes; not counted), 400 answer calls. runs/narrow_eval.json (+ runs/narrow_cache.json: re-analysis
costs 0 credits). Families: Arc intent where reliable (weather / news / stock), else content words — Arc's fine
intents misfire here ("screenshot on a Mac" = cooking_recipe, "ramen places" = general_joke, ibuprofen = weather).

Top-5 relevance, paired on the questions whose final results changed (g@5 = mean grade of the top 5, 0-2):
| variant | n changed | P@5 | g@5 | top-1 graded 2 | nDCG@5 | a "2" in top 5 | Tavily ms (paired p50) |
|---|---|---|---|---|---|---|---|
| include_domains **prefer**, all families | 32 | .76 → .76 | 1.28 → 1.18 | **.41 → .63** | .82 → .90 | .88 → .78 | **−88 ms** (n=74) |
| include_domains **restrict** | 36 | .73 → **.59** | 1.21 → **.91** | .39 → .39 | .81 → .80 | .86 → **.58** | **−219 ms** (n=36) |
| query shaping (generic words out, family noun, explicit dates) | 23 | .67 → **.82** | 1.08 → 1.20 | .35 → .39 | .78 → .80 | .83 → .83 | — |
| shaping + prefer, all families | 36 | .74 → .79 | 1.22 → 1.21 | .39 → .61 | .80 → .89 | .86 → .78 | |
| **shaping + prefer on finance/weather/travel/reviews/local/health/recipe/home (default)** | 33 | .73 → **.80** | 1.19 → **1.26** | .39 → **.58** | .80 → **.86** | .85 → .79 | |
All 80: base P@5 .797 / top-1 .519 / nDCG .830 → default .828 / .595 / .854.
- Domains per family (prefer): helped finance (+0.24 g@5), weather w/ shaping (+0.17), travel, reviews; hurt sports
  (−0.9: ESPN/nba.com pages are schedules/hubs, not the result), convert (−0.5), tech (−0.5), definition (−0.3) →
  those are off by default. Family selection is on this same data (n = 1-6 per family): treat as a prior, not proven.
- restrict: fastest (Tavily's own response_time p50 0.44 → 0.20 s) but drops relevant non-listed sources: rejected.
- Answers (Haiku on each variant's final top 5): keyword-correct 50/53 in every variant, grounded 76/80 base vs
  75/80 default — narrowing changes which page is first, not whether the answer model finds the answer.
- Read-out rerank (entity phrase / content coverage / family domain / freshness on Tavily's order): top-1 +1 of 79,
  nDCG −0.001 → no effect; kept behind NARROW_RERANK=0 (Tavily's order already uses the query; the live final
  ranking is EmbeddingGemma cosine over the pool, not re-measured here).
- Speculative fires (showcase sc01–19, real policy): base 58 fires = 3.05 credits/q, 23 useful (a result in the final
  top 5, or the final query itself), 35 wasted. Gate "≥2 content words" (default) drops the lone-entity-prefix fires
  ("kansas", "manchester", "apple", "$100"): 4–6 fires, 0 useful lost. "≥3" would cut 38 % of speculative credits
  (−1.16/q) but loses 2 useful fires and 1 of 9 early top-1 hits (NARROW_GATE_MIN_CONTENT=3).
- Results while you speak (earliest word whose speculative results already contain the final top-1): base 9/19
  questions, words before the end [1,1,1,2,3,5,5,6,6], p50 at 73 % of the utterance; default 8/19,
  [1,1,2,3,3,5,5,5], 70 %. Narrowing does not make the top-1 appear earlier (partial queries rarely have the family).
- Frame fix (all modes): "what does the word ubiquitous actually mean exactly" — "actually" + a continuation verb was
  treated as a self-correction and dropped "ubiquitous" (query "word mean exactly"); now kept.
- Not done: policy simulator for the −88 ms prefer latency (its Tavily lane is a fixed latency model; the shift is
  small next to the 550 ms + lognormal model) and EmbeddingGemma pool ranking on these results.
Flags (server/config.py): NARROW_DOMAINS off|prefer|restrict, NARROW_DOMAIN_FAMILIES, NARROW_QUERY, NARROW_DATES,
NARROW_GATE / NARROW_GATE_MIN_CONTENT, NARROW_RERANK (+ weights). Re-run: `nice -n 19 .venv/bin/python
eval/narrow_eval.py --variants base,query_domsel --spec base,query_domsel --spec-ids sc0,sc1 --judge --answers base,query_domsel`.

## Deepgram Flux ear + Arc metadata (branch flux-ear; server/ear.py FluxEar, server/arcmeta.py)
Flux API facts, checked against the docs AND the live service (developers.deepgram.com/reference/speech-to-text/listen-flux,
/docs/flux/state, /docs/flux/configuration, /docs/flux/voice-agent-eager-eot): `wss://api.deepgram.com/v2/listen`,
`Authorization: Token`, `model=flux-general-en&encoding=linear16&sample_rate=16000`, `eot_threshold` 0.5–1.0 (0.7),
`eager_eot_threshold` 0.3–0.9 (unset = no eager events; must be <= eot), `eot_timeout_ms` 500–60000 (5000), `keyterm`.
Server sends `Connected`, then `TurnInfo` {event Update|StartOfTurn|EagerEndOfTurn|TurnResumed|EndOfTurn, turn_index,
audio_window_*, transcript, words[{word (punctuated, cased), confidence, start, end}] — the WHOLE turn each time,
end_of_turn_confidence, trigger on EndOfTurn: model|manual|timeout}; Update every ~240 ms of audio. Client: binary
audio (80 ms chunks recommended), `ForceEndTurn`, `CloseStream`, `Configure`; no KeepAlive message (we send silence).
EndOfTurn transcript == the preceding eager transcript unless a TurnResumed came between (docs; observed).

Mapping: Update/StartOfTurn/TurnResumed → on_partial; EagerEndOfTurn → Session.on_eager → soft answer
(FLUX_SOFT=eager, default); EndOfTurn → on_final (confirms the soft answer, or answers the longer question); flush →
ForceEndTurn. Fallback to Arc on connect failure / drop (shared with the nova-3 ear). Defaults = lowest latency:
eager 0.3 (docs min), eot 0.7, eot_timeout 1000 ms.

Real Flux traces (eval/record_flux_traces.py → runs/flux_traces.jsonl, 80 questions = eval set + showcase, real time):
- WER 0.020 (15/742 words) vs Arc Mini builtin 0.123 on the same questions.
- EndOfTurn after the last word: p50 652 / mean 671 / p90 1091 ms (78 model, 5 timeout triggers). Arc's own pause
  endpoint is ~470–600 ms: Flux's FINAL is not faster; its EAGER is.
- 3/80 questions split into two turns by EndOfTurn (trigger=model, conf 0.78–0.80): "weather in London, sorry, I mean
  Tokyo", "Boston sorry / I mean Denver", "which one's better for photos / the iPhone 17 Pro or the Pixel 10 Pro".
  FLUX_EOT=0.85 (docs' "complex pipeline" setting) would likely avoid 2 of 3 at some confirmation latency; not default.

Soft-endpoint gate on eager events (eval/flux_gate_eval.py --arc; trigger = answer start − last word end):
| policy | p50 | mean | p90 | premature soft answers /100 q |
|---|---|---|---|---|
| no soft answer (EndOfTurn) | 652 | 671 | 1091 | 0 |
| **eager (default, FLUX_SOFT=eager)** | **212** | **242** | 599 | 26.2 |
| eager, then wait for Arc command_complete (FLUX_SOFT=eager+arc) | 230 | 280 | 627 | 11.2 |
| eager + tagger gate (FLUX_SOFT=eager+gate) | 354 | 430 | 936 | 8.8 |
| eager + Arc cc event within 0.6 s (no waiting) | 315 | 384 | 815 | 3.8 |
| eager + conf>=0.5 (≈ eager_eot 0.5) | 452 | 490 | 1018 | 8.8 |
| eager + Arc for_me>=0.5 | 234 | 264 | 613 | 25.0 (for-me doesn't help) |
Premature = mostly self-corrections and trailing qualifiers; a premature soft answer costs an answer call and is
superseded (the speculator stays warm), not latency. Arc's command_complete measurably helps (premature 26→11 for
+18 ms p50) → shipped as the opt-in FLUX_SOFT=eager+arc; the tagger gate costs +140 ms p50.

Arc metadata without its decoder (ARC_META=mini|transcriber|off; external-transcript mode on Flux's words,
fallback=None so Arc's builtin decoder never runs; RTF = compute / audio, M3 Pro, real Flux words):
| variant | load | RAM (peak RSS) | CPU per 100 ms chunk | RTF | command_complete after last word |
|---|---|---|---|---|---|
| Arc Mini v1 (1.0.1, already loaded for the Arc ear) | 2.4 s | 0.7 GB (no extra) | 10 ms p50 / 15 max | 0.09–0.10 (0.26 live, contended) | +160–300 ms |
| Arc v1.1 Mini (arc_transcriber 1.2.1) | 3.2 s | 1.0 GB | 10.5 / 14 ms | 0.10 | +160–200 ms; adds audio events |
| Arc Transcriber v1.1 full, CPU, model's 10 s stream chunks | 5 s | 3.3 GB | 0 until flush, flush 0.6–1.0 s | — | only at flush |
| Arc Transcriber, CPU, 1 s chunk / 1 s ctx | 5 s | 3.3 GB | 600–870 ms spikes | 0.39 | +1.3 s |
| Arc Transcriber, MPS, 0.5 s chunk / 1 s ctx | 6 s | 3.1 GB | p95 81 ms, cold 1.1 s | 0.26 | +1.2–1.4 s |
Transcriber (Parakeet-TDT 0.6B tap 12 + adapters) needs arc_transcriber ≥1.2 (local 1.0.1 can't load v1.1 configs;
loaded as `arc_audio_t`). It "runs" in real time on MPS but its metadata lands >1 s after the words → read-out only,
useless for endpointing. Default ARC_META=mini. Read-out: `arc` events + `frame.arc` {for_me, cc_p, cc_events, act,
intent, speaker, role, bio, audio_events, rtf}.

Real A/B EAR=flux vs EAR=arc (eval/run_ab.py, showcase sc01–sc10, WEB_BACKEND=wikipedia because the Tavily key returns
432 "usage limit"; sc06 'current' skipped on Wikipedia → 9 q; times from the VAD speech end so both ears are comparable):
| ear | mode | ready p50 / mean | endpoint after speech end p50 | WER | endpoint |
|---|---|---|---|---|---|
| arc | baseline | 2366 / 2763 | 728 | 0.037 | ear 9 |
| arc | headstart | 2033 / 1878 | 632 | 0.037 | tagger 5, ear 4 |
| flux | baseline | 2174 / 2063 | 688 | 0.000 | ear 9 |
| **flux** | **headstart** | **1197 / 1167** | **244** | **0.000** | eager 9 |
(Same 10 with the 432 Tavily key, i.e. no speculation: flux HS 1640/1830 vs arc HS 2676/2901; endpoint 261 vs 630.)
runs/ab_ear_{arc,flux}{,_wiki}.json.
