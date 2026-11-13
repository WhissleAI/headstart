# Head Start: hard questions, grounded answers

**Sources.** Numbers come from these files:
- **[AB]** `runs/ab_full5.json` (the same file as `runs/ab_latest.json`): the final A/B, 58 spoken questions,
  13 voices.
- **[ROB]** `runs/robustness.md`.
- **[CONV]** `runs/conv.json` and `runs/conv_nomem.json`.
- **[ERR]** `runs/error_analysis.md`.
- **[IDX]** `index/README.md`.
- **[PROG]** `PROGRESS.md`.
- **[DEMO]** `DEMO_CHECKLIST.md`.

Code references point into `server/`.

---

### 1. Is the speedup real, or did you tune the baseline to be slow?
The baseline differs from Head Start only in timing. Both use the same Arc ear, the same lanes, the same
EmbeddingGemma 2 reranker, the same answer model, and the same hedged answer request (`HEDGE_MS`, "both modes",
`server/config.py`). In `mode="baseline"`, `server/speculator.py` waits for the endpoint, then searches, reranks and
answers.

Each question runs in both modes, one after the other, and the order alternates (the `order` field in [AB]).
Latency is measured from the ASR timestamp of the last spoken word to the first answer token. The baseline's
endpoint lag (513 ms at p50) is Arc's own pause rule, not an added delay.

Results [AB]:
- p50: 1538 ms (baseline) vs 1160 ms (Head Start).
- p90: 2126 vs 1763 ms.
- Paired median saving: 348 ms. Head Start was faster on 44 of 58 questions.

### 2. Where does Head Start lose?
It was slower on 14 of 58 questions [AB]:
- 12 were speculative misses: what was searched early did not match the final question.
- 1 was a reused draft (q06) and 1 was a speculative hit (q29).
- 8 of the 14 are local-moments questions. On local questions the p50 is almost a wash: 1316 vs 1348 ms. The local
  lane runs on the laptop and is already fast, so there is little to win.

The levers:
- Draft reuse is already at 53 % of questions.
- Waits on in-flight lanes are capped (`PENDING_CAP_S`).
- A policy simulator, which predicts the real p50 to within 3.9 %, lets us sweep the parameters without spending
  real calls [PROG].

### 3. How much more does speculation cost per query?
Head Start makes 2.38 web calls per question against 0.66 for the baseline [AB]. The baseline is lower because local
questions never touch the web. Both are capped:
- **Web calls:** at most 6 speculative calls plus 1 final per question (`WEB_BUDGET`). The most any question used
  in the final run was 6 [AB].
- **Drafts:** about 2 per question (1.98). Each is capped at 160 output tokens and cancelled as soon as the meaning
  moves on.

On Tavily, each call is a `fast` search at 1 credit [PROG]. That puts Head Start at **about 2.4 credits per
question**. This is an estimate from the measured call count; the fake-Tavily simulation measured 2.8 credits against
1 [PROG]. The credit limits:
- A session budget of 120 credits, and speculation stops 2 credits early, so the final query always has credit [PROG].
- A process cap of 800 credits, below the free plan's 1000 per month [PROG].
- A cache that prevents paying twice for the same query [PROG].

The trade is roughly 1.7 extra search calls for 348 ms off the median.

### 4. What happens when the ASR mishears?
Every ASR revision rebuilds the query frame. If the content words of a running search are no longer in the frame,
`_mark_stale` cancels it. Its results stay in the pool for the reranker, but they don't feed drafts while fresh
results exist.

On q01 the ASR heard "founded condo", then "founded company acronis", then "nvidia". Two searches went stale, and
the third fed the draft that was reused: 715 vs 1390 ms [AB].

When the key word never comes through, neither mode can recover. ASR is the largest root cause of wrong answers:
107 of the 225 wrong or ungrounded rows across all runs [ERR]. Two things help:
- A pluggable Deepgram ear (`EAR=deepgram`) is built. It falls back to Arc if it has no key or the connection drops.
  It has only been tested against a fake server so far [PROG].
- Entity-only queries and a relaxed retry recover part of what survives the mishearing [ERR].

### 5. How do you handle "Canada, actually make that Turkey"?
`server/frame.py` keeps a list of correction cues: "I mean", "sorry", "no wait", "actually", "scratch that" and
others. It also covers the cues as the ASR actually hears them: "Sari, I mean", "no weight", "I met". The corrected
entity replaces the old one before the query is built.

On q27, Head Start fired "capital canada", "capital make", "capital turnover" and "capital turkey". The first three
went stale, and the answer came from a reused draft: 1213 vs 1555 ms [AB]. q33 ("president of France, no wait, of
the United States") is now right in both modes, and Head Start answered it in 1139 vs 2070 ms [AB].

Across all 8 correction questions: 1228 vs 1599 ms at p50, 100 % accurate in both modes [AB].

### 6. Why a 10M tagger rather than asking the answer model to route?
Time. Arc Tagger runs as native C in int8, at 0.7 ms per word and 3 ms per revision [PROG]. That is cheap enough to
re-run on every ASR partial, about every 100 ms. An answer-model router takes about 500 ms per call (README), which
is more than the whole saving.

The tagger returns typed fields: intent, slots, dialogue act and `command_complete`. `command_complete` drives the
semantic endpoint, which answered 16 of the 58 questions [AB].

### 7. Won't the semantic endpoint cut people off?
It answers early but never commits. The soft endpoint needs three things:
- the tagger says the request is complete,
- the speaker has been silent long enough,
- the words have held through one ASR refresh.

The speculator stays warm, and if the speaker keeps talking, the longer question is answered again
(`server/session.py`). In the final run, no question was answered twice and none was split [AB].

The gate was tightened after review. That is why semantic endpoints dropped from 29 of 48 in an earlier run to 16 of
58 now, while the median saving stayed in the same range [PROG, AB].

### 8. Does it survive a noisy room?
Mostly, and we can show where it doesn't [ROB]:

| condition | Head Start faster | median saved | accuracy (baseline / Head Start) |
|---|---|---|---|
| 20 dB babble (other people's voices, a busy hackathon room) | 35 of 48 | 440 ms | 77.1 % / 77.1 % |
| speech 1.21× faster | 16 of 20 | 388 ms | equal |
| speech 0.85× slower | 15 of 20 | 486 ms | equal |
| 10 dB (a loud hall) | 32 of 48 | 173 ms | — |

At 20 dB the noise raises Arc's word error rate from 4 % to 15 %. Accuracy drops by the same amount in both modes:
noise hurts the ear, not the speculation.

10 dB is the weak point. The energy VAD never hears silence, so the semantic endpoint almost never fires (2 of 48).
In noise, Head Start also answered neighbours' chatter more often (9 of 48 vs 2 of 48, inside continuous noise at
20 dB). Fixes are in the code: ASR word-gap silence when the room is noisy, and a chatter guard that skips early
answers for utterances 12 dB quieter than the last question. Neither has been re-measured yet [PROG]. For the
demo, a close mic keeps us at 20 dB or better.

### 9. Where is Tavily? The hackathon is co-hosted by them.
The Tavily lane is built and merged, but we don't have a key yet, so every number here uses the keyless Wikipedia
fallback (`web_backend: wikipedia` in [AB]).

What the lane does (all from [PROG]):
- `fast` depth (1 credit) for both speculative and final searches. `auto_parameters` is off so the depth is never
  silently raised.
- Topics routed by intent: news → `news` with a time range, stocks and currency → `finance`, weather → `max_age_hours`.
- Credits metered per fire and reconciled against `usage.credits`.
- A TTL cache, and a circuit breaker that falls back to Wikipedia.

It is tested only against a local fake built from the API docs. In a simulation of a slow lane (fake Tavily at
600–1500 ms), Head Start was still faster on 18 of 18 runs, saving about 300–340 ms. A slow lane can't be hidden
completely because the final query needs the last word.

A 12-question current-events set (scores, markets, weather, headlines) is ready to run the moment the key exists
[ROB]. The simulator has already chosen a Tavily-specific setting ("T2"): −20 ms at p50 and 17 % fewer answer-model
calls in simulation [PROG].

### 10. How do you know the answers are correct, not just fast?
Answers are keyword-scored, with stems and number words allowed (`eval/run_ab.py`). In the final run [AB]:
- Both modes scored 96.6 %.
- Web questions: 100 % in both modes. Local: 90 % in both.
- Paired: 55 correct in both, and each mode had 1 that only it got right.

We went through every wrong or ungrounded row across all runs [ERR]. Of the 225 rows:

| root cause | rows |
|---|---|
| ASR | 107 |
| query construction | 48 |
| answer model | 30 |
| retrieval | 21 |
| scoring artefacts | 19 |

Query construction, answer-model grounding and retrieval were fixed. Replaying offline, 29 of 48 previously wrong
answers came out right and grounded (9 before the fixes), with 20 of 20 regression checks still passing [PROG].

On the local moments lane, a second-stage ranker raised recall@1 from 0.80 to 0.90 on clean text and from 0.65 to
0.80 on what Arc heard [IDX]. The honest claim is **accuracy parity at lower latency**, not an accuracy gain.

---

### Extra: questions to expect from VCs
- **"Who buys this?"** Anyone who already ships voice search: voice assistants, in-car systems, contact-center agent
  assist, and live-media search (the moments lane already indexes sports, court and meeting audio). It sells as a
  latency layer in front of their existing search API and answer model.
- **"Can you ship Arc commercially?"** Not under the current licence. Arc and Arc Tagger are research releases from
  Whissle under CC BY-NC 4.0, so commercial use needs a Whissle licence. EmbeddingGemma 2 is Apache 2.0.
- **"What if the demo network dies?"** `./run_demo.sh` runs a preflight first. Without the network, the web lane
  serves 57 recorded Wikipedia responses and answers are extracted from the top source. Both are labelled as
  degraded in the UI, and the race stays fair because both modes share the same fallbacks [DEMO].
- **"What's next?"**
  - Get the Tavily key, then re-run the A/B and the current-events set.
  - Re-measure 10 dB with the VAD fix.
  - Have a human test the browser mic.
  - Run the eval with real voices, not only TTS.
