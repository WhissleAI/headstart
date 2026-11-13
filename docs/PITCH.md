# Head Start: pitch

**One line:** Head Start searches while you are still talking, so the answer is ready when you stop.

Every number below comes from a file in this repo. Source key:
- **[AB]** `runs/ab_full2.json` (`summary`): real-audio A/B, 48 spoken questions, 13 voices, run 2026-11-12. `runs/ab_latest.json` has the same summary, and `runs/ab_latest.png` is its chart.
- **[AB1]** `runs/ab_full1.json`: the earlier full run, before the soft endpoint.
- **[CONV]** `runs/conv.json` (with memory) and `runs/conv_nomem.json` (without): 6 two-turn spoken conversations.
- **[IDX]** `index/README.md` and `index/data/retrieval_eval.json`: local moments index and audio-retrieval eval.
- **[PROG]** `PROGRESS.md` "Measured" section: component timings on an M3 Pro.

---

## Slide 1: The problem
**Voice search waits for silence.**

- Today's assistants run in sequence: wait for the speaker to stop, detect the pause, search, then think. The whole
  wait comes after the last word.
- In our own baseline (same ear, same search, same answer model) that wait is **1.55 s at p50 and 1.87 s at p90**
  from the end of the last spoken word to the first answer token [AB].
- **0.51 s** of that is spent just deciding the speaker has stopped (baseline p50 endpoint lag 510 ms) [AB].
- Text search became instant years ago. Voice search still has a second and a half of dead air.

## Slide 2: The insight
**Most of the meaning arrives before the speaker finishes.**

- "Who founded the company…": the intent is known by word 3, and only the entity is still missing.
- A 10M-parameter on-device tagger (**Arc Tagger**) reads the partial sentence in **0.7 ms per word** and returns the
  intent, typed slots, self-corrections and whether the request is complete [PROG]. Routing costs nothing, so we
  can re-route on every word.
- So we treat search like a CPU treats a branch: **speculate early, cancel when wrong, and pay only latency we would
  have paid anyway.**

## Slide 3: How it works
```
mic → Arc (on-device streaming ASR, self-revising words)
    → Arc Tagger (intent · slots · corrections · complete?)   ~1 ms/word
    → Speculator: fire / cancel stale / draft / reuse          ← the search intelligence
    → lanes: web (Tavily API; Wikipedia in these runs) · local moments · spoken-audio lane (EmbeddingGemma 2)
    → answer model streams a 1–2 sentence cited answer
```
- **Fire** once the frame holds at least 3 words and real content (budget: 6 speculative web calls per question,
  plus 1 final; `server/config.py`).
- **Cancel stale**: when the ASR revises a word ("acronis" → "nvidia") or the speaker corrects themselves ("Canada,
  actually make that Turkey"), the in-flight search is cancelled and marked on the timeline.
- **Draft early**: when every lane for the current meaning has returned, the answer is drafted in the background and
  reused at the endpoint if the meaning still matches.
- **Semantic endpoint**: answer when the tagger says the request is complete and there has been 250 ms of silence,
  instead of waiting for the ASR's ~470 ms pause rule [PROG]. The speculator stays warm in case the speaker continues.
- **Hears the question itself**: EmbeddingGemma 2 (Google, Apache 2.0) puts speech and text in one vector space.
  The whole-question audio retrieves the right local moment in the top 4 for **10/10** questions (R@4 1.00) with no
  ASR at all [IDX].
- **Conversation memory**: "When was *it* founded?" resolves to Nvidia, using the previous question or answer.

## Slide 4: Evidence (real audio, controlled A/B) [AB]
"Same ear, same search, same reranker, same answer model. The only difference is *when* we start."

| 48 spoken questions, 13 voices | Baseline | **Head Start** |
|---|---|---|
| p50 time to answer (from the last spoken word) | 1546 ms | **1254 ms** |
| mean | 1586 ms | **1216 ms** |
| p90 | 1866 ms | 1831 ms |
| answer correct (keyword-scored) | 83.3 % | **85.4 %** |
| sources ready, p50 | 926 ms | **568 ms** |
| web calls per question | 0.79 | 2.94 |

- **Paired:** Head Start was faster on **37/48** questions; median saving **435 ms**, mean **370 ms**.
  Accuracy: 39 both correct, 2 correct only in Head Start, 1 correct only in baseline.
- **Where it shines:** questions with fillers, p50 **944 vs 1624 ms**; self-corrections, **1142 vs 1648 ms** at equal
  accuracy (75 % each).
- **How the answers were produced:** 25 reused a draft, 11 were speculative hits, 11 were speculative misses, 1 was
  committed from the pool. Drafts were reused on **52 %** of questions. 29 of 48 endpoints were semantic.
- **Iteration is measured:** the first full run [AB1] saved only 106 ms at the median (26/48 faster), because the
  semantic endpoint threw the speculator away when the speaker paused. Keeping it warm took that to 435 ms [AB].
- **Follow-ups with memory [CONV]:** Head Start answered the second turn in **0.82–1.34 s**, against **1.15–3.01 s**
  for the baseline. The memory changes the search itself: "nvidia founded" and "albert einstein born" instead of
  the bare "founded" and "born" [CONV vs conv_nomem].
- Chart: `runs/ab_latest.png`. Demo capture: `docs/media/headstart_race.gif`.

## Slide 5: Why now, and the impact
- **Why now:** three things only recently became true together.
  1. Streaming on-device ASR that emits revisable partials every ~100 ms (Arc v1: 9 ms of compute per 100 ms chunk) [PROG].
  2. Language understanding cheap enough to run on every partial (Arc Tagger at 0.7 ms per word, on the laptop) [PROG].
  3. One embedding space for speech and text (EmbeddingGemma 2), so retrieval can start from sound.
- **Impact:** about a 0.4 s median saving on every spoken query, up to **1.11 s** on the worst baseline case
  (q42: 1298 vs 2407 ms; both answers on q42 were wrong) [AB]. It adds no extra hardware: the tagger and ASR run on the device, and speculation
  only costs search calls, which are budgeted. The same layer drops in front of any search API, Tavily included,
  for voice assistants, in-car systems, call-center agent assist, and live search over meetings and broadcasts
  (the local "moments" lane).
- **The bigger idea:** search intelligence moves *upstream* of the query. The system decides when, what and
  whether to search while the query is still being formed.

---

## Three likely judge questions

**1. "Speculation must be expensive. What does it cost?"**
About 2.2 extra web calls per question: 2.94 vs 0.79 [AB]. Each question is hard-capped at 6 speculative web calls
plus 1 final (`server/config.py`), and the most any question used was 7 [AB rows]. The tagger and ASR are
on-device, so speculation costs no extra GPU. Searches cancelled as stale stop early, and their results still go into
the reranking pool. Drafts are short (up to 160 output tokens, `server/answer.py`) and are cancelled as soon as the
meaning moves on. In exchange, the median answer arrives 435 ms sooner [AB]. Search calls are the cheapest part
of a voice pipeline. Dead air is what costs you users.

**2. "What if the ASR is wrong?"**
That case is the design, not an exception. Arc revises its own words, and every revision either keeps the search or
marks it stale. On q01 the ASR first heard "founded condo", then "founded company acronis". Both searches were
cancelled, and "founded company nvidia" fed the answer [AB, q01 `spec_queries`]. The answer prompt also tells the
answer model the text is ASR output. When the *final* transcript is wrong, both modes fail the same way: q33
("no wait" heard as "no weight") is wrong in both, and so is conv03 ("Coron pride and prejudice") [AB, CONV].
Head Start does not hide ASR errors. It just stops paying latency for them.

**3. "How is this different from a streaming answer model or voice-to-voice?"**
Those stream the *output*. We start the *input side* early: retrieval and grounding. A voice-to-voice model
still has to wait for the turn to end before it can search, or it answers from memory without sources. Head Start
answers with citations, and its sources are ready at 568 ms vs 926 ms at p50 [AB]. It sits *in front of* any answer
model or voice stack, so it adds to them rather than replacing them.

**(Bonus) "What's defensible?"** The speculator policy (when to fire, cancel, draft, reuse and endpoint), tuned on
real audio with a measured A/B harness. A 10M tagger that gives intent, slots, corrections and completeness at
~1 ms per word on-device, which an answer-model router cannot do in time (README: about 500 ms). And a spoken-audio
retrieval lane over our own moments index.

---

## Honest limitations (say these before the judges do)
- **The web lane ran on Wikipedia, not Tavily.** `TAVILY_API_KEY` is not set yet, so the web lane falls back to
  keyless Wikipedia (`server/lanes.py`, `config.web_backend: "wikipedia"` in [AB]). Adding the key switches the lane
  with no code change. Every number above uses Wikipedia.
- **p90 barely moves:** 1831 vs 1866 ms [AB]. Head Start was *slower* on 11 of 48 questions, mostly speculative
  misses, including 5 of the 10 local-moments questions. On local questions the p50 is 1440 vs 1322 ms. Tail latency
  is dominated by answer-model time-to-first-token outliers (q26: 2364 vs 1573 ms).
- **Accuracy is keyword-scored** (`eval/run_ab.py` `score`, `eval/run_conv.py` `ok`): an answer counts as correct if
  it contains the expected words. conv02 counts as correct even though the first-turn answer says "the sources
  don't cover the Mona Lisa".
- **Small, synthetic eval:** 48 questions spoken by TTS voices (Kokoro and macOS `say`), plus 6 two-turn
  conversations. The ASR heard 36 of 48 questions differently from the script (counting case and punctuation)
  [AB rows]. That is a realistic stress test, not a benchmark.
- **Memory follow-ups:** by keyword scoring, Head Start got 3/5 scored follow-ups right and the baseline 2/5
  [CONV]. PROGRESS.md says "4/5", which does not match `runs/conv.json`, so quote 3/5.
- **Not yet tested by a human:** the browser mic path (AudioWorklet). Race-mode replays and the server-side mic
  path are verified [PROG].
- **Licences:** Arc and Arc Tagger are Whissle's on-device models under CC BY-NC 4.0 (research licence, not yet for
  commercial use). EmbeddingGemma 2 is Google's, under Apache 2.0. The local moments index uses third-party
  YouTube audio, which needs rights clearance before any public showing [IDX].
- **One machine:** all timings are from a single M3 Pro laptop, with modes run in alternating order per question.
