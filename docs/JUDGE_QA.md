# Head Start: hard questions, grounded answers

Numbers come from `runs/ab_full2.json` [AB] (48 spoken questions, 13 voices, A/B), `runs/conv.json` /
`runs/conv_nomem.json` [CONV], `index/data/retrieval_eval.json` [IDX] and PROGRESS.md "Measured" [PROG].
Code references are in `server/`.

---

### 1. Is the speedup real, or did you tune the baseline to be slow?
The baseline shares everything with Head Start except timing. It uses the same Arc ear, the same lanes, the same
EmbeddingGemma 2 reranker, the same answer model and the same hedged answer request (`server/config.py`
`HEDGE_MS`, "both modes"). `mode="baseline"` in `server/speculator.py` waits for the endpoint, then searches,
reranks and answers. Each question runs in both modes one after the other, alternating which goes first (the
`order` field in [AB] rows). Latency is measured from the ASR timestamp of the last spoken word to the first answer
token. The result: p50 1546 → 1254 ms, paired median saving 435 ms, faster on 37/48 [AB]. The baseline's
endpoint lag (510 ms p50) is Arc's own pause rule, not an artificial delay.

### 2. p90 barely moved (1866 → 1831 ms). Why should I care about the median?
That's fair. The tail comes from two things Head Start does not control yet: answer-model time-to-first-token
outliers (q26 took 2364 ms in Head Start and 1573 ms in the baseline, on a speculative miss) and speculative misses
on local-moments questions (local10: 2050 vs 1313 ms). Head Start was slower on 11/48 questions [AB]. The fixes are
already in the design: hedged requests to the answer model and a 1.0 s cap on waiting for in-flight lanes
(`PENDING_CAP_S`). Of the 11 slower questions, 8 were speculative misses, 1 a pool commit, 1 a speculative hit and 1 a draft reuse [AB]. Draft reuse
happened on 52 % of questions, and raising that rate is the lever for the tail.

### 3. How much more does speculation cost per query?
Web calls go from 0.79 to 2.94 per question [AB]. The baseline is under 1 because local questions never hit the
web. Each question is hard-capped at `WEB_BUDGET=6` speculative web calls plus 1 final, and the most any question
used was 7 [AB]. Drafts average 1.69 per question (81 in total) [AB]. Each draft is capped at 160 output tokens
(`server/answer.py`) and is cancelled once the meaning moves on, so stale drafts stop early. The tagger and ASR run
on-device and cost nothing per call. Whether this is cost-effective depends on the search provider's per-call price,
which we have not measured yet. The trade is roughly 2 extra search calls for 435 ms off the median.

### 4. What happens when the ASR mishears?
Mishearing is the normal case: Arc's final text differed from the script on 36/48 questions (counting case and
punctuation) [AB]. Every ASR revision rebuilds the query frame. If a running search's content words are no longer in
the frame, `_mark_stale` cancels it (`server/speculator.py`). Its results stay in the pool for the reranker, but they
do not feed drafts while fresh results exist. If the ASR flips back, the search is revived. On q01 the ASR heard
"founded condo", then "founded company acronis", then "nvidia". Two searches went stale, and the third fed a draft
that was reused [AB]. The semantic endpoint also requires the words to hold for one ASR refresh (`SEM_STABLE_MS=420`)
before answering. When the *final* words are wrong, both modes fail together: in q33 "no wait" was heard as
"no weight", and both modes answered about France.

### 5. How do you handle "Canada, actually make that Turkey"?
`server/frame.py` has a list of correction cues ("I mean", "sorry", "no wait", "actually", "scratch that"…), plus
the versions the ASR actually produces ("Sari, I mean", "no weight", "I met"). The corrected entity replaces the
earlier one before the query is built. On q27 Head Start fired "capital canada", "capital make", "capital turnover"
and "capital turkey". Three went stale, and the answer came from a reused draft in 1092 ms vs 1557 ms [AB].
Across the 8 correction questions the p50 is 1142 vs 1648 ms at equal accuracy, 75 % each [AB].

### 6. Why a 10M tagger and not just ask the answer model to route?
Time. Arc Tagger runs in native C int8 at 0.7 ms per word, and 3 ms for a revision [PROG]. That is cheap enough to
re-run on every ASR partial, about every 100 ms. An answer-model router costs about 500 ms per call (README), which
is longer than the whole budget we save. The tagger also gives what the speculator needs as typed fields: intent,
slots, dialogue act, and `command_complete`. `command_complete` drives the semantic endpoint, which answered 29 of 48
questions before Arc's own pause endpoint [AB].

### 7. Isn't the semantic endpoint going to cut people off?
It answers early, but it does not commit. The soft endpoint fires after the tagger says the request is complete,
250 ms of silence have passed, and the words have held through one ASR refresh. The speculator stays warm, and if
the speaker keeps talking, the longer question is answered again (`server/session.py` `_soft_answer` / `on_final`).
The first version threw the speculator away on a pause. That run saved only 106 ms at the median, with 26/48 faster
(`runs/ab_full1.json`). Keeping it warm took the result to 435 ms and 37/48 [AB]. In the final run, 0 questions were
answered twice and 0 were split [AB `answered_twice`, `split_questions`].

### 8. What does EmbeddingGemma 2 actually add beyond a text embedder?
It puts speech and text in one space. That lets us embed the *audio of the spoken question* and search the local
moments index with no transcript. On 10 local questions, whole-question audio gets R@4 = 1.00 against the combined
text+audio index, compared with 0.90 for the text query [IDX]. The limits: R@1 is only 0.50, and the first 1.5 s of
audio is useless (R@1 0.10) because it is mostly the cue phrase. So the audio lane runs only on local-routed
questions with at least 1.8 s of speech, and it is a recall lane, not the answer. It also serves as the reranker for
web results, with one model for every lane.

### 9. Where is Tavily? The hackathon is co-hosted by them.
The web lane is written for Tavily (`server/lanes.py`: `tavily()` with `search_depth=fast`, topic routing for news
intents). It switches automatically when `TAVILY_API_KEY` is set. We have not received a key yet, so every number in
this repo uses the keyless Wikipedia fallback, which makes real network calls of about 200 ms each (q01 trace: 202,
211, 204 ms). That makes our numbers conservative for answer quality. Several misses are Wikipedia coverage, not
speculation: q42 Toyota returned only the North American headquarters, and the conv02 Mona Lisa search returned
"Pain of Salvation". With Tavily we expect better accuracy and similar latency. We will re-run the A/B the moment
the key arrives.

### 10. How do you know the answers are correct, not just fast?
They are keyword-scored: an answer counts as correct if it contains the expected terms, such as "jensen" and
"huang" (`eval/run_ab.py` `score`). On 48 questions, Head Start scored 85.4 % and the baseline 83.3 %. Paired, 39
were correct in both, 2 only in Head Start and 1 only in the baseline [AB]. Keyword scoring is lenient. conv02 counts
as correct although its first answer says the sources do not cover the Mona Lisa, and the claim is the model's own
knowledge [CONV]. The honest claim is accuracy *parity* at lower latency, not an accuracy gain. Follow-ups with
memory: Head Start got 3/5 scored follow-ups and the baseline 2/5 [CONV]. Without memory, the follow-up searches
were bare ("founded", "museum", "born") [conv_nomem].

---

### Extra: questions to expect from VCs
- **"Who buys this?"** Anyone who already ships voice search: voice assistants, in-car systems, contact-center
  agent assist, and live-media search (the moments lane already indexes sports, court and meeting audio). It sells
  as a latency layer in front of the customer's existing search API and answer model.
- **"Can you ship Arc commercially?"** Not under the current licence. Arc and Arc Tagger are CC BY-NC 4.0 research
  releases from Whissle, so commercial use needs a Whissle licence. EmbeddingGemma 2 is Apache 2.0.
- **"What's the moat if a big lab copies it?"** The policy (fire, cancel, draft, reuse, endpoint) is tuned on a real
  audio A/B harness, and it depends on an on-device understanding model fast enough to run on every partial. A big
  lab's models are built to answer *after* the turn ends. Running models at 1 ms per word is the moat.
- **"What's next?"** The Tavily key and a re-run of the A/B; a human test of the browser mic; bringing p90 down with
  a higher draft-reuse rate; and real-voice evaluation beyond TTS voices.
