# Head Start: voice search that starts before you finish talking

Every voice search today waits for silence, then searches, then thinks. Head Start understands the question **while it is
being spoken**, searches from about the third word, drafts the answer, and has it ready when the speaker stops.

```
mic ─▶ Arc (streaming ASR + events) ─▶ Arc Tagger (10M, 0.7 ms/word) ─▶ query frame ─▶ Speculator ─▶ lanes ─▶ answer
          words, revisions,              intent, slots, corrections,     (intent, slots,   fire / cancel /   web (Tavily|Wikipedia)
          command_complete               command_complete                 route)            reuse / draft     local moments (EmbeddingGemma 2 text)
                                                                                                              audio (EmbeddingGemma 2: the spoken
                                                                                                                     question's own audio)
```

| piece | role | why it matters |
|---|---|---|
| **Arc v1** (Whissle, on-device) | live words every ~100 ms, self-revising; its own `command_complete` | words + revisions while the person is still talking |
| **Arc Tagger v0.3** (Whissle, 10M, native C int8) | partial sentence → intent, typed slots, dialogue act, `command_complete` | understands the query in ~1 ms/word instead of an LLM router's ~500 ms; updates as words revise |
| **Speculator** (`server/speculator.py`) | when to search, what to search, cancel stale, draft early, verify reuse | the search intelligence |
| **EmbeddingGemma 2** (Google, 740M, text+audio+image) | reranks the pool; local moments index; embeds the *spoken audio* of the question | one space for audio and text: retrieval can start before the words settle |
| **Claude Haiku 5.5** | streaming 1–2 sentence answer with citations | thinking off, effort low: ~600 ms to first token |

## The speculator in one screen
- **Query frame** = (fine intent, typed slots, content words) from the tagger. Its signature changes only when the meaning
  changes, so filler words and repeated partials don't re-search.
- **Fire** when ≥3 words and (≥2 content words or a typed slot) and the frame held for 2 updates (1 if it has a typed
  slot). Budget 4 web calls per question; local lanes are free.
- **Self-corrections** ("Lakers… sorry I mean the Warriors", "Paris, actually make that London") replace the corrected
  entity before querying.
- **Stale**: a search whose words the speaker revised away is cancelled in flight; its results stay in the pool (the
  reranker decides) but don't feed drafts while fresh results exist. Revived if the ASR flips back.
- **Drafts**: as soon as every lane of the current meaning has returned, Haiku drafts the answer in the background.
  At the endpoint a draft is reused if its meaning matches exactly, or if the final question only *added* words that
  the draft's own answer already contains ("how tall is the Eiffel" → "…Eiffel Tower"). Otherwise: answer from the
  in-flight/landed search for the final query, or (only if nothing new was said) commit from the pool.
- **Semantic endpointing**: answer when the tagger says the request is complete, the speaker has been silent 250 ms
  and the words held through one ASR refresh, instead of waiting for the ASR's own pause endpoint (~470 ms). If the
  speaker keeps going, the longer question is answered again.

## Run
```sh
./run_demo.sh                                               # demo: prereqs, start, warm, preflight GO/NO-GO, open
.venv/bin/python -m uvicorn server.app:app --port 8770     # warm-up ~30-60 s; open http://localhost:8770
.venv/bin/python -m pytest -q tests                         # deterministic tests (fake lanes, LLM, Deepgram, Tavily)
.venv/bin/python eval/run_ab.py                             # real-audio A/B → runs/ab_latest.{json,png}
```
Keys in `.env` (git-ignored): `ANTHROPIC_API_KEY`; `TAVILY_API_KEY` switches the web lane from Wikipedia to Tavily.
Ear: `EAR=deepgram` + `DEEPGRAM_API_KEY` streams the same audio to Deepgram nova-3 (interim results, 300 ms endpointing)
for the words; without a key, or if the socket fails/drops, the run falls back to Arc (the open utterance is re-fed).
`/api/health` and the `ready` event report `ear` (active) and `ear_requested`.
`EAR=flux` streams to Deepgram **Flux** (`flux-general-en`, `wss://api.deepgram.com/v2/listen`) instead: Flux's turn model
gives the endpoints. `EagerEndOfTurn` → Head Start's soft answer at once (`FLUX_SOFT=eager`; `eager+gate` also requires
our tagger gate; `off` ignores it), `TurnResumed` → the longer question is answered again, `EndOfTurn` → the ear's final
(confirms the soft answer). Lowest-latency defaults: `FLUX_EAGER_EOT=0.3` (docs minimum), `FLUX_EOT=0.7`,
`FLUX_EOT_TIMEOUT_MS=1000`; same Arc fallback as nova-3. Tuning: `DG_*` in `server/config.py`.
Tavily lane knobs (`server/config.py`): `TAVILY_SPEC_DEPTH`/`TAVILY_FINAL_DEPTH` (fast), `TAVILY_SESSION_CREDITS`,
`TAVILY_CACHE_SCOPE`, `TAVILY_ANSWER`. Measure a new key with `eval/tavily_probe.py --yes`; run keyless against the fake
with `uvicorn --app-dir tests fake_tavily:app --port 8799` + `TAVILY_API_KEY=tvly-fake TAVILY_BASE_URL=http://127.0.0.1:8799`.
Rebuild the native tagger: see `vendor/arctag` (`cc -O3 -shared ... -framework Accelerate`, details in PROGRESS.md).

Degraded mode (DEMO_CHECKLIST.md): if the live web lane fails, it serves `runs/web_cache.json` (recorded Wikipedia
responses for the eval set; `eval/record_web_cache.py`; `WEB_BACKEND=cache` forces it); if the answer model is
unreachable, answers are the best-matching sentence of the top source. Both modes share the fallbacks; both are
labelled in `/api/health`, the `ready` event, the metric (`web_replayed`, `answer_fallback`) and the UI.

## Honest notes
- Latency is measured from the end of the last spoken word (ASR word timestamps) to the first answer token, same ear,
  same lanes, same reranker, same model in both modes; modes run sequentially per question, alternating order.
- Arc Mini mishears some synthetic voices ("a polio eleven"); the answer prompt tells the model the text is ASR output.
- Arc, Arc Tagger: CC BY-NC 4.0 (research licence). EmbeddingGemma 2: Apache 2.0.
