# Head Start: 3-minute demo script

Setup: `.venv/bin/python -m uvicorn server.app:app --port 8770`, open http://localhost:8770 at 1280×720, mode **Race**.
Warm-up takes ~40 s; check the header shows ready. Have the replay list ready in case the room mic is bad.

1. **Problem (20 s).** "Every voice assistant waits for you to stop, then searches, then thinks: two seconds of dead
   air. Search engines got instant; voice didn't."
2. **Live race (60 s).** Press Space and ask: *"Who founded the company Nvidia?"* (or replay q01).
   Point at: words arriving; the **Tagger** stage lighting up with intent `qa_factoid` at word 3; web searches firing
   on the timeline *while you are still talking*; the stale "acronis" search (ASR mishearing) cancelled the moment
   the word was revised; the Head Start pane answering first and the delta in big type.
3. **Self-correction (40 s).** *"What's the capital of Canada, actually make that Turkey."* (q27) The Canada search
   goes stale, Turkey fires, the answer is still first. "It's not just faster; it understands corrections mid-sentence."
4. **Moments search (30 s).** Switch to Local moments, replay local_05: *"In the hearing, where was the gun found?"*
   The audio lane fires on the raw sound of the question (EmbeddingGemma 2 puts speech and text in one space) and
   the answer cites the moment with its timestamp.
5. **Evidence (30 s).** Results panel: real-audio A/B over 48 spoken questions (13 voices, fillers, corrections,
   local moments): p50/p90 time-to-answer, accuracy parity, web calls per question. "Same ASR, same search, same
   reranker, same answer model; the only difference is when we start."

Lines for judges: "10-million-parameter tagger in C, ~1 ms per word, on the laptop: routing for free."
"Speculation is budgeted (≤6 web calls per question) and every miss is shown on the timeline."
Backup: if Wi-Fi fails, the local-moments questions still work offline except for the answer model.
