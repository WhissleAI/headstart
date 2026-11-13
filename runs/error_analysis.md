# Error analysis: wrong and ungrounded answers (A/B runs, 2026-11-12)

Scope: every row that was **wrong** (keyword check failed) or **ungrounded** (the answer's first sentence says the
sources lack the answer: `speculator._non_answer`) in `ab_full1`, `ab_full2`, `ab_full3`, `ab_snr20`, `ab_snr20_post`,
`ab_snr10`, `ab_rate_fast`, `ab_rate_slow`, both modes. 225 rows, grouped below by (question, text the ASR heard): 68 groups.
Wrong and ungrounded rows are counted separately (`W` / `U`).

Excluded: `ab_rate08` / `ab_rate125` (38/40 rows bad in both modes). Their audio and question ids are misaligned
(q01 "who founded the company Nvidia" was heard as "tell me what is the calendar in new york", q04 "who painted the Mona
Lisa" as "repeat the alarm at least seven"), so this is a harness problem, not the system's. `ab_rate_fast` / `ab_rate_slow`
are the clean re-runs.

`conv.json` (2-turn memory, not in the table): conv03 is ASR ("Who wrote" → "Coron", HS turn 1; then "she" carries
"coron"). conv05 baseline turn 2 answers "Kingdom of Württemberg" without Ulm (answer/retrieval granularity). conv06
turn 2 (both modes) hedges on "playstation headquarters": retrieval, and the key itself is debatable (`tokyo`; Sony
Interactive Entertainment is headquartered in San Mateo). conv04 turn 2 is unscored, but "there" resolved to the
previous question's entity (Australia) instead of the place named in the previous answer (Canberra): a memory
resolution gap, not addressed here.

## Root causes (primary cause per group)

| cause | groups | rows | what it looks like | fix |
|---|---|---|---|---|
| **ASR** (arc_heard wrong) | 42 | 107 | mostly snr10/snr20: "Nvidia" → "and video", "Jaws" → "jewels", "penicillin" → "Kennesaw" | not fixable downstream when the key word is gone. Partly recovered: entity query on what survives, relaxed retry, and an answer prompt that answers the meant question instead of listing unrelated source facts |
| **query construction** | 9 | 48 | fillers/contractions leaked (`i've wanted many`, `mm`, `didn't`); correction cue mis-tagged as an entity ("Sari," = place, "Scratch" = organization), so the wrong span was "corrected away"; untagged replacement made the fallback drop the verb ("born") and keep Mozart; cue heard as "no way" | **fixed** (frame.py) |
| **answer model** | 10 | 30 | the right fact is in the sources (or follows from them: Congo "second only to the Nile") but the answer opens with "the sources don't…", or comments on the corrected-away entity ("The sources don't cover Microsoft, but…") | **fixed** (corrected question to the model + grounding rules in the prompt) |
| **retrieval** | 3 | 21 | right query, wrong articles: "capital turkey" → Capital punishment in Turkey; "headquarters toyota" → only US subsidiaries; Africa → Congo/Niger, no Nile | **fixed**: entity-only query in parallel (Turkey, Toyota, Beethoven, London, Pride and Prejudice articles) |
| **ranking** | 0 | 0 | none clearly seen. Pools aren't logged (fires keep their top 3 only), so a reranker loss would show up here as retrieval | — |
| **scoring artefact** | 4 | 19 | answer right, keyword check too strict: "nailed" vs `nails`, "fifteen seconds" vs `15`, "prepping" vs `prep`, `winner` AND `won` both required | not changed (eval/run_ab.py has uncommitted edits on main by the robustness agent). Suggested: stem match (`nail(s|ed|ing)`), number words ↔ digits, `winner|won` as alternatives |

Local-lane groups (`local*`) appear in the table but weren't replayed: they need the local index plus the reranker. Their
hedges ("The sources don't name anyone called Bajaj") are the same answer-model pattern, so the prompt change applies.

## Per-group table

`replay` = offline before → after on the same heard text (eval/replay_answers.py, see below). "hedged" = ungrounded.

| id | heard (answered utterance) | recorded query | rows | cause | detail | replay before → after |
|---|---|---|---|---|---|---|
| q21 | I've always wanted to know how many bones are there in the | `i've wanted many bones adult human body` | 7W 3U (full1, full2, full3, rate_fast, rate_slow) | **query** + retrieval | fillers i've/wanted/many leaked; Wikipedia ANDs all words → Embalming, Jennifer's Body | wrong (hedged) → **right** |
| q27 | What's the capital of Canada actually make that turkey | `capital turkey` | 0W 15U (full1, full2, full3, rate_fast, rate_slow, snr10, snr20, snr20_post) | **retrieval** + answer | 'capital turkey' → Capital punishment in Turkey…; no Turkey article; answer hedged ('sources don't name…') while inferring Ankara | right (hedged) → **right** |
| q29 | Who founded Microsoft? Sorry. I met Amazon | `founded met amazon` | 0W 2U (full1) | **answer** + query | answer opens 'The sources don't cover Microsoft' (the corrected-away entity) → flagged ungrounded; full1 also leaked cue word 'met' | right (hedged) → **right** |
| q30 | Where was Mozart born actually make that Beethoven | `mozart beethoven` | 8W 2U (full1, full2, full3, snr20, snr20_post) | **query** | correction fallback (Beethoven untagged) dropped 'born' (word before cue) and KEPT Mozart → 'mozart beethoven' | wrong (hedged) → **right** |
| q31 | What river runs through Paris, Sari, I mean London | `river runs through paris london` | 0W 7U (full1, full2, full3, rate_fast, rate_slow) | **query** + retrieval | tagger tags cue 'Sari,' as a place → it was 'corrected away', Paris kept; no Thames/London article; answer hedged | right (hedged) → **right** |
| q33 | Who was the first president of France, no weight of the Un | `first president france weight united states` | 4W 4U (full1, full2, full3, rate_slow) | **query** + answer | full1/2: cue 'no weight' unknown (fixed on main before full3); full3: answer hedged about France | right (hedged) → **right** |
| q41 | What is the longest river in Africa | `longest river africa` | 0W 6U (full1, full2, snr10, snr20_post) | **answer** + retrieval | Nile article not in top 4; Congo article says 'second only to the Nile' but answer disclaimed 'sources don't state that directly' | right → **right** |
| q42 | Where is the headquarters of Toyota | `headquarters toyota` | 4W 0U (full1, full2) | **retrieval** + answer | only subsidiaries (Toyota Motor North America, Sales USA) + Toyota, Aichi; answer picked Plano, Texas | wrong → **right** |
| local01 | When did the announcer call Hunter's three in the game | `announcer call hunter's three game` | 3W 0U (full1, full2, rate_slow) | **scoring** | answer says 'nailed', expect 'nails' | not replayed (local lane) |
| local02 | What did Bajaj say about the objection in the hearing | `bajaj say objection hearing` | 0W 10U (full1, full2, full3, snr20, snr20_post) | **asr** + answer | 'the judge' → 'Bajaj'; right but hedged about 'Bajaj' | not replayed (local lane) |
| local04 | What did the commentator say about Lonzo Ball in the game | `commentator say lonzo ball game` | 4W 6U (full1, full2, full3, snr20, snr20_post) | **answer** + scoring | local transcript garbled ('Ponto ball'); hedges; 'three-pointer' vs expect 'triple' | not replayed (local lane) |
| local06 | what did the judge ask about the office's wedding fifteen  | `judge office's wedding fifteen seconds` | 5W 0U (full1, full2, full3) | **asr** + scoring | 'officers waiting' → 'office's wedding'; HS answer correct but says 'fifteen', expect '15' | not replayed (local lane) |
| local08 | What did the commentator say about who won the debate | `commentator say won debate` | 10W 0U (full1, full2, full3, snr20, snr20_post) | **scoring** | answers say 'winner' and 'lost'/'won' — expect needs BOTH 'winner' and 'won' | not replayed (local lane) |
| local09 | In the recording, what did Chris Christie say about Debate | `recording chris christie say debate trump` | 5W 0U (full1, full2, full3) | **scoring** + asr | answers say 'prepping … wrestling match … boxing match'; expect 'prep' (whole word) fails | not replayed (local lane) |
| q29 | Who founded Microsoft Sorry I met Amazon | `founded amazon` | 0W 2U (full3) | **answer** + query | answer opens 'The sources don't cover Microsoft' (the corrected-away entity) → flagged ungrounded; full1 also leaked cue word 'met' | right → **right** |
| local10 | In the clip, when did they say Jean U.S.A one bronze | `clip jean usa one bronze` | 0W 1U (full3) | **asr** + answer | 'Team USA won' → 'Jean U.S.A. one'; right but hedged | not replayed (local lane) |
| q01 | who founded the company and video | `founded company video` | 4W 0U (snr20, snr20_post) | **asr** | 'Nvidia' → 'and video' (snr20) | wrong → **wrong** |
| q04 | repainted the Mona Lysa | `repainted mona lysa` | 1W 1U (snr20, snr20_post) | **asr** + retrieval | 'who painted' → 'repainted', 'Lisa' → 'Lysa': zero hits; now relaxed retry; answer still right only from model knowledge | right → **right (hedged)** |
| q10 | To erode the novel 100 years of solitude | `erode novel 100 years solitude` | 1W 3U (snr20, snr20_post) | **asr** + retrieval | 'who wrote' → 'to erode'; AND query finds nothing relevant; now correct from model knowledge (ungrounded, says so) | wrong (hedged) → **right (hedged)** |
| q11 | What is the pinnacle single for gold | `pinnacle single gold` | 4W 0U (snr20, snr20_post) | **asr** | 'chemical symbol' → 'pinnacle single' | wrong (hedged) → **wrong (hedged)** |
| q18 | Mm, can you tell me who invented the telephone | `mm invented telephone` | 4W 0U (snr20, snr20_post) | **query** | filler 'mm' leaked → 'mm invented telephone' → Phone connector, Théâtrophone | wrong (hedged) → **right** |
| q20 | Hey, okay So who was the first person to walk on the moon | `first person walk moon` | 0W 2U (snr20, snr20_post) | **answer** + retrieval | 'List of people who have flown to the Moon' etc.; answer hedged 'don't name the first … directly' | wrong (hedged) → **right** |
| q21 | Ivan was wanted to know how many phones are there in the a | `ivan wanted many phones adult human body` | 3W 1U (snr20, snr20_post) | **asr** + query | 'I've always' → 'Ivan was', 'bones' → 'phones' | wrong (hedged) → **right (hedged)** |
| q24 | Um, I forget who directed the movie jewels | `directed movie jewels` | 4W 0U (snr20, snr20_post) | **asr** | 'Jaws' → 'jewels' | wrong → **wrong** |
| q26 | Ned, who wrote Hamlet, sorry, I mean pride and prejudice | `ned wrote pride prejudice` | 4W 0U (snr20, snr20_post) | **asr** + retrieval | leading 'Ned,' + correction; 'ned wrote pride prejudice' finds composers; entity query 'pride and prejudice' (post-cue phrase) finds the novel | right (hedged) → **right** |
| q28 | When was Google founded No, wait I mean, Apple Five | `founded apple five` | 2W 0U (snr20, snr20_post) | **asr** + retrieval | trailing ASR word 'Five' → Apple Corps / cider vinegar; now answers 1976 (from knowledge) | wrong (hedged) → **right** |
| q31 | What river runs through Paris, Sari, I mean, London | `river runs through paris london` | 0W 1U (snr20) | **query** + retrieval | tagger tags cue 'Sari,' as a place → it was 'corrected away', Paris kept; no Thames/London article; answer hedged | right (hedged) → **right** |
| q33 | Who was the first president of France no way of United Sta | `first president france way united states` | 4W 0U (snr20, snr20_post) | **query** + answer | cue heard as 'no way' (not in the cue list) → France kept, model answered Louis-Napoléon | right (hedged) → **right** |
| local01 | Wendy the announcer called Hunter's three in the game | `wendy announcer called hunter's three game` | 4W 0U (snr20, snr20_post) | **asr** + scoring | 'When did' → 'Wendy'; answer still says 'nailed' | not replayed (local lane) |
| local07 | In the meeting What did the teacher say, Sam You will want | `meeting teacher sam` | 0W 2U (snr20, snr20_post) | **asr** + answer | right but hedged | not replayed (local lane) |
| local09 | In the recording, what did Chris Christie say about Dave P | `recording chris christie dave prithwick trump` | 4W 0U (snr20, snr20_post) | **asr** + scoring | 'debate prep' → 'Dave Prithwick' | not replayed (local lane) |
| q08 | Who was the first woman to win a Nobel Prize | `first woman win nobel prize` | 0W 2U (snr10, snr20_post) | **answer** | List of women Nobel laureates in sources; answer: 'sources don't name the first woman' then Curie | right → **right** |
| q04 | complain to the Mona Lisa | `complain mona lisa` | 2W 0U (snr10) | **asr** | 'who painted' → 'complain to' | wrong (hedged) → **wrong** |
| q05 | What's how tall is Mount Deborah's | `tall mount deborah's` | 2W 0U (snr10) | **asr** | 'Everest' → 'Deborah's' | wrong (hedged) → **wrong (hedged)** |
| q06 | I would have discovered Kennesaw | `have discovered kennesaw` | 2W 0U (snr10) | **asr** | 'penicillin' → 'Kennesaw' | wrong → **wrong** |
| q07 | what's the new yoga for from rental hall | `new yoga rental hall` | 2W 0U (snr10) | **asr** | whole question garbled | wrong → **wrong** |
| q11 | What is the chemical symbol to go of | `chemical symbol go` | 0W 1U (snr10) | **asr** + answer | 'gold' → 'go of'; answer right but hedged; now plain | wrong (hedged) → **right** |
| q12 | Python, the programming language you created | `python programming language created` | 1W 0U (snr10) | **asr** + answer | 'who created it' → 'you created'; sources lacked 'Python (programming language)', model credited PSF (replay: right) | right → **right** |
| q13 | the great wallet chime, which Dynasty built most of it now | `great wallet chime dynasty built most` | 0W 2U (snr10) | **asr** + answer | 'Great Wall of China' → 'great wallet chime'; right but hedged; now plain | right (hedged) → **right** |
| q16 | I think she'll be a ruttier with you, Barnard | `she'll ruttier barnard` | 2W 0U (snr10) | **asr** | whole question garbled | wrong → **wrong (hedged)** |
| q18 | Mm-hmm What do you mean I don't know I don't know I don't  | `mm-hmm mean don't` | 2W 0U (snr10) | **asr** | whole question garbled (snr10) | wrong (hedged) → **wrong (hedged)** |
| q20 | What a game Okay So what's the first person to walk on the | `game first person walk moon` | 1W 1U (snr10) | **asr** + answer | noise word 'game' leaked into query; answer hedged | right (hedged) → **right** |
| q21 | Ivan was wanting to know how many phones are there in the  | `ivan wanting many phones adult human body` | 2W 0U (snr10) | **asr** | as q21 snr20 | wrong (hedged) → **wrong (hedged)** |
| q24 | Uh, I forget who's erected the movie jewels | `erected movie jewels` | 2W 0U (snr10) | **asr** | 'directed … Jaws' → 'erected … jewels' | wrong → **wrong** |
| q26 | Netly wrote Hamlet I'm sorry I mean pride and prejudice | `netly wrote hamlet pride prejudice` | 1W 1U (snr10) | **asr** + retrieval | entity query 'pride and prejudice' after the cue | wrong → **right** |
| q28 | When was Google founded No way I mean, Apple Michael | `founded way apple michael` | 0W 2U (snr10) | **asr** + answer | 'no way' cue + trailing 'Michael'; right but hedged about Google | right (hedged) → **right** |
| q30 | Where was Mozart born Actually, make that date happen | `mozart date happen` | 2W 0U (snr10) | **asr** | 'Beethoven' → 'date happen' | wrong (hedged) → **right** |
| q31 | What river runs on Paris, Sari, I mean, London | `river runs paris london` | 2W 0U (snr10) | **query** + retrieval | as q31 (cue mis-tag) | right (hedged) → **right** |
| q32 | Look current he does Japan use Scratch that switzerland | `current japan use switzerland` | 2W 0U (snr10) | **asr** + query | 'what currency' → 'look current he'; 'Scratch' tagged as organization (now ignored) | wrong → **wrong** |
| q33 | who was the first president of France, no late of United S | `first president france late united states` | 1W 1U (snr10) | **asr** + answer | cue 'no wait' → 'no late' (unrecognised); entity query + prompt now answer Washington | right (hedged) → **right** |
| q42 | Where is the headquarter of Toyota | `headquarter toyota` | 2W 0U (snr10) | **retrieval** | 'headquarter toyota' → museums; entity query 'toyota' finds the company | wrong (hedged) → **right** |
| local01 | England did the announcer call 103 in the game | `england announcer call 103 game` | 2W 0U (snr10) | **asr** | garbled → cricket moments | not replayed (local lane) |
| local02 | What did Vajaj say about the objection in the hero | `vajaj objection hero` | 0W 2U (snr10) | **asr** + answer | same | not replayed (local lane) |
| local04 | We'd like to have a commentator say about Lonzo Ball in th | `we'd have commentator lonzo ball game` | 2W 0U (snr10) | **asr** + ranking | gold moment not in top 5 | not replayed (local lane) |
| local05 | in the hearing Where was the gunpowder | `hearing gunpowder` | 1W 1U (snr10) | **asr** + answer | 'gun' → 'gunpowder' | not replayed (local lane) |
| local06 | What did Rajasqy that the office is waiting 15 seconds | `rajasqy office waiting 15 seconds` | 1W 1U (snr10) | **asr** | 'the judge' → 'Rajasqy' → routed to web | not replayed (local lane) |
| local07 | In the meeting, what did the teacher say Samuel one | `meeting teacher samuel one` | 0W 1U (snr10) | **asr** + answer | right but hedged | not replayed (local lane) |
| local08 | What did Norman Baker say about who won the debate | `norman baker won debate` | 2W 0U (snr10) | **asr** | 'the commentator' → 'Norman Baker' → web route | not replayed (local lane) |
| local09 | In the recruiting, what did Christie say about David Brett | `recruiting christie david brett trump` | 2W 0U (snr10) | **asr** | 'recording' → 'recruiting' → local cue lost → web | not replayed (local lane) |
| local10 | in the club When did they say Team USA one problem | `club team usa one problem` | 1W 1U (snr10) | **asr** | 'clip' → 'club' → local cue lost | not replayed (local lane) |
| q08 | Who's the first woman to win a Nobel Prize | `first woman win nobel prize` | 0W 1U (rate_fast) | **answer** | as q08 | right → **right** |
| q08 | who is the first woman to win a Nobel Prize | `first woman win nobel prize` | 0W 1U (rate_fast) | **answer** | as q08 | right → **right** |
| q12 | I found the programming language who created it | `found programming language created` | 2W 0U (rate_fast) | **asr** | 'Python' → 'I found' | wrong (hedged) → **wrong (hedged)** |
| q29 | Who founded Microsoft Sorry, I'm Ed Amazon | `founded i'm ed amazon` | 2W 0U (rate_fast) | **answer** + query | answer opens 'The sources don't cover Microsoft' (the corrected-away entity) → flagged ungrounded; full1 also leaked cue word 'met' | wrong (hedged) → **right (hedged)** |
| local01 | When did the announcer call hunters three in the game | `announcer call hunters three game` | 1W 0U (rate_fast) | **scoring** | 'nailed' vs 'nails' | not replayed (local lane) |
| q08 | Who was the first woman to win a no-belt prize | `first woman win no-belt prize` | 2W 0U (rate_slow) | **asr** | 'Nobel' → 'no-belt' | right → **right** |
| q25 | And I didn't know, why is the sky blue | `didn't sky blue` | 0W 2U (rate_slow) | **query** | utterance split; contraction "didn't" leaked → 'didn't sky blue' → songs | wrong (hedged) → **right** |
| q29 | Who founded Microsoft Sorry, I met Amazon | `founded amazon` | 0W 2U (rate_slow) | **answer** + query | answer opens 'The sources don't cover Microsoft' (the corrected-away entity) → flagged ungrounded; full1 also leaked cue word 'met' | wrong (hedged) → **right (hedged)** |

## Fixes (branch `answer-quality`)

1. **Query construction** (`server/frame.py`)
   - Stopwords: spoken fillers (`mm`, `hmm`, `mhm`, `uh-huh`, `er`, `erm`, `ah`), contractions with and without
     the apostrophe (`i've`/`ive`, `didn't`/`didnt`, …), and `wanted wanting many much have has had`.
   - Correction cue mis-tagged as an entity (`Sari,` → place, `Scratch` → organization): spans that overlap a cue are
     ignored, so the real entity before the cue gets replaced.
   - Correction with an untagged replacement ("Mozart born, actually make that Beethoven"): the last entity before
     the cue is dropped, not the last content word. Word-for-word corrections without entities ("Lakers game, sorry,
     Warriors game") work as before.
   - New cue `no way` (how the ASR heard "no wait" under noise).
   - `Frame.clean_text` / `resolved_text`: the question minus the cue and the corrected-away entity ("Who founded
     Amazon"). It goes to the answer model and the reranker, so the answer no longer opens with "The sources don't
     cover Microsoft".
   - `Frame.entity_query()`: the entity phrase as spoken, keeping its inner stopwords ("pride and prejudice", "united
     states"), extended over the proper-noun run around the tagger's span ("Mount" → "mount kilimanjaro", "Apollo" →
     "apollo 11"). After a correction with no tagged replacement, it is the phrase after the cue. The carried entity
     ("when was it founded") counts too. Empty when there's no entity or the phrase equals the query.
2. **Retrieval** (`server/lanes.py`, `server/speculator.py`)
   - `wikipedia(query, entity=…)`: the entity query (top 2) runs in parallel with the content query (top 4), and its
     articles are appended without duplicates. The reranker picks the top 5 from both. Only on the Wikipedia backend;
     Tavily takes natural language and ignores it.
   - Relaxed retry: a content query with zero hits (and no entity hits) is retried once without its first word
     ("repainted mona lysa" → "mona lysa"). Wikipedia ANDs every term, so one leading misheard word or filler sinks it.
   - Checked and **not** used: `srenablerewrites` / `srinfo=suggestion` (verified in the API's paraminfo). For
     "repainted mona lysa" the suggestion was "retained mona lisa": it fixes "lysa" and breaks "repainted". The
     rewritten search returned ship and film articles. Switching `classic_noboostlinks` → `engine_autoselect` changed
     little on the failing queries (Turkey enters at #4, Toyota at #4; still no Nile).
3. **Answer prompt** (`server/answer.py`): if a source states the answer, or the answer follows directly from it,
   state it with the citation and no disclaimer. Never remark on parts of the question the sources don't cover
   (corrected-away name, misheard word). Only when no source supports the answer: say so in a few words, then give the
   confident answer without a citation, and don't pad with unrelated source facts. Such answers are still flagged
   ungrounded by `_non_answer`, so grounding stays honest. The prompt's examples are generic, not taken from the eval
   set.
4. Tests: `tests/test_answer_quality.py` adds 15 tests: fillers, cue mis-tag, untagged replacement, corrected question,
   entity query and carry, four corrections as actually heard (native tagger), and the Wikipedia merge, dedup and
   relaxed retry against an `httpx.MockTransport`. The full suite has 81 tests, all passing (66 before).

## Offline before/after (eval/replay_answers.py → runs/replay_answers.json)

No audio. For each row the script takes the text the system actually answered (`question_heard`). It rebuilds the final
frame with the native Arc Tagger, fed word by word as live. Then: query → real Wikipedia → Claude Haiku 5.5 via
`server/answer.py`'s model and options → the same keyword scorer, plus `_non_answer`. "before" uses `frame.py`, the
SYSTEM prompt and the single-query Wikipedia lane at `af29048`; "after" uses this branch. Query fidelity: the replayed
"before" frame reproduces the recorded query on 66/68 items. The 2 misses are full1/full2 rows recorded before the
`met` / `no weight` cues existed.

| set | n | right before → after | **right and grounded** before → after | hedged ("sources don't…") before → after |
|---|---|---|---|---|
| previously wrong or ungrounded (all non-local groups, unique heard text) | 48 | 19 → **33** | 9 → **29** | 30 → **11** |
| regression: right and grounded in both modes of full3 | 20 | 20 (live) → **20** | 20 → **20** | 0 → 0 |

Weighted by how often each group failed (153 failing web rows): after the change, 107 rows are right and grounded,
14 right but hedged, and 32 still wrong. Those 32 are ASR failures where the key word is gone ("and video", "jewels",
"Kennesaw", "pinnacle single", "date happen") plus q32, where "currency" was heard as "current he".

Where the gains come from (per-row output in the JSON):
- query: q21 bones, q18 telephone, q25 sky blue, q30 Beethoven, q31 London (×3), q33 "no way"
- entity query: q27 Turkey, q42 Toyota (×2), q26 Pride and Prejudice (×2), q30 Ludwig van Beethoven, q31 London, q33
- prompt / corrected question: q29 Amazon, q08 first woman Nobel (×4, already right and now grounded), q41 Nile (×2),
  q20, q13, q11, q28

Caveats:
- The reranker is skipped in both variants: docs go to the model in lane order, top 5. In "after" the entity
  article is usually #5, and a second entity article can be cut. Live, the reranker picks the top 5 of up to 6.
- One sample per item, so LLM noise applies to the ±1–2 items that flipped without a retrieval change.
- Regression "before" is the recorded live answer (right and grounded by construction), not a replay. That kept
  the answer-model budget.
- Spend: the measured run made 111 Wikipedia and 116 Haiku calls. A first attempt fired every search at once, got
  HTTP 429 from Wikipedia and aborted. Its calls weren't counted; it aborted during the Wikipedia phase, so few if
  any answer calls went out. The script now limits Wikipedia to 2 concurrent calls with backoff on 429, caches by
  query, and hard-caps both budgets.
- Latency (not the goal here, and not measured live: no audio runs were allowed): the entity query is a second
  Wikipedia call in parallel, made only when the frame has an entity different from the query. The lane waits for
  the slower of the two. Wikipedia p50 is about 190 ms, so expect a small tail cost. Check it on the next real A/B.

## Not fixed / next
- Scorer strictness (19 rows): stems, number words, `winner|won`. This needs `eval/run_ab.py`, which has uncommitted
  edits on main.
- ASR-dominated failures are most of what's left: Deepgram ear or n-best rescoring. Wikipedia's "did you mean"
  doesn't help (see above).
- Conversation memory: "there" should prefer a place named in the previous answer (conv04).
- Leaked reasoning seen once in the replay (q26 snr10: "The user likely meant … no, more plausibly …"). Rare, but
  `_non_answer` doesn't catch it.
