"""Query frames: what the partial question means right now, built from Arc Tagger read-outs.

A frame is (intent, slots, content words). Its signature changes only when the meaning changes, so a partial that
re-sends the same words, or adds a filler word, doesn't trigger another search.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

STOP = set("""a an the is are was were be been being am do does did of in on at to for from by with about into over
what what's whats who who's whos whom whose which when where where's why how how's hows can could would should will
shall may might must i me my you your we our us it its it's this that these those there here please tell show give find
search look up hey ok okay so um uh like just and or but if then than also really very some any get got let let's
lets know want need yeah yes no oh well right now wondering wonder curious quick question asked ask
always ever forget forgot remember mind sure guess think kind sort actually basically literally anyone somebody
someone thing things they them their he she his her him it's say said says tell told
mm mmm hmm hm mhm mm-hmm uh-huh er erm ah huh i've i'd i'm i'll you've you'd you're we've didn't don't doesn't
isn't wasn't can't couldn't won't wouldn't haven't hasn't wanted wanting many much have has had
ive im didnt dont doesnt isnt wasnt cant couldnt wont wouldnt""".split())
# ^ last three lines: spoken fillers and contractions that leaked into real queries ("i've wanted many bones adult human
# body", "mm invented telephone", "didn't sky blue"); Wikipedia ANDs every word, so one leaked filler sinks the search
# hey/ok wake fillers are already in STOP; these mark a spoken self-correction
CORRECTION = [("i", "mean"), ("sorry",), ("no", "wait"), ("actually",), ("rather",), ("scratch", "that"),
              ("not", "that"), ("make", "that"), ("i", "meant"),
              # how live ASR actually hears them (eval set): "Sari, I mean", "no weight", "I met"
              ("sari",), ("no", "weight"), ("i", "met"), ("i", "mean", "it"), ("no", "way")]
INTENT_HINT = {
    "discourse.intent.weather_query": "weather forecast",
    "discourse.intent.news_query": "news",
    "discourse.intent.qa_stock": "stock price",
    "discourse.intent.qa_currency": "exchange rate",
    "discourse.intent.recommendation_locations": "best places",
    "discourse.intent.recommendation_events": "events",
    "discourse.intent.recommendation_movies": "movies",
    "discourse.intent.cooking_recipe": "recipe",
    "discourse.intent.transport_traffic": "traffic",
    "discourse.intent.qa_definition": "definition",
}
WEB_INTENTS = {"discourse.intent.weather_query", "discourse.intent.news_query", "discourse.intent.qa_stock",
               "discourse.intent.qa_currency", "discourse.intent.qa_factoid", "discourse.intent.qa_definition",
               "discourse.intent.recommendation_locations", "discourse.intent.recommendation_events",
               "discourse.intent.recommendation_movies", "discourse.intent.cooking_recipe",
               "discourse.intent.cooking_query", "discourse.intent.transport_query",
               "discourse.intent.transport_traffic", "intent.request_information"}
# Phrases that point at the local corpus (the recorded moments) instead of the web.
LOCAL_CUES = ["in the game", "the announcer", "the commentator", "in the clip", "in the recording", "in the video",
              "in the meeting", "in the call", "in the hearing", "the judge", "did he say", "did she say",
              "did they say", "was said", "the moment", "in the replay", "on the call", "the justice",
              "the panel", "the teacher", "in the commentary", "on the podcast", "in the broadcast", "the lawyer",
              "the attorney", "the witness", "in the debate", "the commentators", "the host"]
NEWS_INTENTS = {"discourse.intent.news_query"}
# words after "actually" that continue the question instead of replacing a word ("what does the word ubiquitous
# actually mean exactly": "ubiquitous" was corrected away and the web query became "word mean exactly")
NOT_REPLACEMENT = {"mean", "means", "meant", "exactly", "work", "works", "is", "do", "does", "worth", "cost", "costs",
                   "happen", "happened", "happening", "go", "goes", "say", "says"}

PRONOUNS = {"it", "its", "it's", "he", "she", "his", "her", "him", "they", "them", "their", "there", "that", "this",
            "those", "these"}
PERSONAL = {"he", "she", "his", "her", "him", "hers", "himself", "herself"}
WEAK_PRONOUNS = {"there", "that", "this", "those", "these"}     # often not referential: only for short questions
NUMBER_TYPES = {"date", "time", "time_of_day", "duration", "money", "percent", "cardinal", "ordinal", "quantity",
                "number", "numeral", "age", "year"}

# --- the rest of Arc Tagger's per-word read-out (beyond intent + slots) -----------------------------------------------
# Personal data that must not leave the device in a web query: span types precise enough to redact. (private_name is
# NOT here: on this eval it fires on "Nvidia", "Mount", "Toyota", "Mozart"; the utterance-level pii.request / pii.leak
# flags fire on most ordinary questions, so neither is used.)
PII_REDACT = {"safety.pii.phone_number": "phone", "safety.pii.email": "email", "safety.pii.street_address": "address",
              "safety.pii.card_account_number": "card", "safety.pii.gov_id": "id", "safety.pii.date_of_birth": "birthday",
              "safety.pii.url_handle": "handle"}
# Relative time said in the question (Arc's typed entity.number.date / duration / time_of_day span, or the words):
# drives Tavily freshness. Absolute dates ("in 2024", "March 3rd 2025") mean history: no freshness filter.
_WHEN_DAY = re.compile(r"\b(today|tonight|right now|now|this morning|this afternoon|this evening|currently|live|"
                       r"24 hours|twenty four hours|at the moment)\b")
_WHEN_WEEK = re.compile(r"\b(yesterday|last night|weekend|this week|past week|last week|sunday|saturday|monday|"
                        r"tuesday|wednesday|thursday|friday|recent|recently|latest|lately|few days|past few days)\b")
# results of games / races / matches are news (dated pages), not encyclopedia pages
_RESULTS = {"game", "match", "score", "scores", "won", "win", "winner", "race", "prix", "beat", "lost", "playoff",
            "playoffs", "final", "finals", "result", "results", "standings", "fight", "tournament"}
_WEATHER_W = {"weather", "rain", "raining", "rainy", "snow", "snowing", "sunny", "temperature", "forecast", "cold", "hot",
              "warm", "wind", "windy", "humid", "humidity", "storm", "umbrella", "degrees", "cloudy", "fog", "heat"}
_MARKET = re.compile(r"\b(s ?(?:and|&)? ?p 500|s p 500|nasdaq|dow jones|the dow|stock market)\b")
_FINANCE_W = {"stock", "stocks", "price", "worth", "rate", "exchange", "bitcoin", "crypto", "ethereum", "market",
              "shares", "dollars", "euros", "dollar", "euro", "yen", "pounds"}

_PUNCT = re.compile(r"[^\w'\- ]+")


def norm(s: str) -> str:
    return _PUNCT.sub("", s.lower()).strip()


@dataclass
class Frame:
    words: list[str]
    intent: str = ""                 # best fine intent (discourse.intent.*) that is on, else coarse intent.*
    intent_p: float = 0.0
    coarse: str = ""
    slots: tuple = ()                # sorted ((type, text), ...)
    content: tuple = ()              # content words, in order, after correction handling
    corrected: bool = False
    carried: str = ""                # entity inherited from the previous question ("when was it founded")
    followup_of: str = ""            # previous question, when this one is an elliptical follow-up ("and when?")
    route: str = "web"               # web | local | both
    act_question: bool = False
    command_complete: bool = False
    spans: list = field(default_factory=list)
    dropped: frozenset = frozenset()  # token indices removed by a self-correction (cue words + corrected-away entity)
    phrases: tuple = ()              # entity phrases as spoken (internal stopwords kept): "pride and prejudice"
    # the rest of the read-out (shown live; freshness / topic / redaction use it)
    acts: tuple = ()                 # dialogue acts that are on: ("question",), ("command_request",) ...
    directed: float | None = None    # discourse.assistant.device_directed probability
    redacted: tuple = ()             # kinds of personal data kept out of the web query: ("phone", "email")
    redact_tok: frozenset = frozenset()
    cue: frozenset = frozenset()      # token indices of the self-correction cue ("actually make that")
    content_idx: tuple = ()          # token index of each content word (what goes into the search query)
    jailbreak: bool = False          # safety.jailbreak.attempt on the spoken question
    when: str = ""                   # "day" | "week" | "" : relative time the question is about
    when_text: str = ""              # the words that said it ("last night", "right now")
    ent_types: tuple = ()            # entity mention types in spoken order (person, organization, place, team, ...)
    num_types: tuple = ()            # typed number spans (date, money, percent, duration, ...)

    @property
    def n(self):
        return len(self.words)

    @property
    def text(self):
        return " ".join(self.words)

    @property
    def sig(self):
        return (self.intent, self.slots, self.content)

    @property
    def meaning_sig(self):
        """Coarser signature used to decide if a drafted answer still fits: intent + slots + content words."""
        return (self.intent, self.slots, tuple(sorted(set(self.content))))

    @property
    def clean_text(self) -> str:
        """The question as the speaker meant it: correction cue and the corrected-away entity removed ("who founded
        Microsoft, sorry, I meant Amazon" → "who founded Amazon"), so the answer model doesn't comment on Microsoft."""
        if not self.dropped and not self.redact_tok:
            return self.text
        out, masked = [], False
        for i, w in enumerate(self.words):
            if i in self.dropped:
                continue
            if i in self.redact_tok:
                if not masked:
                    out.append("[private]")       # personal data never leaves the device
                masked = True
                continue
            masked = False
            out.append(w)
        return " ".join(out)

    @property
    def resolved_text(self) -> str:
        """The question with its carried referent spelled out, for ranking and for the answer model."""
        t = self.clean_text
        if self.followup_of:
            return f"{t} (follow-up to: {self.followup_of})"
        return f"{t} ({self.carried})" if self.carried else t

    def entity_query(self) -> str:
        """A second, entity-only web query ("turkey", "ludwig beethoven" → the entity's own article, whose intro
        usually holds the capital / birthplace / founder / HQ that the content query's articles only mention in
        passing). The phrase said after a correction wins, then the last entity. Empty when it adds nothing."""
        if not self.phrases:
            return ""
        e = self.phrases[-1]
        return "" if e == self.query() or not e.strip() else e

    def query(self) -> str:
        """Search query: content words in spoken order + an intent hint word when it adds information."""
        q = list(self.content)
        hint = INTENT_HINT.get(self.intent, "") if (self.intent_p >= 0.5 and self.n >= 3) else ""
        if hint and not any(h in q for h in hint.split()):
            q.append(hint)
        return " ".join(q)

    def web_hints(self) -> dict:
        """Tavily topic / freshness from the read-out (lanes.web_plan applies them):
          - finance intent, or a money span / finance word with a price-like question -> topic finance (+ max age)
          - news intent -> topic news; a relative time said ("today", "last night") -> time_range day | week
          - a result question (game / race / score) said with a relative time ("last night", "this weekend",
            "right now") -> topic news + that time_range: results are dated news pages, not encyclopedia articles
        Weather keeps lanes.web_plan's own max_age. Returns {} when nothing applies (plain general search)."""
        h: dict = {}
        toks = set(self.content)
        if self.intent == "discourse.intent.weather_query" and not toks & _WEATHER_W:
            # Arc's weather intent with no weather word ("is it bad to take ibuprofen on an empty stomach"): not a
            # forecast, so no 6-hour freshness filter (it would drop every evergreen health page)
            return {"weather_off": True, "why": "no weather word"}
        if self.intent in ("discourse.intent.qa_stock", "discourse.intent.qa_currency") and self.intent_p >= 0.35:
            h["topic"] = "finance"
        elif _MARKET.search(" ".join(norm(w) for w in self.words)):
            h["topic"] = "finance"
        elif ("money" in self.num_types or toks & {"bitcoin", "crypto", "ethereum", "stock", "stocks", "shares"}) and \
                toks & _FINANCE_W and self.intent != "discourse.intent.weather_query":
            h["topic"] = "finance"
        elif self.intent == "discourse.intent.news_query":
            h["topic"] = "news"
            if self.when:
                h["time_range"] = self.when
        elif self.when and (toks & _RESULTS or "team" in self.ent_types or "event_name" in self.ent_types) \
                and self.intent != "discourse.intent.weather_query":
            h["topic"] = "news"
            h["time_range"] = self.when
        if h:
            h["why"] = self.when_text or ""
        return h

    def readout(self) -> dict:
        """Compact live read-out for the UI frame card."""
        h = self.web_hints()
        fresh = ""
        if h.get("topic") and h.get("topic") != "general":
            fresh = h["topic"] + (f" · past {h['time_range']}" if h.get("time_range") else "")
        elif self.intent == "discourse.intent.weather_query" and not h.get("weather_off"):
            fresh = "weather · last 6 h"
        return {"acts": list(self.acts), "directed": None if self.directed is None else round(self.directed, 2),
                "redacted": list(self.redacted), "jailbreak": self.jailbreak, "when": self.when,
                "when_text": self.when_text, "fresh": fresh, "ent_types": list(self.ent_types),
                "num_types": list(self.num_types)}

    def has_content(self) -> bool:
        return len(self.content) >= 2 or bool(self.slots)

    def to_json(self):
        return {"intent": self.intent, "p": round(self.intent_p, 2), "coarse": self.coarse,
                "slots": [list(s) for s in self.slots], "content": list(self.content), "route": self.route,
                "corrected": self.corrected, "carried": self.carried, "query": self.query(), "n": self.n,
                "command_complete": self.command_complete, "readout": self.readout(),
                # per-token read-out for the annotated transcript (indices into words)
                "content_idx": list(self.content_idx), "dropped": sorted(self.dropped), "cue": sorted(self.cue),
                "redact": sorted(self.redact_tok)}


def _correction_at(toks: list[str]) -> int:
    """Index just after the last self-correction marker, or -1."""
    last = -1
    for i in range(len(toks)):
        for pat in CORRECTION:
            if tuple(toks[i:i + len(pat)]) == pat:
                last = max(last, i + len(pat))
    return last


def _cue_tokens(toks: list[str], cut: int) -> set:
    """Token indices of every correction cue that ends at or before `cut`."""
    out = set()
    if cut <= 0:
        return out
    for i in range(len(toks)):
        for pat in CORRECTION:
            if tuple(toks[i:i + len(pat)]) == pat and i + len(pat) <= cut:
                out.update(range(i, i + len(pat)))
    return out


def _cap(w: str) -> bool:
    return bool(w) and (w[0].isupper() or w[0].isdigit())


def _phrases(words: list[str], toks: list[str], slot_spans: list[dict], cut: int, drop_tok: set) -> list[str]:
    """Entity phrases for the entity-only query, in spoken order. Mention spans keep their inner stopwords ("united
    states", "pride and prejudice") and grow over the proper-noun run around them (the tagger's span is sometimes
    just "Mount" of "Mount Kilimanjaro", "Apollo" of "Apollo 11"); after a correction with no tagged entity, the words
    after the cue are the replacement ("sorry, I mean pride and prejudice"), trimmed of edge stopwords."""
    out = []
    for s in slot_spans:
        if _cls(s) != "mention":
            continue
        a, b = s["start"], s["end"]
        while b < len(words) and b not in drop_tok and _cap(words[b]) and toks[b] and toks[b] not in STOP:
            b += 1
        while a - 1 > 0 and a - 1 not in drop_tok and _cap(words[a - 1]) and toks[a - 1] and toks[a - 1] not in STOP:
            a -= 1
        ws = [t for t in toks[a:b] if t]
        while ws and ws[0] in STOP:
            ws = ws[1:]
        while ws and ws[-1] in STOP:
            ws = ws[:-1]
        if ws:
            out.append((s["start"], " ".join(ws)))
    if cut > 0 and not any(st >= cut for st, _ in out):
        ws = [toks[i] for i in range(cut, len(toks)) if toks[i] and i not in drop_tok]
        while ws and ws[0] in STOP:
            ws = ws[1:]
        while ws and ws[-1] in STOP:
            ws = ws[:-1]
        if ws and len(ws) <= 5:
            out.append((cut, " ".join(ws)))
    out.sort()
    return [p for _, p in out]


def _cls(s: dict) -> str:
    """Correction class: any entity mention replaces any entity mention; numbers replace the same number type."""
    return "mention" if s["type"].startswith("entity.mention") else s["type"]


def build(words: list[str], spans: list[dict], current: dict | None, events_seen: set | None = None,
          prev_intent: tuple | None = None, local_ok: bool = True, carry: dict | None = None) -> Frame:
    toks = [norm(w) for w in words]
    f = Frame(words=list(words), spans=spans)
    cur = current or {}
    intents = [x for x in (cur.get("intent") or []) if x.get("on")]
    fine = [x for x in intents if x["tag"].startswith("discourse.intent.")]
    coarse = [x for x in intents if x["tag"].startswith("intent.")]
    if fine:
        f.intent, f.intent_p = fine[0]["tag"], float(fine[0]["p"])
    elif coarse:
        f.intent, f.intent_p = coarse[0]["tag"], float(coarse[0]["p"])
    f.coarse = coarse[0]["tag"] if coarse else ""
    if (not fine or (f.intent.startswith("intent.") and len(words) > 2)) and prev_intent and prev_intent[0]:
        # a correction or filler word can drop the fine intent for a word or two; keep the last confident one
        if prev_intent[0].startswith("discourse.intent.") and not fine:
            f.intent, f.intent_p = prev_intent
    f.act_question = any(x["tag"] == "discourse.act.question" and x.get("on") for x in (cur.get("act") or []))
    f.acts = tuple(x["tag"].split(".")[-1] for x in (cur.get("act") or [])
                   if x.get("on") and x["tag"].startswith("discourse.act."))
    f.directed = next((float(x["p"]) for x in (cur.get("act") or [])
                       if x["tag"] == "discourse.assistant.device_directed"), None)
    f.jailbreak = any(x["tag"] == "safety.jailbreak.attempt" and x.get("on") for x in (cur.get("safety") or []))
    # personal data spans: their words stay out of the query, the slots and the entity phrases
    pii = [s for s in spans if s["type"] in PII_REDACT and s["end"] <= len(toks)]
    redact_tok = set()
    for s in pii:
        redact_tok.update(range(s["start"], s["end"]))
    f.redact_tok = frozenset(redact_tok)
    f.redacted = tuple(dict.fromkeys(PII_REDACT[s["type"]] for s in pii))
    f.command_complete = bool(events_seen and "discourse.assistant.command_complete" in events_seen)

    # slots: entity mentions + typed numbers (drop the generic entity.numeral duplicate)
    slot_spans = [s for s in spans if s.get("family") in ("entity", "number") and s["type"] != "entity.numeral"
                  and s["end"] <= len(toks) and not set(range(s["start"], s["end"])) & redact_tok]
    cut = _correction_at(toks)
    drop_tok = set()
    cue_tok = _cue_tokens(toks, cut)
    f.cue = frozenset(cue_tok)
    if cue_tok:
        # the tagger sometimes tags the cue itself as an entity ("Sari," → place, "Scratch" → organization); such a
        # span would be "corrected away" instead of the real one (q31: Paris survived, "Sari" was dropped)
        slot_spans = [s for s in slot_spans if not set(range(s["start"], s["end"])) & cue_tok]
    if cut > 0:
        f.corrected = True
        after_types = {_cls(s) for s in slot_spans if s["start"] >= cut}
        # for each corrected class, drop the LAST span of that class said before the marker
        drop_ids = set()
        for c in after_types:
            before = [i for i, s in enumerate(slot_spans) if s["start"] < cut and _cls(s) == c]
            if before:
                drop_ids.add(before[-1])
        keep = []
        for i, s in enumerate(slot_spans):
            if i in drop_ids:
                drop_tok.update(range(s["start"], s["end"]))      # the corrected-away entity
                continue
            keep.append(s)
        slot_spans = keep
        # marker words themselves
        for i in range(len(toks)):
            for pat in CORRECTION:
                if tuple(toks[i:i + len(pat)]) == pat and i + len(pat) <= cut:
                    drop_tok.update(range(i, i + len(pat)))
        before_mentions = [s for s in slot_spans if s["start"] < cut and _cls(s) == "mention"]
        if not after_types and before_mentions and any(toks[i] and toks[i] not in STOP for i in range(cut, len(toks))):
            # "where was Mozart born, actually make that Beethoven" with the replacement untagged: the corrected thing
            # is the entity said before the cue, not the last content word ("born" was dropped, Mozart kept)
            s0 = before_mentions[-1]
            drop_tok.update(range(s0["start"], s0["end"]))
            slot_spans = [s for s in slot_spans if s is not s0]
        elif not after_types:
            # "the Lakers game, sorry, Warriors game": no typed span after the marker. Treat the word right after
            # the marker as replacing the content word right before it.
            after = [i for i in range(cut, len(toks)) if toks[i] and toks[i] not in STOP]
            before = [i for i in range(0, cut) if toks[i] and toks[i] not in STOP and i not in drop_tok]
            if after and before and not all(toks[i] in NOT_REPLACEMENT for i in after):
                # drop the before-token(s) that the after-phrase repeats around (e.g. "game")
                rep = {toks[i] for i in after}
                for i in reversed(before):
                    if toks[i] in rep:
                        continue
                    drop_tok.add(i)
                    break
    if drop_tok:
        # a slot whose words were corrected away goes too ("capital of Canada, actually make that Turkey")
        slot_spans = [s for s in slot_spans if not set(range(s["start"], s["end"])) & drop_tok]
    f.slots = tuple(sorted({(s["type"].split(".")[-1], norm(s["text"])) for s in slot_spans}))
    f.dropped = frozenset(drop_tok)
    f.ent_types = tuple(dict.fromkeys(s["type"].split(".")[-1] for s in slot_spans
                                      if s["type"].startswith("entity.mention")))
    f.num_types = tuple(dict.fromkeys(s["type"].split(".")[-1] for s in slot_spans
                                      if s["type"].startswith("entity.number")))
    low_kept = " ".join(t for i, t in enumerate(toks) if t and i not in drop_tok)
    time_spans = [norm(s["text"]) for s in slot_spans
                  if s["type"] in ("entity.number.date", "entity.number.duration", "entity.number.time_of_day")]
    for src in time_spans + [low_kept]:
        m = _WHEN_DAY.search(src)
        if m:
            f.when, f.when_text = "day", m.group(0)
            break
        m = _WHEN_WEEK.search(src)
        if m:
            f.when, f.when_text = "week", m.group(0)
            break
    f.phrases = tuple(_phrases(f.words, toks, slot_spans, cut, drop_tok | redact_tok))
    money_tok = {i for s in slot_spans if s["type"] in ("entity.number.money", "entity.number.percent")
                 for i in range(s["start"], s["end"])}
    seen, content, content_idx = set(), [], []
    for i, t in enumerate(toks):
        if i in money_tok and t and i not in drop_tok and i not in redact_tok:
            # keep a typed amount as said: "$100", "15%" (norm() would strip the unit symbol: "100")
            raw = re.sub(r"[^\w$%.,€£¥]", "", words[i]).strip(".,").lower()
            t = raw or t
        if not t or t in STOP or i in drop_tok or i in redact_tok or t in seen:
            continue
        seen.add(t)
        content.append(t)
        content_idx.append(i)
    f.content_idx = tuple(content_idx)
    # conversation memory: "who founded Nvidia?" → "when was it founded?" — a pronoun and no entity of its own
    # inherits the previous question's entity (carry = that frame's mention slots)
    has_mention = any(t not in NUMBER_TYPES for t, _ in f.slots)
    strong = any(t in PRONOUNS - WEAK_PRONOUNS for t in toks)
    weak = any(t in WEAK_PRONOUNS for t in toks) and len(content) <= 2
    ent = None
    if carry and not has_mention and (strong or weak):
        # he/she → a person (often named only in the previous ANSWER: "relativity?" → "Einstein" → "where was he
        # born"); it/they/there → the thing the previous question was about, else any entity
        personal = any(t in PERSONAL for t in toks)
        order = ["person", "thing"] if personal else ["thing", "person"]
        for k in order:
            if carry.get(k):
                ent = carry[k][0]
                break
    # elliptical follow-ups with no pronoun: "and when?", "what about Nvidia?", "how about in 2020?", "and the CEO?"
    prev = (carry or {}).get("prev")
    head2 = toks[:2]
    elliptic = bool(prev) and (toks[:1] in (["and"], ["also"]) or head2 in (["what", "about"], ["how", "about"])
                               or (len(content) == 0 and len(toks) <= 4))
    if elliptic and not ent:
        pc = list(prev.get("content") or [])
        pents = [w for e in prev.get("entities") or [] for w in e.split()]
        swap = head2 in (["what", "about"], ["how", "about"])
        wh = None if swap else next((t for t in toks if t in ("when", "where", "who", "why", "how", "which", "whom",
                                                               "whose")), None)
        if has_mention or (swap and content):
            # "what about Nvidia": same question, new subject → swap the previous entity for the new one (the words
            # after "about" are the new subject even when the tagger doesn't mark them in so short a phrase)
            base = [w for w in pc if w not in pents]
            content = base + [w for w in content if w not in base]
        else:
            # "and when?": the previous question's subject and verb + the new question word
            content = pc + [w for w in content if w not in pc]
            if prev.get("entities"):
                f.carried = prev["entities"][0]
                f.phrases = (prev["entities"][0],)
        if wh and wh not in content:
            content.append(wh)
        f.followup_of = prev.get("question", "")
        if not f.intent and prev.get("intent"):
            f.intent = prev["intent"]
    if ent:
        f.carried = ent[1]
        f.phrases = (ent[1],)
        f.slots = tuple(sorted(set(f.slots) | {ent}))
        if prev and not content:
            # "and when was that?": a pronoun AND nothing else said → the previous question's meaning + question word
            content = [w for w in (prev.get("content") or [])]
            wh = next((t for t in toks if t in ("when", "where", "who", "why", "how", "which")), None)
            if wh and wh not in content:
                content.append(wh)
            f.followup_of = prev.get("question", "")
        for w in reversed(ent[1].split()):
            if w not in content:
                content.insert(0, w)
    f.content = tuple(content)

    low = " " + " ".join(toks) + " "
    local = any(" " + c + " " in low for c in LOCAL_CUES)
    # local corpus only when the question points at it ("in the game", "the judge", ...): searching it for every
    # web question put unrelated moments into the sources (eval smoke run, q26)
    f.route = "local" if (local and local_ok) else "web"
    return f
