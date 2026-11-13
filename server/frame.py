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
someone thing things""".split())
# hey/ok wake fillers are already in STOP; these mark a spoken self-correction
CORRECTION = [("i", "mean"), ("sorry",), ("no", "wait"), ("actually",), ("rather",), ("scratch", "that"),
              ("not", "that"), ("make", "that"), ("i", "meant")]
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
    route: str = "web"               # web | local | both
    act_question: bool = False
    command_complete: bool = False
    spans: list = field(default_factory=list)

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

    def query(self) -> str:
        """Search query: content words in spoken order + an intent hint word when it adds information."""
        q = list(self.content)
        hint = INTENT_HINT.get(self.intent, "") if (self.intent_p >= 0.5 and self.n >= 3) else ""
        if hint and not any(h in q for h in hint.split()):
            q.append(hint)
        return " ".join(q)

    def has_content(self) -> bool:
        return len(self.content) >= 2 or bool(self.slots)

    def to_json(self):
        return {"intent": self.intent, "p": round(self.intent_p, 2), "coarse": self.coarse,
                "slots": [list(s) for s in self.slots], "content": list(self.content), "route": self.route,
                "corrected": self.corrected, "query": self.query(), "n": self.n,
                "command_complete": self.command_complete}


def _correction_at(toks: list[str]) -> int:
    """Index just after the last self-correction marker, or -1."""
    last = -1
    for i in range(len(toks)):
        for pat in CORRECTION:
            if tuple(toks[i:i + len(pat)]) == pat:
                last = max(last, i + len(pat))
    return last


def _cls(s: dict) -> str:
    """Correction class: any entity mention replaces any entity mention; numbers replace the same number type."""
    return "mention" if s["type"].startswith("entity.mention") else s["type"]


def build(words: list[str], spans: list[dict], current: dict | None, events_seen: set | None = None,
          prev_intent: tuple | None = None, local_ok: bool = True) -> Frame:
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
    f.command_complete = bool(events_seen and "discourse.assistant.command_complete" in events_seen)

    # slots: entity mentions + typed numbers (drop the generic entity.numeral duplicate)
    slot_spans = [s for s in spans if s.get("family") in ("entity", "number") and s["type"] != "entity.numeral"
                  and s["end"] <= len(toks)]
    cut = _correction_at(toks)
    drop_tok = set()
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
        if not after_types:
            # "the Lakers game, sorry, Warriors game": no typed span after the marker. Treat the word right after
            # the marker as replacing the content word right before it.
            after = [i for i in range(cut, len(toks)) if toks[i] and toks[i] not in STOP]
            before = [i for i in range(0, cut) if toks[i] and toks[i] not in STOP and i not in drop_tok]
            if after and before:
                # drop the before-token(s) that the after-phrase repeats around (e.g. "game")
                rep = {toks[i] for i in after}
                for i in reversed(before):
                    if toks[i] in rep:
                        continue
                    drop_tok.add(i)
                    break
    f.slots = tuple(sorted({(s["type"].split(".")[-1], norm(s["text"])) for s in slot_spans}))
    seen, content = set(), []
    for i, t in enumerate(toks):
        if not t or t in STOP or i in drop_tok or t in seen:
            continue
        seen.add(t)
        content.append(t)
    f.content = tuple(content)

    low = " " + " ".join(toks) + " "
    local = any(" " + c + " " in low for c in LOCAL_CUES)
    # local corpus only when the question points at it ("in the game", "the judge", ...): searching it for every
    # web question put unrelated moments into the sources (eval smoke run, q26)
    f.route = "local" if (local and local_ok) else "web"
    return f
