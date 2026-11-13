"""Search narrowing from Arc Tagger's read-out (intent, typed entity / number spans, relative time).

Three independent levers, each behind a config flag and measured by eval/narrow_eval.py (runs/narrow_eval.json):
  family()       : what kind of question this is (weather, sports, finance, recipe, tech, health, definition, ...),
                   from Arc's intent where it is reliable plus the content words where it is not (Arc's fine intents
                   are noisy on this eval: "screenshot on a Mac" = cooking_recipe, "ramen places" = general_joke)
  domains()      : Tavily include_domains for that family (NARROW_DOMAINS: off | prefer | restrict)
  shape_query()  : query construction from the slots (NARROW_QUERY): generic spoken words dropped, a family noun added
                   ("lakers game last night" -> "... score"), relative dates spelled out ("tomorrow" -> "November 13
                   2026") so the search engine sees the date the pages carry
  rerank()       : reorder a result list with the read-out (NARROW_RERANK): entity phrase / content-word coverage in
                   title+snippet, family domain match, freshness when the question is about a relative time
  specific()     : is a partial frame worth a speculative credit (NARROW_GATE)
"""
from __future__ import annotations

import datetime as dt
import re
from urllib.parse import urlparse

from . import config

WEATHER_W = {"weather", "rain", "raining", "snow", "snowing", "sunny", "temperature", "forecast", "humid", "humidity",
             "storm", "umbrella", "degrees", "cloudy", "windy"}
RESULT_W = {"score", "scores", "won", "win", "winner", "beat", "lost", "playoff", "playoffs", "standings", "game",
            "match", "prix", "race", "fixture", "fixtures", "playing", "league", "tournament"}
SPORT_NAMES = {"nba", "nfl", "nhl", "mlb", "f1", "formula", "lakers", "warriors", "celtics", "chiefs", "united",
               "premier", "football", "soccer", "basketball", "baseball", "hockey", "tennis", "cricket", "grand"}
HEALTH_W = {"symptoms", "symptom", "ibuprofen", "fever", "throat", "strep", "pain", "medicine", "dose", "dosage",
            "disease", "infection", "stomach", "headache", "allergy", "vaccine", "blood", "pressure", "diabetes",
            "tylenol", "acetaminophen", "aspirin", "flu", "covid", "cold", "cough", "rash"}
COOK_W = {"recipe", "bake", "baking", "boil", "boiled", "cook", "cooking", "buttermilk", "pancakes", "egg", "eggs",
          "yolk", "substitute", "instead", "oven", "roast", "fry", "grill", "flour", "dough", "sauce"}
CONVERT_W = {"ounces", "ounce", "teaspoons", "teaspoon", "tablespoon", "tablespoons", "cups", "cup", "miles", "mile",
             "kilometers", "km", "grams", "pounds", "liters", "gallons", "inches", "feet", "convert", "conversion"}
TECH_W = {"iphone", "ipad", "mac", "macbook", "android", "windows", "pixel", "screenshot", "screen", "charge",
          "charging", "laptop", "wifi", "bluetooth", "app", "settings", "reset", "update", "printer", "router"}
HOME_W = {"stain", "carpet", "clean", "cleaning", "remove", "mold", "smell", "laundry", "wash"}
DEF_W = {"mean", "meaning", "definition", "define", "word", "term", "stands"}
TRAVEL_W = {"visa", "passport", "flight", "flights", "visit", "travel", "trip", "vacation", "airport", "layover"}
REVIEW_W = {"best", "better", "vs", "versus", "review", "reviews", "compare", "comparison", "recommend", "top"}
PRODUCT_W = {"vacuum", "phone", "laptop", "headphones", "camera", "tv", "pro", "iphone", "pixel", "galaxy", "watch",
             "earbuds", "router", "mattress", "blender"}
LOCAL_W = {"restaurant", "restaurants", "places", "bar", "bars", "cafe", "coffee", "ramen", "pizza", "sushi", "food",
           "eat", "near", "nearby", "hotel", "hotels"}
HOURS_W = {"open", "close", "closes", "closing", "hours", "opening"}
NEWS_W = {"news", "headlines", "launch", "upcoming", "announced", "announcement", "latest"}
FINANCE_W = {"stock", "stocks", "shares", "bitcoin", "crypto", "ethereum", "nasdaq", "dow", "s&p", "market", "euros",
             "dollars", "exchange", "rate"}

FAMILY_DOMAINS = {
    "weather": ["weather.gov", "accuweather.com", "weather.com", "wunderground.com", "timeanddate.com",
                "metoffice.gov.uk", "weather-forecast.com"],
    "sports": ["espn.com", "cbssports.com", "nba.com", "nfl.com", "formula1.com", "skysports.com", "bbc.com",
               "premierleague.com", "sports.yahoo.com", "foxsports.com", "theathletic.com", "reuters.com"],
    "finance": ["finance.yahoo.com", "marketwatch.com", "cnbc.com", "reuters.com", "bloomberg.com", "coindesk.com",
                "xe.com", "investing.com", "google.com", "fool.com"],
    "news": ["reuters.com", "apnews.com", "bbc.com", "cnn.com", "nytimes.com", "theverge.com", "techcrunch.com",
             "cnbc.com", "npr.org", "theguardian.com"],
    "health": ["mayoclinic.org", "clevelandclinic.org", "nhs.uk", "cdc.gov", "medlineplus.gov", "webmd.com",
               "healthline.com", "hopkinsmedicine.org", "nih.gov"],
    "recipe": ["seriouseats.com", "allrecipes.com", "foodnetwork.com", "bonappetit.com", "simplyrecipes.com",
               "kingarthurbaking.com", "thekitchn.com", "bbcgoodfood.com", "epicurious.com", "americastestkitchen.com"],
    "convert": ["kingarthurbaking.com", "calculatorsoup.com", "rapidtables.com", "metric-conversions.org",
                "inchcalculator.com", "convertunits.com", "allrecipes.com", "britannica.com"],
    "tech": ["support.apple.com", "apple.com", "support.google.com", "support.microsoft.com", "howtogeek.com",
             "macrumors.com", "theverge.com", "lifewire.com", "pcmag.com"],
    "home": ["goodhousekeeping.com", "thespruce.com", "bobvila.com", "realsimple.com", "marthastewart.com",
             "bhg.com", "wikihow.com"],
    "definition": ["merriam-webster.com", "dictionary.cambridge.org", "dictionary.com", "vocabulary.com",
                   "britannica.com", "investopedia.com", "nerdwallet.com", "en.wikipedia.org", "irs.gov"],
    "travel": ["travel.state.gov", "japan.travel", "lonelyplanet.com", "visiticeland.com", "cntraveler.com",
               "tripadvisor.com", "kayak.com", "google.com", "expedia.com"],
    "reviews": ["rtings.com", "nytimes.com", "techradar.com", "cnet.com", "tomsguide.com", "theverge.com",
                "consumerreports.org", "gsmarena.com", "dxomark.com", "pcmag.com"],
    "local": ["yelp.com", "tripadvisor.com", "eater.com", "timeout.com", "theinfatuation.com", "seattletimes.com"],
    "reference": ["en.wikipedia.org", "britannica.com", "history.com", "nasa.gov", "nationalgeographic.com",
                  "biography.com"],
}
# a noun the search engine needs that the speaker left implicit
FAMILY_NOUN = {"weather": "forecast", "sports": "score", "finance": "price", "recipe": "", "health": "",
               "definition": "definition", "convert": "conversion"}
# words that carry no search information in a spoken question (beyond frame.STOP)
GENERIC = {"gonna", "going", "doing", "exactly", "something", "usual", "really", "one's", "anymore", "lot", "happening",
           "kid", "decimal", "first-", "else", "stuff", "whatever", "anything", "okay", "um", "hey", "should"}
_MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
           "November", "December"]
_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


def today() -> dt.date:
    return dt.date.fromisoformat(config.NARROW_TODAY) if config.NARROW_TODAY else dt.date.today()


def _toks(f) -> set:
    return set(f.content)


def family(f) -> str:
    """The question's family from the read-out ('' = general web search, no narrowing)."""
    t = _toks(f)
    intent = f.intent
    h = f.web_hints()
    if t & WEATHER_W:
        return "weather"
    if h.get("topic") == "finance" or intent in ("discourse.intent.qa_stock", "discourse.intent.qa_currency") and \
            t & FINANCE_W:
        return "finance"
    if (t & RESULT_W) and (t & SPORT_NAMES or "team" in f.ent_types or h.get("time_range")):
        return "sports"
    if (intent == "discourse.intent.news_query" and (f.intent_p >= 0.5 or f.when)) or h.get("time_range") \
            or t & NEWS_W:
        return "news"
    if t & HEALTH_W:
        return "health"
    if t & CONVERT_W and (t & {"many", "much"} or "how" in [w.lower() for w in f.words] or "in" in f.words):
        return "convert"
    if t & COOK_W:
        return "recipe"
    if t & HOME_W:
        return "home"
    if t & TECH_W and not t & REVIEW_W:
        return "tech"
    if t & DEF_W:
        return "definition"
    if t & {"ira", "401k", "roth"}:
        return "definition"
    if t & TRAVEL_W:
        return "travel"
    if t & REVIEW_W and (t & PRODUCT_W or "product" in f.ent_types):
        return "reviews"
    if t & LOCAL_W and not t & HOURS_W:
        return "local"
    if t & HOURS_W:
        return ""                     # opening hours: the official site, which we can't name in advance
    if intent in ("discourse.intent.qa_factoid", "intent.request_information", "discourse.intent.qa_definition",
                  "") and not f.when:
        return "reference"
    return ""


def domains(f, mode: str | None = None) -> tuple[list[str], str]:
    """(include_domains, include_domains_mode) for a frame; ([], '') when no narrowing applies."""
    mode = mode or config.NARROW_DOMAINS
    if mode in ("", "off", "0"):
        return [], ""
    fam = family(f)
    allowed = set(config.NARROW_DOMAIN_FAMILIES.split(",")) if config.NARROW_DOMAIN_FAMILIES else None
    if not fam or fam not in FAMILY_DOMAINS or (allowed is not None and fam not in allowed):
        return [], ""
    m = mode
    if m == "restrict" and fam in config.NARROW_PREFER_ONLY.split(","):
        m = "prefer"
    return list(FAMILY_DOMAINS[fam]), m


def _date_str(d: dt.date) -> str:
    return f"{_MONTHS[d.month - 1]} {d.day} {d.year}"


def resolve_dates(words: list[str], ref: dt.date | None = None) -> dict[str, str]:
    """Relative day words said in the question -> explicit date strings ("tomorrow" -> "November 13 2026")."""
    ref = ref or today()
    out = {}
    for w in words:
        lw = w.lower()
        if lw in ("today", "today's", "tonight"):
            out[lw] = _date_str(ref)
        elif lw == "tomorrow":
            out[lw] = _date_str(ref + dt.timedelta(days=1))
        elif lw == "yesterday":
            out[lw] = _date_str(ref - dt.timedelta(days=1))
        elif lw in _WEEKDAYS:          # the most recent such day (results questions are about the past)
            back = (ref.weekday() - _WEEKDAYS.index(lw)) % 7 or 7
            out[lw] = _date_str(ref - dt.timedelta(days=back))
    return out


def shape_query(f, base: str) -> str:
    """The frame's query, shaped by the read-out: generic words out, relative dates spelled out, family noun in."""
    if not config.NARROW_QUERY:
        return base
    fam = family(f)
    ws = [w for w in base.split() if w not in GENERIC]
    if config.NARROW_DATES and fam in ("weather", "sports", "news", "finance"):
        dates = resolve_dates(ws)
        if "night" in ws and "last" in ws:
            dates["last"] = _date_str(today() - dt.timedelta(days=1))
            ws = [w for w in ws if w != "night"]
        if "weekend" in ws:
            sat = today() - dt.timedelta(days=(today().weekday() - 5) % 7 or 7)
            if "this" in f.text.lower() and fam == "weather" and today().weekday() < 5:
                sat = today() + dt.timedelta(days=(5 - today().weekday()))
            dates["weekend"] = _date_str(sat)
        ws = [dates.get(w, w) for w in ws if w not in ("past", "this")]
    if f.web_hints().get("weather_off"):            # Arc's weather intent on a non-weather question: drop its hint
        ws = [w for w in ws if w not in ("weather", "forecast")] or ws
    noun = FAMILY_NOUN.get(fam, "")
    st = set(ws)
    if fam == "sports" and (st & {"score", "won", "win", "winner", "result", "beat", "next", "playing", "when"}
                            or not st & {"game", "match", "race", "prix"}):
        noun = ""
    if fam == "finance" and st & {"rate", "exchange", "worth", "doing"}:
        noun = ""
    for n in noun.split():
        if n not in ws:
            ws.append(n)
    return " ".join(dict.fromkeys(ws)) or base


def specific(f) -> bool:
    """A partial frame is worth a speculative credit only with >= NARROW_GATE_MIN_CONTENT content words: a lone
    entity prefix ("kansas", "manchester", "apple", "$100") never returned a result the final query kept
    (runs/narrow_eval.json: 0 of 4-6 such fires useful). 3 would cut 38 % of speculative credits but loses one
    early top-1 hit in 9 (measured; not the default)."""
    if not config.NARROW_GATE:
        return True
    return len(f.content) >= config.NARROW_GATE_MIN_CONTENT


# ------------------------------------------------------------------------------------------------------- rerank
def _host(url: str) -> str:
    try:
        return urlparse(url).netloc.lower().removeprefix("www.")
    except ValueError:
        return ""


def _on_domain(url: str, doms) -> bool:
    h = _host(url)
    return any(h == d or h.endswith("." + d) or (("/" in d) and d in url) for d in doms)


def _age_days(pub) -> float | None:
    if not pub:
        return None
    s = str(pub)
    for fmt in ("%a, %d %b %Y %H:%M:%S %Z", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            d = dt.datetime.strptime(s if "%Z" in fmt else s[:len(dt.datetime(2000, 1, 1).strftime(fmt))], fmt).date()
            return (today() - d).days
        except ValueError:
            continue
    return None


def doc_score(f, d, rank: int) -> float:
    """Read-out score for one result (higher is better). Base = its position in the lane's order."""
    s = 1.0 / (1 + rank)
    text = f"{d.title} {d.text[:600]}".lower()
    phrases = [p for p in f.phrases if p]
    if phrases:
        hit = sum(1 for p in phrases if p in text)
        s += config.NARROW_W_ENTITY * hit / len(phrases)
        if hit == 0:
            s -= config.NARROW_W_ENTITY          # names none of the entities the question is about
    cw = [w for w in f.content if len(w) > 2]
    if cw:
        s += config.NARROW_W_COVER * sum(1 for w in cw if w in text) / len(cw)
    fam = family(f)
    if fam in FAMILY_DOMAINS and _on_domain(d.url, FAMILY_DOMAINS[fam]):
        s += config.NARROW_W_DOMAIN
    if f.when or fam in ("news", "sports", "finance", "weather"):
        age = _age_days((d.meta or {}).get("published_date"))
        if age is not None:
            s += config.NARROW_W_FRESH * (1.0 if age <= 2 else 0.5 if age <= 7 else 0.0 if age <= 60 else -0.5)
    return s


def rerank(f, docs: list) -> list:
    """docs reordered by doc_score (stable for ties)."""
    if not config.NARROW_RERANK or not docs:
        return list(docs)
    sc = [(doc_score(f, d, i), -i) for i, d in enumerate(docs)]
    order = sorted(range(len(docs)), key=lambda i: sc[i], reverse=True)
    return [docs[i] for i in order]
