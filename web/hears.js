/* "What Arc hears": a live, word-by-word annotated transcript that makes Arc's understanding visible as it happens.
   Driven by the same server events as the rest of the page (app.js forwards each one to HSHears.event).

   Word layer   words as heard; revisions replace the old word (struck, fading); stop words dim; content words (what goes
                into the search query) bold; typed spans underlined with their type + a normalised value ("five day" →
                5 days, "tomorrow" → Sat, Nov 14); a self-correction strikes the corrected-away words and marks the
                replacement; private data shows ●●● with a lock; a carried referent shows a ghost chip ("it → Nvidia");
                ear word confidence shows as a wavy underline on uncertain words; long pauses open a little space.
   Pins         on the word where it happened: intent first confident, dialogue act, command complete (Arc Tagger
                and Arc audio), and under the word where each speculative search fired (🔎, dim when stale, bright
                when it fed the answer).
   Tail         after the last word: Flux EagerEndOfTurn / TurnResumed / EndOfTurn ticks, the endpoint, and when the
                standard pipeline starts searching, each with its time after the last word.
   Chips        one row under the words: intent, act, search family / freshness, route, privacy, safety, Arc audio
                (to the assistant, speaker / role, audio events), memory. */
(() => {
  "use strict";
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const human = (s) => String(s || "").replace(/^discourse\.intent\./, "").replace(/^intent\./, "").replace(/_/g, " ");
  const secs = (ms) => (ms < 0 ? "−" : "+") + (Math.abs(ms) / 1000).toFixed(2) + " s";
  const SLOT_SHORT = { organization: "org", organisation: "org", location: "place", place: "place", city: "place", country: "place",
    person: "person", date: "date", time: "time", time_of_day: "time", cardinal: "number", ordinal: "ordinal", number: "number",
    numeral: "number", money: "money", percent: "percent", quantity: "quantity", year: "year", event_name: "event", event: "event",
    product: "product", work_of_art: "title", duration: "duration", team: "team", age: "age" };
  const slotLabel = (t) => SLOT_SHORT[String(t).toLowerCase()] || String(t).replace(/_/g, " ");
  const isNumType = (t) => /^(cardinal|ordinal|money|percent|date|time|quantity|numeral|year|number|duration|age)/.test(String(t));

  // ---------------------------------------------------------------- normalised values (client side)
  const ONES = { zero: 0, one: 1, a: 1, an: 1, two: 2, three: 3, four: 4, five: 5, six: 6, seven: 7, eight: 8, nine: 9, ten: 10,
    eleven: 11, twelve: 12, thirteen: 13, fourteen: 14, fifteen: 15, sixteen: 16, seventeen: 17, eighteen: 18, nineteen: 19 };
  const TENS = { twenty: 20, thirty: 30, forty: 40, fifty: 50, sixty: 60, seventy: 70, eighty: 80, ninety: 90 };
  const ORD = { first: 1, second: 2, third: 3, fourth: 4, fifth: 5, sixth: 6, seventh: 7, eighth: 8, ninth: 9, tenth: 10 };
  const DAYS = ["sunday", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday"];
  // "five" → 5, "twenty one" → 21, "two hundred" → 200, "1.5" → 1.5; null if not a number phrase
  function wordsToNum(toks) {
    let total = 0, cur = 0, any = false;
    for (const t of toks) {
      if (/^\d+(\.\d+)?$/.test(t.replace(/,/g, ""))) { cur += parseFloat(t.replace(/,/g, "")); any = true; }
      else if (t in ONES && !(t === "a" || t === "an") ) { cur += ONES[t]; any = true; }
      else if ((t === "a" || t === "an") && !any) { cur += 1; any = true; }
      else if (t in TENS) { cur += TENS[t]; any = true; }
      else if (t === "hundred") { cur = (cur || 1) * 100; any = true; }
      else if (t === "thousand") { total += (cur || 1) * 1000; cur = 0; any = true; }
      else if (t === "million") { total += (cur || 1) * 1e6; cur = 0; any = true; }
      else if (t === "and" && any) continue;
      else return null;
    }
    return any ? total + cur : null;
  }
  const fmtDay = (d) => d.toLocaleDateString(undefined, { weekday: "short", month: "short", day: "numeric" });
  const addDays = (d, n) => { const x = new Date(d); x.setDate(x.getDate() + n); return x; };
  function normValue(type, text, before = "") {
    const t = String(type).split(".").pop().toLowerCase();
    const s = String(text || "").toLowerCase().replace(/[^\w$%.' -]/g, "").trim();
    const toks = s.split(/[\s-]+/).filter(Boolean);
    const today = new Date();
    if (t === "date" || t === "time_of_day" || t === "time") {
      if (/\btoday\b|\bright now\b|\bnow\b|\bcurrently\b/.test(s)) return fmtDay(today);
      if (/\btonight\b|\bthis evening\b/.test(s)) return fmtDay(today) + ", evening";
      if (/\btomorrow\b/.test(s)) return fmtDay(addDays(today, 1));
      if (/\byesterday\b/.test(s)) return fmtDay(addDays(today, -1));
      if (/\blast night\b/.test(s)) return fmtDay(addDays(today, -1)) + ", night";
      if (/\bweekend\b/.test(s)) {
        const past = /\b(past|last)\b/.test(before + " " + s);
        const sat = past ? addDays(today, -(((today.getDay() - 6 + 7) % 7) || 7)) : addDays(today, (6 - today.getDay() + 7) % 7);
        return `${sat.toLocaleDateString(undefined, { month: "short", day: "numeric" })}–${addDays(sat, 1).toLocaleDateString(undefined, { day: "numeric" })}`;
      }
      if (/\bnext week\b/.test(s)) return "week of " + addDays(today, 7 - today.getDay() + 1).toLocaleDateString(undefined, { month: "short", day: "numeric" });
      if (/\bthis week\b|\bpast week\b|\blast week\b/.test(s)) return s.includes("this") ? "this week" : "past 7 days";
      const dn = DAYS.findIndex((d) => s.includes(d));
      if (dn >= 0) {
        const back = /\blast\b|\bon\b.*\bgame\b/.test(s);
        const delta = back ? -((today.getDay() - dn + 7) % 7 || 7) : ((dn - today.getDay() + 7) % 7 || 7);
        return fmtDay(addDays(today, delta));
      }
      if (/\bnext month\b/.test(s)) { const d = new Date(today.getFullYear(), today.getMonth() + 1, 1); return d.toLocaleDateString(undefined, { month: "long", year: "numeric" }); }
      if (/\bthis month\b/.test(s)) return today.toLocaleDateString(undefined, { month: "long", year: "numeric" });
      if (/\bnext year\b/.test(s)) return String(today.getFullYear() + 1);
      if (/\blast year\b/.test(s)) return String(today.getFullYear() - 1);
      const y = s.match(/\b(1[89]\d\d|20\d\d)\b/);
      if (y) return y[1];
      return "";
    }
    if (t === "duration") {
      const m = s.match(/^(.*?)\s*(second|minute|hour|day|night|week|month|year)s?\b/);
      if (m) {
        const n = m[1] ? wordsToNum(m[1].split(/\s+/).filter(Boolean)) : 1;
        if (n != null) return `${n} ${m[2]}${n === 1 ? "" : "s"}`;
      }
      return "";
    }
    if (t === "money") {
      const m = s.match(/^(.*?)\s*(dollars?|bucks|euros?|pounds?|yen|cents?)$/);
      const n = m ? wordsToNum(m[1].replace(/\$/g, "").split(/\s+/).filter(Boolean)) : null;
      if (n != null) return ({ d: "$", b: "$", e: "€", p: "£", y: "¥", c: "¢" }[m[2][0]] || "") + n.toLocaleString();
      return "";
    }
    if (t === "percent") { const n = wordsToNum(toks.filter((x) => x !== "percent" && x !== "%")); return n != null ? n + "%" : ""; }
    if (t === "ordinal") { const o = toks.map((x) => ORD[x]).find((x) => x); return o ? o + (o === 1 ? "st" : o === 2 ? "nd" : o === 3 ? "rd" : "th") : ""; }
    if (t === "cardinal" || t === "number" || t === "numeral" || t === "quantity" || t === "age" || t === "year") {
      const n = wordsToNum(toks);
      return n != null && String(n) !== s ? n.toLocaleString() : "";
    }
    return "";
  }

  // ---------------------------------------------------------------- state
  let S = null, box = null, raf = 0;
  function fresh() {
    return { words: [], wordsMode: null, wordsSig: "", frame: null, spans: [], revised: {}, intentPins: [], actPin: null,
      ccTagger: null, ccAudio: [], fires: new Map(), used: new Set(), arc: null, ear: [], endpoint: null, speechEnd: null,
      baselineSearch: null, family: null, lastWordEnd: null, done: false };
  }

  function mount() {
    const tr = document.getElementById("transcript");
    if (!tr) return false;
    tr.classList.add("hears-host");
    tr.innerHTML = `<div class="hears">
      <div class="hw-line" id="hwLine"><span class="hw-ph">Pick a question, or hold <kbd>Space</kbd> and ask; release when you are done.</span></div>
      <div class="hw-chips" id="hwChips"></div></div>`;
    box = tr;
    return true;
  }

  function start() {
    S = fresh();
    mount();
  }
  function newQuestion() { start(); }

  // ---------------------------------------------------------------- events
  function event(e) {
    if (!S) start();
    if (e.mode === "sys") { if (e.t === "run_done") { S.done = true; schedule(); } return; }
    if (e.mode === "ear") {
      if (["EagerEndOfTurn", "TurnResumed", "EndOfTurn", "StartOfTurn"].includes(e.kind)) {
        S.ear.push({ kind: e.kind, at: e.at, wordEnd: e.word_end, p: e.eot_p });
        schedule();
      } else if (e.kind === "arc_command_complete" && e.t_audio != null) { S.ccAudio.push({ t: e.t_audio, p: e.p }); schedule(); }
      return;
    }
    const m = e.mode;
    switch (e.t) {
      case "words": {
        const sig = (e.words || []).map((w) => w.w).join(" ");
        // both sessions send the same words; follow Head Start until it has endpointed, then whoever is still listening
        if (m !== "headstart" && S.wordsMode === "headstart" && !(S.endpoint && S.endpoint.mode === "headstart")) return;
        if (sig === S.wordsSig) return;
        // a new question (live mic / conversation): words that start after the last answered speech end
        const t0 = e.words && e.words.length ? e.words[0].t0 : null;
        if (S.speechEnd != null && t0 != null && t0 * 1000 > S.speechEnd + 50) { S = fresh(); }
        const old = S.words;
        const nw = e.words || [];
        nw.forEach((w, i) => { if (old[i] && old[i].w !== w.w) S.revised[i] = { was: old[i].w, at: performance.now() }; });
        S.wordsSig = sig; S.wordsMode = m; S.words = nw;
        schedule();
        break;
      }
      case "frame": {
        if (S.wordsMode && m !== S.wordsMode) return;
        S.frame = e;
        S.spans = e.spans || [];
        const n = e.n || S.words.length;
        if (e.intent && (+e.p || 0) >= 0.5) {
          const lastPin = S.intentPins.at(-1);
          if (!lastPin || lastPin.intent !== e.intent) S.intentPins.push({ intent: e.intent, p: +e.p, word: n - 1 });
          else lastPin.pNow = +e.p;
          if (S.intentPins.length > 2) S.intentPins.shift();
        }
        const acts = (e.readout && e.readout.acts) || [];
        if (acts.length && !S.actPin) S.actPin = { act: acts[0], word: n - 1 };
        if (e.cc_word != null) S.ccTagger = e.cc_word;
        if (e.arc) S.arc = e.arc;
        schedule();
        break;
      }
      case "arc":
        S.arc = e;
        schedule();
        break;
      case "spec": {
        if (e.lane === "budget") return;
        const k = m + ":" + e.id;
        let f = S.fires.get(k);
        if (!f) {
          f = { id: e.id, mode: m, lane: e.lane, query: e.query, state: "fired", ms: e.ms, word: e.word };
          S.fires.set(k, f);
        }
        if (e.state !== "fired") f.state = f.state === "stale" && e.state === "done" && !e.revived ? "stale" : e.state;
        if (e.word != null) f.word = e.word;
        if (e.narrow) S.family = e.narrow;
        if (e.topic && e.topic !== "general") S.topic = e.topic + (e.time_range ? ` · past ${e.time_range}` : "");
        if (m === "baseline" && !S.baselineSearch && e.state === "fired") S.baselineSearch = e.ms;
        schedule();
        break;
      }
      case "used":
        for (const id of e.fires || []) S.used.add(m + ":" + id);
        schedule();
        break;
      case "endpoint":
        if (!S.endpoint || m === "headstart") S.endpoint = { mode: m, ms: e.ms, by: e.by };
        if (e.speech_end_ms != null) S.speechEnd = Math.max(S.speechEnd ?? 0, e.speech_end_ms);
        if (e.frame && (!S.wordsMode || m === S.wordsMode)) S.frame = { ...S.frame, ...e.frame, spans: S.frame ? S.frame.spans : [] };
        schedule();
        break;
      case "extended":
        S.endpoint = null;
        schedule();
        break;
    }
  }

  function schedule() {
    if (raf) return;
    raf = requestAnimationFrame(() => { raf = 0; render(); });
  }

  // ---------------------------------------------------------------- render
  function wordAtTime(t) {
    // the word an audio time belongs to: the last word that ended before it (+50 ms)
    let k = -1;
    S.words.forEach((w, i) => { if (w.t1 != null && w.t1 <= t + 0.05) k = i; });
    return k < 0 ? S.words.length - 1 : k;
  }

  function render() {
    if (!box || !document.getElementById("hwLine")) mount();
    const line = document.getElementById("hwLine");
    const W = S.words;
    if (!W.length) { renderChips(); return; }
    const f = S.frame || {};
    const n = W.length;
    const content = new Set((f.content_idx || []).filter((i) => i < n));
    const dropped = new Set(f.dropped || []), cue = new Set(f.cue || []), redact = new Set(f.redact || []);
    const haveFrame = !!S.frame && Array.isArray(f.content_idx);
    // typed spans → per word
    const spanAt = new Array(n).fill(null);
    for (const s of S.spans || []) {
      const ty = String(s.type || "");
      const fam = ty.startsWith("entity.number") || ty.startsWith("number") ? "num" : ty.startsWith("entity") ? "ent" : ty.startsWith("safety.pii") ? "pii" : null;
      if (!fam || ty === "entity.numeral") continue;
      for (let i = s.start; i < s.end && i < n; i++) spanAt[i] = { s, fam, first: i === s.start, last: i === s.end - 1 };
    }
    // the replacement after a self-correction: the first kept content word after the cue
    const cueEnd = cue.size ? Math.max(...cue) : -1;
    const repl = cueEnd >= 0 ? [...content].filter((i) => i > cueEnd).sort((a, b) => a - b)[0] : null;
    const corrWas = cueEnd >= 0 ? W.filter((_, i) => dropped.has(i) && !cue.has(i)).map((w) => w.w).join(" ") : "";
    // carried referent: the pronoun it hangs on
    const PRON = new Set(["it", "its", "he", "she", "his", "her", "him", "they", "them", "their", "there", "that", "this"]);
    const pronAt = f.carried ? W.findIndex((w) => PRON.has(String(w.w).toLowerCase().replace(/[^a-z]/g, ""))) : -1;
    // pins on words
    const top = Array.from({ length: n }, () => []);
    for (const p of S.intentPins) if (p.word >= 0 && p.word < n)
      top[p.word].push(`<span class="pin intent" title="Arc Tagger: intent first confident at this word (p ${p.p.toFixed(2)}${p.pNow != null ? `, now ${p.pNow.toFixed(2)}` : ""})">${esc(human(p.intent))} <b>${(p.pNow ?? p.p).toFixed(2)}</b></span>`);
    if (S.actPin && S.actPin.word < n) top[S.actPin.word].push(`<span class="pin act" title="Arc Tagger: dialogue act">${esc(human(S.actPin.act))}</span>`);
    const ccA = new Set(S.ccAudio.map((c) => wordAtTime(c.t)));
    for (const c of (S.arc && S.arc.cc_events) || []) if (c.t != null) ccA.add(wordAtTime(c.t));
    const ccT = S.ccTagger != null && S.ccTagger < n ? S.ccTagger : null;
    const ccWords = new Set([...ccA, ...(ccT != null ? [ccT] : [])]);
    for (const i of ccWords) {
      if (i < 0 || i >= n) continue;
      const who = [ccT === i ? "tagger" : null, ccA.has(i) ? "audio" : null].filter(Boolean);
      top[i].push(`<span class="pin cc" title="command complete: Arc ${who.join(" + Arc ")} says the request is complete at this word">✓ complete <small>${who.join("+")}</small></span>`);
    }
    // search pins under words (Head Start's speculative searches, at the word they fired on)
    const bot = Array.from({ length: n }, () => []);
    for (const [k, fi] of S.fires) {
      if (fi.mode !== "headstart" || fi.lane.startsWith("final") || fi.word == null) continue;
      const i = Math.min(n - 1, Math.max(0, fi.word - 1));
      const used = S.used.has(k);
      const st = used ? "used" : fi.state;
      const ic = fi.lane === "draft" ? "✎" : fi.lane === "audio" ? "♪" : fi.lane === "local" ? "◆" : "🔎";
      const q = fi.lane === "draft" ? "draft answer" : fi.lane === "audio" ? "audio search" : fi.query;
      bot[i].push(`<span class="spin s-${esc(st)}" title="${esc(fi.lane)} search fired at this word: “${esc(fi.query)}” · ${esc(st)}">${ic}<i>${esc(String(q).length > 22 ? String(q).slice(0, 21) + "…" : q)}</i></span>`);
    }
    // words
    let html = "";
    W.forEach((w, i) => {
      const sp = spanAt[i];
      const cls = ["hw"];
      let txt = esc(w.w), under = "", pre = "";
      if (redact.has(i) || (sp && sp.fam === "pii")) {
        cls.push("redact");
        txt = `<span class="lock">🔒</span>●●●`;
        if (!sp || sp.first) under = `<span class="ul pii">private · kept on device</span>`;
      } else if (dropped.has(i) && !cue.has(i)) cls.push("dropped");
      else if (cue.has(i)) cls.push("cue");
      else if (sp) {
        cls.push("slot", sp.fam, sp.first ? "s-first" : "", sp.last ? "s-last" : "");
        if (sp.first) {
          const val = normValue(sp.s.type, sp.s.text, W.slice(Math.max(0, i - 2), i).map((x) => x.w).join(" ").toLowerCase());
          under = `<span class="ul ${sp.fam}">${esc(slotLabel(String(sp.s.type).split(".").pop()))}${val ? ` <b>${esc(val)}</b>` : ""}</span>`;
        }
      }
      if (haveFrame && !redact.has(i) && !dropped.has(i) && !cue.has(i)) cls.push(content.has(i) ? "content" : "stop");
      if (i === repl) {
        cls.push("repl");
        pre = `<span class="arrow" title="replaces “${esc(corrWas)}”">→</span>`;
        under = under ? under.replace(/<\/span>$/, ` <i class="rp">replaces ${esc(corrWas)}</i></span>`) : `<span class="ul corr">replaces ${esc(corrWas)}</span>`;
      }
      if (i === pronAt) under = `<span class="ul mem" title="conversation memory: resolved from the previous turn">${esc(w.w)} → ${esc(f.carried)}</span>`;
      const conf = w.conf != null ? +w.conf : null;
      if (conf != null && conf < 0.85) cls.push("unsure");
      // a pause before this word (from word timestamps) opens a little space
      const prev = W[i - 1];
      const gap = prev && prev.t1 != null && w.t0 != null ? w.t0 - prev.t1 : 0;
      const style = [conf != null ? `--conf:${Math.max(0, Math.min(1, (0.85 - conf) / 0.35)).toFixed(2)}` : "", gap > 0.25 ? `margin-left:${Math.min(34, 8 + gap * 30).toFixed(0)}px` : ""].filter(Boolean).join(";");
      const rev = S.revised[i] && performance.now() - S.revised[i].at < 1800 ? `<span class="was">${esc(S.revised[i].was)}</span>` : "";
      if (rev) cls.push("rev");
      const title = [conf != null ? `confidence ${conf.toFixed(2)}` : "", gap > 0.25 ? `pause ${gap.toFixed(2)} s before` : "", w.t0 != null ? `${w.t0.toFixed(2)}–${(w.t1 ?? w.t0).toFixed(2)} s` : ""].filter(Boolean).join(" · ");
      html += `<span class="${cls.filter(Boolean).join(" ")}" data-i="${i}"${style ? ` style="${style}"` : ""}${title ? ` title="${esc(title)}"` : ""}>` +
        `<span class="hw-top">${top[i].join("")}</span><span class="hw-x">${pre}${txt}${rev}</span>` +
        `<span class="hw-ul">${under}</span><span class="hw-bot">${bot[i].join("")}</span></span>`;
    });
    html += renderTail();
    if (line._h !== html) {
      line._h = html;
      // keep word nodes stable where possible (entering animation only for new words)
      const before = line.querySelectorAll(".hw").length;
      line.innerHTML = html;
      line.querySelectorAll(".hw").forEach((el, i) => { if (i >= before) el.classList.add("in"); });
      declutter(line);
    }
    renderChips();
    // revision marks fade: re-render once they expire
    if (Object.values(S.revised).some((r) => performance.now() - r.at < 1800)) setTimeout(schedule, 1850);
  }

  // after the last word: when each end-of-turn signal arrived, relative to the end of the last word
  function renderTail() {
    const W = S.words;
    const lastEnd = W.length && W.at(-1).t1 != null ? W.at(-1).t1 : null;
    const ticks = [];
    for (const e of S.ear) {
      if (e.kind === "StartOfTurn") continue;
      const ref = e.wordEnd ?? lastEnd;
      const dt = e.at != null && ref != null ? (e.at - ref) * 1000 : null;
      const lab = { EagerEndOfTurn: "eager end", TurnResumed: "resumed", EndOfTurn: "end of turn" }[e.kind];
      ticks.push({ cls: "flux " + e.kind, lab, dt, title: `Deepgram Flux ${e.kind}${e.p != null ? ` (p ${(+e.p).toFixed(2)})` : ""}` });
    }
    if (S.endpoint && S.speechEnd != null) {
      const dt = S.endpoint.ms - S.speechEnd;
      ticks.push({ cls: "ep", lab: S.endpoint.by === "eager" || S.endpoint.by === "eager+arc" ? "answers (eager)" : S.endpoint.by === "semantic" || S.endpoint.by === "tagger" ? "answers (semantic)" : "endpoint", dt, title: "Head Start's endpoint: the answer starts from here" });
    }
    if (S.baselineSearch != null && S.speechEnd != null)
      ticks.push({ cls: "bl", lab: "standard starts searching", dt: S.baselineSearch - S.speechEnd, title: "the standard pipeline's first search, after the end of the question" });
    if (!ticks.length) return "";
    ticks.sort((a, b) => (a.dt ?? 1e9) - (b.dt ?? 1e9));
    return `<span class="hw-tail">${ticks.map((t) => `<span class="tk ${t.cls}" title="${esc(t.title)}"><i></i>${esc(t.lab)}${t.dt != null ? ` <b>${secs(t.dt)}</b>` : ""}</span>`).join("")}</span>`;
  }

  // pins sharing a line can collide: lift a pin to a second row when it overlaps the previous one
  function declutter(line) {
    // layout positions (offset*, unaffected by the pop-in transform) relative to the line
    const pos = (e) => { const c = e.closest(".hw") || e.closest(".hw-tail"); return { x: (c ? c.offsetLeft : 0) + e.offsetLeft, w: e.offsetWidth, y: c ? c.offsetTop : 0 }; };
    for (const cls of [".hw-top .pin", ".hw-bot .spin", ".hw-ul .ul"]) {
      const els = Array.from(line.querySelectorAll(cls));
      els.forEach((e) => e.classList.remove("lift"));
      let prev = null;
      for (const e of els) {
        const r = pos(e);
        if (prev && Math.abs(r.y - prev.y) < 6 && r.x < prev.x + prev.w + 4) { e.classList.add("lift"); continue; }
        prev = r;
      }
    }
  }

  function chip(cls, k, v, title) { return `<span class="hc c-${cls}"${title ? ` title="${esc(title)}"` : ""}>${k ? `<small>${esc(k)}</small>` : ""}${v != null ? esc(v) : ""}</span>`; }
  function renderChips() {
    const el = document.getElementById("hwChips");
    if (!el) return;
    const f = S.frame || {}, r = f.readout || {}, a = S.arc || {};
    const out = [];
    const it = S.intentPins.at(-1);
    if (f.intent) out.push(chip("intent", "intent", `${human(f.intent)} ${(+f.p || 0).toFixed(2)}`, "Arc Tagger intent (confidence)"));
    else if (it) out.push(chip("intent", "intent", `${human(it.intent)} ${(it.pNow ?? it.p).toFixed(2)}`, "last confident intent"));
    if (r.acts && r.acts.length) out.push(chip("act", "act", r.acts.map(human).join(", "), "Arc Tagger dialogue act"));
    if (S.family) out.push(chip("fam", "search", S.family, "search family (narrowing: query shape + preferred sources)"));
    const fresh = r.fresh || S.topic;
    if (fresh) out.push(chip("fresh", "fresh", fresh + (r.when_text ? ` (“${r.when_text}”)` : ""), "freshness chosen from what was said"));
    if (f.route) out.push(chip("route", "route", f.route, "web or the local moments index"));
    if (f.carried) out.push(chip("mem", "memory", `→ ${f.carried}`, "resolved from the previous turn"));
    if (r.redacted && r.redacted.length) out.push(chip("priv", "🔒", r.redacted.join(", ") + " kept on device", "personal data never goes into the web query"));
    if (r.jailbreak) out.push(chip("warn", "⚠", "prompt injection: no early search", "safety.jailbreak.attempt"));
    if (a.for_me != null) out.push(chip(a.for_me >= 0.5 ? "arc" : "warn", "Arc audio", a.for_me >= 0.5 ? `to the assistant ${a.for_me.toFixed(2)}` : `not to the assistant ${a.for_me.toFixed(2)}`, "Arc audio head: device-directed"));
    else if (r.directed != null) out.push(chip(r.directed >= 0.5 ? "arc" : "warn", "for me", r.directed.toFixed(2), "device-directed (Arc Tagger)"));
    const tail = (x) => human(String(x || "").split(".").pop());
    if (a.speaker || a.role) out.push(chip("arc", "speaker", [a.speaker, a.role ? tail(a.role) : ""].filter(Boolean).join(" · "), "Arc audio: speaker id + role"));
    if (a.act && a.act.label && !(r.acts && r.acts.length)) out.push(chip("arc", "act (audio)", human(a.act.label)));
    for (const ev of (a.audio_events || []).slice(0, 2)) if (ev.tag) out.push(chip("arc", "sound", tail(ev.tag)));
    const bios = [...new Set((a.bio || []).map((b) => b.tag).filter(Boolean).map(tail))].slice(0, 2);
    if (bios.length) out.push(chip("arc", "voice", bios.join(", "), "Arc audio: non-speech vocal events"));
    const h = out.join("");
    if (el._h !== h) { el._h = h; el.innerHTML = h; }
  }

  window.HSHears = { start, newQuestion, event, normValue, state: () => S };
})();
