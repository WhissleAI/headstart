/* Film mode: a clean 1920x1080 layout for recording the race (/?film=1, or press F).
   Driven by the same server events as the main UI (app.js forwards every event and run start to HSFilm).
   URL params: film=1 · q=<file.wav> auto-replays after delay=<ms> (default 1500) · focus=timeline|readout|watches
   (b-roll close-ups) · theme=light (default dark). window.__film.state() exposes timings for film/capture.py. */
(() => {
  "use strict";
  const P = new URLSearchParams(location.search);
  const want = P.get("film") === "1" || P.has("focus");
  const $ = (s, r = document) => r.querySelector(s);
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const MODES = ["baseline", "headstart"];          // left, right
  const LABEL = { baseline: "Standard voice search — transcribe, then search", headstart: "Arc — searches while you speak" };
  const SHORT = { baseline: "Standard", headstart: "Arc" };
  const fmt = (ms) => (ms == null || !isFinite(ms) ? "—" : (ms < 0 ? "−" : "") + (Math.abs(ms) / 1000).toFixed(2) + " s");
  const fmtW = (ms) => (ms == null || !isFinite(ms) ? "—" : (ms < 0 ? "−" : "") + (Math.abs(ms) / 1000).toFixed(2) + "<small>s</small>");
  const fmt1 = (ms) => (Math.abs(ms) / 1000).toFixed(1) + " s";

  let root = null, S = null, raf = 0;

  function fresh() {
    return {
      started: false, startPerf: null, base: null, words: [], wordsMode: null, wordsSig: "", spans: [], frame: null,
      lastIntent: null, lastSlots: null, fires: new Map(), order: [], se: {}, speechEnd: null, endpointMs: {},
      sourcesMs: {}, sources: {}, answer: {}, firstMs: {}, metric: {}, error: {}, done: false, events: [], axEnd: 3000,
    };
  }

  // ---------------------------------------------------------------- DOM
  function build() {
    root = document.createElement("div");
    root.id = "film";
    root.innerHTML = `
      <div class="f-top">
        <div class="f-brand"><span class="f-logo"><i></i><i></i><i></i></span><b>Fast Voice Search</b><span class="f-sep">·</span><span>live web search by Tavily</span></div>
        <div class="f-note">Same recognizer · same web search · same answer model. Only the start time differs.</div>
      </div>
      <section class="f-q">
        <div class="f-qlab"><span class="f-ear"></span><span class="f-qstate">Listening</span></div>
        <div class="f-words" id="fWords"><span class="f-ph">Ask a question…</span></div>
      </section>
      <section class="f-read">
        <div class="f-rlab">Arc reads the question as you speak</div>
        <div class="f-rbody" id="fRead"><span class="f-muted">waiting for the first words…</span></div>
      </section>
      <section class="f-panes">
        ${MODES.map((m) => `
        <div class="f-pane ${m}" id="fp-${m}">
          <div class="f-ph2"><i class="f-dot"></i><span>${esc(LABEL[m])}</span></div>
          <div class="f-head">
            <div class="f-hl"><div class="f-status" data-k="status"></div><div class="f-delta" data-k="delta"></div></div>
            <div class="f-watch"><div class="f-big" data-k="big"></div><div class="f-wsub" data-k="wsub">from when you stop talking</div></div>
          </div>
          <div class="f-ans" data-k="ans"></div>
          <div class="f-rq" data-k="rq"></div>
          <div class="f-res" data-k="res"></div>
        </div>`).join("")}
      </section>
      <section class="f-tl">
        <div class="f-tlaxis"><div class="f-tll"></div><div class="f-tlt" id="fAxis"></div></div>
        <div class="f-tlrow you"><div class="f-tll">You</div><div class="f-tlt" id="fTlYou"></div></div>
        <div class="f-tlrow baseline"><div class="f-tll"><i class="f-dot"></i>Standard</div><div class="f-tlt" id="fTl-baseline"></div></div>
        <div class="f-tlrow headstart"><div class="f-tll"><i class="f-dot"></i>Arc</div><div class="f-tlt" id="fTl-headstart"></div></div>
        <div class="f-tlmarks"><div class="f-tll"></div><div class="f-tlt" id="fMarks"></div></div>
      </section>`;
    document.body.appendChild(root);
  }

  function activate() {
    if (!root) build();
    document.body.classList.add("film");
    const f = P.get("focus");
    if (f) document.body.dataset.focus = f;
    document.documentElement.dataset.theme = P.get("theme") || "dark";
    api.active = true;
    reset();
    measure();
  }
  function deactivate() { document.body.classList.remove("film"); api.active = false; }

  function reset() {
    S = fresh();
    if (!root) return;
    $("#fWords").innerHTML = '<span class="f-ph">Ask a question…</span>';
    $("#fRead").innerHTML = '<span class="f-muted">waiting for the first words…</span>';
    root.querySelector(".f-qstate").textContent = "Listening";
    root.classList.remove("hearing", "heard");
    for (const m of MODES) {
      const p = $("#fp-" + m);
      p.classList.remove("winner", "loser", "done");
      for (const k of ["delta", "status", "ans", "rq", "res"]) p.querySelector(`[data-k="${k}"]`).innerHTML = "";
      p.querySelector('[data-k="res"]')._sig = null;
      results(m);
      p.querySelector('[data-k="big"]').innerHTML = fmtW(0);
      p.querySelector('[data-k="big"]').className = "f-big idle";
      p.querySelector('[data-k="wsub"]').textContent = "from when you stop talking";
      status(m);
    }
    for (const id of ["fAxis", "fTlYou", "fTl-baseline", "fTl-headstart", "fMarks"]) $("#" + id).innerHTML = "";
  }

  // ---------------------------------------------------------------- clock
  const now = () => (S && S.base != null ? performance.now() - S.base : 0);
  function sync(ms) {
    if (typeof ms !== "number") return;
    const c = performance.now() - ms;
    S.base = S.base == null ? c : Math.min(S.base, c);
  }
  // server ms 0 = audio sample 0, as epoch ms (the capture script aligns the question audio with it)
  const audio0Epoch = () => (S && S.base != null ? Date.now() - (performance.now() - S.base) : null);

  // ---------------------------------------------------------------- events
  function start() {
    if (!api.active) return;
    reset();
    S.started = true;
    S.startPerf = performance.now();
    loop();
  }

  function event(e) {
    if (!api.active || !S || !S.started) return;
    S.events.push({ ...e, _rx: Math.round(performance.now() - S.startPerf), docs: undefined, words: e.words });
    if (e.mode === "sys") {
      if (e.t === "run_done") { S.done = true; settle(); }
      return;
    }
    if (typeof e.ms === "number") sync(e.ms);
    if (e.mode === "ear") return;
    const m = e.mode;
    switch (e.t) {
      case "words": {
        const sig = (e.words || []).map((w) => w.w).join(" ");
        // both sessions send the same words; follow Arc until it has endpointed, then whoever is still listening
        if (m !== "headstart" && S.wordsMode === "headstart" && S.endpointMs.headstart == null) return;
        if (sig === S.wordsSig) return;
        S.wordsSig = sig; S.wordsMode = m;
        S.words = e.words || [];
        renderWords();
        root.classList.add("hearing");
        break;
      }
      case "frame":
        if (m !== S.wordsMode && S.wordsMode) return;
        S.frame = e;
        S.spans = e.spans || [];
        if (e.corrected && Array.isArray(e.slots)) {
          const keep = new Set(e.slots.map((x) => String(x[1]).toLowerCase()));
          S.spans = S.spans.map((sp) => (/^entity|^number/.test(sp.type) && !keep.has(String(sp.text).toLowerCase().replace(/[^\w' -]/g, "").trim()) ? { ...sp, dropped: true } : sp));
        }
        if (e.intent) S.lastIntent = e;
        if (e.slots && e.slots.length) S.lastSlots = e.slots;
        renderSpans();
        renderRead();
        break;
      case "endpoint":
        S.endpointMs[m] = e.ms;
        if (e.speech_end_ms != null) { S.se[m] = e.speech_end_ms; S.speechEnd = Math.max(S.speechEnd ?? 0, e.speech_end_ms); }
        root.classList.add("heard");
        root.querySelector(".f-qstate").textContent = "Heard";
        if (e.frame) { S.frame = { ...S.frame, ...e.frame, final: true }; renderRead(); }
        status(m);
        break;
      case "spec": {
        const k = m + ":" + e.id;
        let f = S.fires.get(k);
        if (!f) { f = { k, id: e.id, mode: m, lane: e.lane, query: e.query, state: "fired", t0: e.ms, t1: null }; S.fires.set(k, f); S.order.push(k); }
        if (e.state === "fired") f.t0 = e.ms;
        else if (e.state === "stale") { if (f.state === "fired") f.t1 = e.ms; f.state = "stale"; }
        else if (e.state === "reused") { f.state = "reused"; f.t1 = e.ms; }
        else { if (f.state !== "stale") f.state = e.state; if (f.t1 == null) f.t1 = e.ms; }
        if (e.state === "done" || e.revived) f.docs = specDocs(e) || f.docs;
        status(m);
        results(m);
        break;
      }
      case "sources":
        S.sourcesMs[m] = e.ms; S.sources[m] = e.docs || [];
        status(m);
        results(m);
        break;
      case "used": {
        const ids = new Set(e.fires || []);
        for (const f of S.fires.values()) if (f.mode === m) f.used = ids.has(f.id);
        break;
      }
      case "answer":
        if (S.firstMs[m] == null) S.firstMs[m] = e.ms;
        S.answer[m] = (S.answer[m] || "") + (e.delta || "");
        renderAnswer(m);
        status(m);
        break;
      case "answer_error":
        S.error[m] = e.error || "answer failed";
        renderAnswer(m);
        break;
      case "extended":
        // the speaker kept going after an early endpoint: that answer is dropped, a new one follows
        for (const k of ["endpointMs", "sourcesMs", "sources", "answer", "firstMs", "metric", "error", "se"]) delete S[k][m];
        S.speechEnd = Object.keys(S.se).length ? Math.max(...Object.values(S.se)) : null;
        renderAnswer(m);
        status(m);
        break;
      case "metric":
        S.metric[m] = e;
        if (S.se[m] == null && S.endpointMs[m] != null) { S.se[m] = S.endpointMs[m] - (e.endpoint_lag_ms || 0); S.speechEnd = Math.max(S.speechEnd ?? 0, S.se[m]); }
        for (const f of S.fires.values()) if (f.mode === m && f.lane === "draft" && f.state === "fired") { f.state = "unused"; f.t1 = S.endpointMs[m]; }
        renderAnswer(m);
        status(m);
        settle();
        break;
    }
  }

  // ready time from when the speaker really stopped (latest last-word end any side saw), as in the main UI
  function ready(m) {
    const met = S.metric[m];
    if (!met) return null;
    const se = S.se[m] ?? S.speechEnd;
    return se + met.ready_ms - (S.speechEnd ?? se);
  }

  // ---------------------------------------------------------------- question words
  function renderWords() {
    const box = $("#fWords");
    if (box.querySelector(".f-ph")) box.innerHTML = "";
    let nodes = Array.from(box.querySelectorAll(":scope > .fw"));
    S.words.forEach((wd, i) => {
      let n = nodes[i];
      if (!n) {
        n = document.createElement("span");
        n.className = "fw in";
        n.innerHTML = `<span class="tx"></span>`;
        n.querySelector(".tx").textContent = wd.w;
        box.appendChild(n);
        box.appendChild(document.createTextNode(" "));
      } else if (n.querySelector(".tx").textContent !== wd.w) {
        const old = n.querySelector(".tx").textContent;
        n.querySelector(".tx").textContent = wd.w;
        n.classList.remove("rev", "in"); void n.offsetWidth; n.classList.add("rev");
        n.querySelector(".was")?.remove();
        const was = document.createElement("span");
        was.className = "was"; was.textContent = old;
        n.appendChild(was);
        setTimeout(() => was.remove(), 2200);
      }
    });
    nodes = Array.from(box.querySelectorAll(":scope > .fw"));
    while (nodes.length > S.words.length) { const n = nodes.pop(); n.nextSibling && n.nextSibling.nodeType === 3 && n.nextSibling.remove(); n.remove(); }
    renderSpans();
    fit(box, 40);
  }

  const SLOT_SHORT = { organization: "org", organisation: "org", location: "place", place: "place", city: "place", country: "place",
    person: "person", date: "date", time: "time", cardinal: "number", ordinal: "ordinal", number: "number", numeral: "number",
    money: "money", percent: "percent", quantity: "quantity", year: "year", event: "event", product: "product", work_of_art: "title" };
  const slotLabel = (t) => SLOT_SHORT[String(t).toLowerCase()] || String(t).replace(/_/g, " ");
  const isNum = (t) => /^(cardinal|ordinal|money|percent|date|time|quantity|numeral|year|number|duration|age)/.test(String(t));

  function renderSpans() {
    const nodes = Array.from($("#fWords").querySelectorAll(":scope > .fw"));
    nodes.forEach((n) => { n.classList.remove("ent", "num", "drop"); n.querySelector(".lab")?.remove(); });
    for (const s of S.spans || []) {
      const fam = String(s.type).startsWith("number") ? "num" : String(s.type).startsWith("entity") ? "ent" : null;
      if (!fam || s.type === "entity.numeral") continue;
      for (let i = s.start; i < s.end && i < nodes.length; i++) {
        nodes[i].classList.add(fam);
        if (s.dropped) nodes[i].classList.add("drop");
        if (i === s.start) {
          const l = document.createElement("span");
          l.className = "lab";
          l.textContent = s.dropped ? "corrected" : slotLabel(String(s.type).split(".").pop());
          nodes[i].appendChild(l);
        }
      }
    }
  }

  // ---------------------------------------------------------------- Arc read-out
  // Known frame fields get fixed renderings; anything else the tagger adds (dialogue acts, events, number spans, PII
  // flags…) is rendered generically so a richer read-out shows up without a UI change.
  const KNOWN = new Set(["t", "mode", "ms", "intent", "p", "coarse", "slots", "content", "route", "corrected", "carried", "query",
    "n", "command_complete", "final", "spans", "_rx", "_wall", "t_clip", "docs", "words"]);
  const human = (s) => String(s).replace(/^(discourse\.)?(intent|act|event|dialog(ue)?_?act|da)\./, "").replace(/[._]/g, " ").trim();
  function genericChips(f) {
    const out = [];
    for (const [k, v] of Object.entries(f)) {
      if (KNOWN.has(k) || v == null || v === false || v === "" || (Array.isArray(v) && !v.length)) continue;
      const key = human(k);
      if (v === true) out.push(`<span class="fc flag">${esc(key)}</span>`);
      else if (typeof v === "number") out.push(`<span class="fc"><small>${esc(key)}</small>${esc(+v.toFixed(2))}</span>`);
      else if (typeof v === "string") out.push(`<span class="fc"><small>${esc(key)}</small>${esc(human(v))}</span>`);
      else if (Array.isArray(v)) {
        for (const x of v.slice(0, 4)) {
          if (Array.isArray(x)) out.push(`<span class="fc ${isNum(x[0]) ? "num" : ""}"><small>${esc(human(x[0]))}</small>${esc(x[1])}</span>`);
          else if (x && typeof x === "object") {
            const lab = x.label ?? x.type ?? x.name ?? x.kind ?? key;
            const val = x.text ?? x.value ?? x.span ?? (x.p != null ? (+x.p).toFixed(2) : "");
            out.push(`<span class="fc ${isNum(lab) ? "num" : ""}"><small>${esc(human(lab))}</small>${esc(val)}</span>`);
          } else out.push(`<span class="fc"><small>${esc(key)}</small>${esc(human(x))}</span>`);
        }
      } else if (typeof v === "object") {
        const ents = Object.entries(v).filter(([, y]) => y != null && y !== false && y !== "").slice(0, 3);
        if (ents.length) out.push(`<span class="fc"><small>${esc(key)}</small>${esc(ents.map(([a, b]) => (b === true ? human(a) : `${human(a)} ${typeof b === "number" ? +b.toFixed(2) : b}`)).join(" · "))}</span>`);
      }
    }
    return out;
  }
  function renderRead() {
    const f = S.frame;
    if (!f) return;
    const parts = [];
    const it = f.intent ? f : S.lastIntent;
    if (it && it.intent) parts.push(`<span class="fc intent"><small>intent</small>${esc(human(it.intent))}${it.p != null ? `<em>${(+it.p).toFixed(2)}</em>` : ""}</span>`);
    const slots = (f.slots && f.slots.length ? f.slots : (!f.corrected && S.lastSlots) || []);
    for (const [t, v] of slots) parts.push(`<span class="fc ${isNum(t) ? "num" : "ent"}"><small>${esc(slotLabel(t))}</small>${esc(v)}</span>`);
    if (f.corrected) parts.push(`<span class="fc flag corr">self-correction</span>`);
    if (f.carried) parts.push(`<span class="fc flag mem">“it” → ${esc(f.carried)}</span>`);
    if (f.command_complete) parts.push(`<span class="fc flag">question complete</span>`);
    parts.push(...genericChips(f));
    if (f.query) parts.push(`<span class="fc query"><small>search</small>${esc(f.query)}</span>`);
    const box = $("#fRead");
    const html = parts.join("") || '<span class="f-muted">listening…</span>';
    if (box._h === html) return;
    // only chips of a new kind animate in (an intent whose score moved, or a new search query, just updates)
    const key = (c) => c.className + "|" + (c.querySelector("small") ? c.querySelector("small").textContent : c.textContent);
    const seen = new Set(Array.from(box.children).map(key));
    box._h = html;
    box.innerHTML = html;
    for (const c of box.children) if (!seen.has(key(c))) c.classList.add("new");
    // never clip: drop trailing generic chips (keeping the search query) until it fits
    const kids = () => Array.from(box.children).filter((c) => !c.classList.contains("intent") && !c.classList.contains("query") && !c.classList.contains("ent") && !c.classList.contains("num"));
    let dropped = 0;
    while (box.scrollHeight > box.clientHeight + 2 && kids().length) { kids().at(-1).remove(); dropped++; }
    if (dropped) { const more = document.createElement("span"); more.className = "fc more"; more.textContent = `+${dropped}`; box.insertBefore(more, box.querySelector(".query")); }
  }

  // ---------------------------------------------------------------- panes
  const pk = (m, k) => $("#fp-" + m).querySelector(`[data-k="${k}"]`);
  function status(m) {
    const box = pk(m, "status");
    let s;
    const searches = Array.from(S ? S.fires.values() : []).filter((f) => f.mode === m && f.lane !== "draft");
    if (S.metric[m]) s = "";
    else if (S.firstMs[m] != null) s = "writing the answer…";
    else if (S.endpointMs[m] == null) {
      if (m === "headstart") s = searches.length ? `searching while you speak · ${searches.length} search${searches.length > 1 ? "es" : ""}` : "listening…";
      else s = "waiting for you to finish…";
    } else if (S.sourcesMs[m] == null) s = m === "headstart" ? "picking the best early results…" : "searching the web…";
    else s = "reading sources…";
    if (box.textContent !== s) box.textContent = s;
    box.classList.toggle("live", !!s);
  }

  function renderAnswer(m) {
    const box = pk(m, "ans");
    const txt = S.answer[m] || "";
    const had = box.classList.contains("has");
    box.classList.toggle("has", !!(txt || S.error[m]));
    if (!txt && !S.error[m]) { box.innerHTML = ""; return; }
    if (S.error[m] && !txt) { box.innerHTML = `<span class="f-muted">no answer (${esc(S.error[m])})</span>`; return; }
    const html = esc(txt).replace(/\s*\[(\d+(?:\s*[,–-]\s*\d+)*)\]/g, (_, g) => g.split(/\s*[,–-]\s*/).map((d) => `<sup class="ci">${d}</sup>`).join(""))
      + (S.metric[m] ? "" : '<span class="caret"></span>');
    box.innerHTML = html;
    fitAnswer(box, txt);
    // the answer takes room from the results list: lay the list out again for the new height
    if (!had || box._lastH !== box.offsetHeight) { box._lastH = box.offsetHeight; pk(m, "res")._sig = null; results(m); }

  }
  function fitAnswer(box, txt) {
    if (fit(box, 22)) return;
    // still too long at the smallest size: cut at the last whole sentence that fits
    const sents = txt.match(/[^.!?]+[.!?]+(\s|$)|[^.!?]+$/g) || [txt];
    for (let n = sents.length - 1; n >= 1; n--) {
      const t = sents.slice(0, n).join("").trim();
      box.innerHTML = esc(t).replace(/\s*\[(\d+(?:\s*[,–-]\s*\d+)*)\]/g, (_, g) => g.split(/\s*[,–-]\s*/).map((d) => `<sup class="ci">${d}</sup>`).join(""));
      if (box.scrollHeight <= box.clientHeight + 1) return;
    }
  }
  // ---------------------------------------------------------------- live search results (the hero visual)
  // shared with the main UI: web/results.js (HSResults) holds the view logic, item markup and the FLIP animation
  const R = window.HSResults;
  const specDocs = R.specDocs;
  function resultView(m) {
    return R.view({ fires: Array.from(S.fires.values()).filter((f) => f.mode === m), sourcesMs: S.sourcesMs[m],
      sources: S.sources[m], query: (S.frame && S.frame.query) || "" });
  }
  function results(m) {
    if (!S) return;
    const v = resultView(m);
    const rq = pk(m, "rq"), box = pk(m, "res");
    const head = R.head(v, { mode: m, answered: !!S.answer[m], endpointed: S.endpointMs[m] != null });
    if (rq._h !== head) { rq.innerHTML = head; rq._h = head; }
    rq.className = "f-rq k-" + v.kind;
    box.className = "f-res k-" + v.kind;
    R.flip(box, v.docs, { cls: "ri", px: 17 });
  }

  function settle() {
    for (const m of MODES) if (S.metric[m]) {
      const big = pk(m, "big");
      big.innerHTML = fmtW(ready(m));
      big.className = "f-big final";
      if (ready(m) < 0) pk(m, "wsub").textContent = "before you stopped talking";
      $("#fp-" + m).classList.add("done");
    }
    const a = S.metric.headstart, b = S.metric.baseline;
    if (!a || !b) return;
    const d = ready("baseline") - ready("headstart");
    const [win, lose] = d >= 0 ? ["headstart", "baseline"] : ["baseline", "headstart"];
    const ad = Math.abs(d);
    if (ad < 100) {
      for (const m of MODES) pk(m, "delta").innerHTML = `<span class="dl">dead heat</span>`;
      return;
    }
    pk(win, "delta").innerHTML = `<span class="dv">${fmt1(ad)} faster</span>`;
    pk(lose, "delta").innerHTML = "";
    $("#fp-" + win).classList.add("winner");
    $("#fp-" + lose).classList.add("loser");
  }

  // ---------------------------------------------------------------- stopwatches + timeline (every frame)
  function tick() {
    for (const m of MODES) {
      if (S.metric[m]) continue;
      const big = pk(m, "big");
      if (S.se[m] == null || S.speechEnd == null) continue;
      const end = S.firstMs[m] != null ? S.firstMs[m] : now();
      const v = fmtW(Math.max(0, end - S.speechEnd));
      if (big._v !== v) { big.innerHTML = v; big._v = v; }
      big.className = "f-big running";
    }
    timeline();
  }

  // approximate glyph widths of the timeline's word chips and bar labels (focus modes use bigger type)
  const cw = { word: 15, bar: 13 };
  const ctx2d = document.createElement("canvas").getContext("2d");
  const textW = (t, px, mono) => { ctx2d.font = mono ? `500 ${px}px "JetBrains Mono", monospace` : `600 ${px}px Inter, sans-serif`; return ctx2d.measureText(t).width; };
  function measure() {
    const cs = getComputedStyle(root);
    cw.word = parseFloat(cs.getPropertyValue("--tl-word")) || 15;
    cw.bar = parseFloat(cs.getPropertyValue("--tl-bar")) || 13;
  }
  function timeline() {
    const running = !S.done;
    let need = 2500;
    const lw = S.words.at(-1);
    if (lw && lw.t1 != null) need = Math.max(need, lw.t1 * 1000);
    for (const f of S.fires.values()) need = Math.max(need, f.t1 ?? (running ? now() : f.t0));
    for (const m of MODES) {
      if (S.metric[m] && S.se[m] != null) need = Math.max(need, S.se[m] + S.metric[m].ready_ms);
      if (S.endpointMs[m] != null && !S.metric[m] && running) need = Math.max(need, now());
    }
    const racing = !(S.metric.headstart && S.metric.baseline);
    if (racing && S.base != null) need = Math.max(need, now());
    S.axEnd = Math.max(S.axEnd, need * 1.1 + 150);
    const span = S.axEnd;
    const X = (ms) => Math.max(0, Math.min(100, (ms / span) * 100));

    const axis = $("#fAxis");
    const step = [250, 500, 1000, 2000][[250, 500, 1000, 2000].findIndex((s) => span / s <= 10)] || 2000;
    let ah = "";
    for (let t = 0; t <= span; t += step) ah += `<i style="left:${X(t)}%"><span>${t % 1000 ? (t / 1000).toFixed(1) : t / 1000} s</span></i>`;
    if (axis._h !== ah) { axis.innerHTML = ah; axis._h = ah; }

    // spoken words at their times
    const you = $("#fTlYou");
    const tw = you.clientWidth || 1500;
    let yh = "", prevEnd = -1e9;
    const ws = S.words.filter((w) => w.t0 != null);
    ws.forEach((w, i) => {
      const a = Math.max((X(w.t0 * 1000) / 100) * tw, prevEnd + 3);
      const next = i + 1 < ws.length ? (X(ws[i + 1].t0 * 1000) / 100) * tw : Infinity;
      let b = Math.max((X((w.t1 ?? w.t0 + 0.2) * 1000) / 100) * tw, a + 4);
      b = Math.min(b, Math.max(a + 4, next - 3));
      const wd = b - a;
      yh += `<div class="fsw" style="left:${a.toFixed(1)}px;width:${wd.toFixed(1)}px">${wd >= textW(w.w, cw.word) + 10 ? esc(w.w) : ""}</div>`;
      prevEnd = b;
    });
    if (you._h !== yh) { you.innerHTML = yh; you._h = yh; }

    // per side: searches (stacked into lanes) + the answer bar
    for (const m of MODES) {
      const track = $("#fTl-" + m);
      const bars = [];
      for (const k of S.order) {
        const f = S.fires.get(k);
        if (f.mode !== m || f.lane === "draft") continue;
        const t1 = f.t1 ?? (running ? now() : f.t0);
        bars.push({ cls: `srch s-${f.state}${f.used ? " s-used" : ""}${f.lane.startsWith("final") ? " s-after" : ""}`, t0: f.t0, t1, label: f.query });
      }
      if (S.metric[m] && S.se[m] != null) {
        const r = S.se[m] + S.metric[m].ready_ms;
        bars.push({ cls: "ansbar", t0: S.sourcesMs[m] ?? S.endpointMs[m], t1: r, label: "answer", ans: true });
      } else if (S.sourcesMs[m] != null) bars.push({ cls: "ansbar s-live", t0: S.sourcesMs[m], t1: S.firstMs[m] ?? now(), label: "answer", ans: true });
      // lanes: greedy stacking so overlapping searches don't hide each other (max 3)
      const lanes = [];
      for (const b of bars) {
        if (b.ans) { b.lane = 0; continue; }
        let li = lanes.findIndex((end) => b.t0 >= end + 30);
        if (li < 0) { li = Math.min(lanes.length, 2); }
        lanes[li] = Math.max(lanes[li] ?? 0, b.t1);
        b.lane = li;
      }
      const nl = Math.max(1, lanes.length);
      let h = "";
      for (const b of bars) {
        const top = b.ans ? 0 : (b.lane / nl) * 100;
        const ht = b.ans ? 100 : 100 / nl;
        const l = X(b.t0), w = Math.max(0.4, X(b.t1) - X(b.t0));
        const px = (w / 100) * tw;
        const bw = (t) => textW(t, b.ans ? cw.bar + 2 : cw.bar, !b.ans) + 18;
        const lab = px >= bw(b.label) ? b.label : !b.ans && px >= bw("search") ? "search" : "";
        h += `<div class="fbar ${b.cls}" style="left:${l}%;width:${w}%;top:${top}%;height:${ht}%">${lab ? `<span>${esc(lab)}</span>` : ""}</div>`;
      }
      if (track._h !== h) { track.innerHTML = h; track._h = h; }
    }

    // markers: you stopped talking, each side's answer
    let mh = "";
    if (racing && S.base != null) mh += `<i class="mk now" style="left:${X(now())}%"></i>`;
    if (S.speechEnd != null) mh += `<i class="mk spk" style="left:${X(S.speechEnd)}%"><span>you stopped talking</span></i>`;

    const mk = $("#fMarks");
    if (mk._h !== mh) { mk.innerHTML = mh; mk._h = mh; }
    // label collision: the two "answers" labels and the speech-end label
    const labs = Array.from(mk.querySelectorAll(".mk span"));
    labs.forEach((s) => s.classList.remove("lo", "flip"));
    labs.forEach((s) => { const r = s.getBoundingClientRect(); if (r.right > innerWidth - 20) s.classList.add("flip"); });
    for (let i = 1; i < labs.length; i++) {
      const a = labs[i - 1].getBoundingClientRect(), b = labs[i].getBoundingClientRect();
      if (b.left < a.right + 8 && b.right > a.left - 8 && !labs[i - 1].classList.contains("lo")) labs[i].classList.add("lo");
    }
  }

  function loop() {
    cancelAnimationFrame(raf);
    let tail = 0;
    const step = () => {
      if (!api.active || !S) return;
      tick();
      if (!S.done) tail = 0; else tail++;
      if (tail < 20) raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
  }

  // shrink font (px) until the element's content fits its box; false if it still overflows at min
  // (the CSS size is the maximum: focus modes set larger ones)
  function fit(elm, min) {
    elm.style.fontSize = "";
    let s = parseFloat(getComputedStyle(elm).fontSize) || min;
    while (s > min && elm.scrollHeight > elm.clientHeight + 1) { s -= 2; elm.style.fontSize = s + "px"; }
    return elm.scrollHeight <= elm.clientHeight + 1;
  }

  // ---------------------------------------------------------------- state for the capture script
  function state() {
    if (!S) return null;
    const out = { started: S.started, done: S.done, both: !!(S.metric.headstart && S.metric.baseline), audio0_epoch_ms: audio0Epoch(),
      speech_end_ms: S.speechEnd, question: S.words.map((w) => w.w).join(" "), query: S.frame && S.frame.query, frame: S.frame, sides: {} };
    for (const m of MODES) {
      const met = S.metric[m] || {};
      out.sides[m] = { label: LABEL[m], ready_after_stop_ms: ready(m), endpoint_ms: S.endpointMs[m] ?? null, sources_ms: S.sourcesMs[m] ?? null,
        first_token_ms: S.firstMs[m] ?? null, ready_abs_ms: S.se[m] != null && met.ready_ms != null ? S.se[m] + met.ready_ms : null,
        answer: S.answer[m] || "", error: S.error[m] || null,
        sources: (S.sources[m] || []).map((d) => ({ title: d.title, url: d.url, source: d.source })),
        metric: S.metric[m] || null,
        searches: Array.from(S.fires.values()).filter((f) => f.mode === m).map((f) => ({ lane: f.lane, query: f.query, state: f.state, t0: f.t0, t1: f.t1, used: !!f.used })) };
    }
    const a = out.sides.headstart.ready_after_stop_ms, b = out.sides.baseline.ready_after_stop_ms;
    if (a != null && b != null) { out.delta_ms = b - a; out.winner = Math.abs(b - a) < 100 ? "tie" : b > a ? "arc" : "standard"; }
    return out;
  }

  const api = { active: false, event, start, activate, deactivate, state, events: () => (S ? S.events : []) };
  window.HSFilm = api;
  window.__film = api;
  if (want) activate();          // loaded at the end of <body>: the page's DOM is already there
  addEventListener("keydown", (e) => {
    if (e.target.closest && e.target.closest("input, textarea")) return;
    if ((e.key === "f" || e.key === "F") && !e.metaKey && !e.ctrlKey && !e.altKey) (api.active ? deactivate : activate)();
  });

  // ?q=<file>: auto-start a replay once the server is connected and warm
  const q = P.get("q");
  if (want && q) {
    const delay = +(P.get("delay") || 1500);
    const t0 = performance.now();
    const go = () => {
      const hs = window.__hs;
      if (hs && hs.ready) setTimeout(() => hs.replay(q), delay);
      else if (performance.now() - t0 < 120000) setTimeout(go, 100);
    };
    go();
  }
})();
