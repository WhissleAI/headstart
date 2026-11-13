/* Head Start UI: plain JS, no build. Talks to server/app.py over /ws (protocol in its docstring). */
(() => {
  "use strict";

  const $ = (s, r = document) => r.querySelector(s);
  const el = (tag, cls, txt) => { const e = document.createElement(tag); if (cls) e.className = cls; if (txt != null) e.textContent = txt; return e; };
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const store = { get(k) { try { return localStorage.getItem(k); } catch { return null; } }, set(k, v) { try { localStorage.setItem(k, v); } catch { /* private mode */ } } };
  const MODE_NAME = { headstart: "Head Start", baseline: "Baseline" };
  const fmtS = (ms) => (ms == null || !isFinite(ms) ? "—" : (ms < 0 ? "−" : "") + (Math.abs(ms) / 1000).toFixed(Math.abs(ms) < 9950 ? 2 : 1) + " s");
  const stripIntent = (s) => String(s || "").replace(/^discourse\.intent\./, "").replace(/^intent\./, "").replace(/_/g, " ");

  // ------------------------------------------------------------------ toast (no alert())
  let toastTimer = null;
  function toast(msg, err = false) {
    const t = $("#toast");
    t.textContent = msg;
    t.className = "toast show" + (err ? " err" : "");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => (t.className = "toast" + (err ? " err" : "")), err ? 4500 : 2500);
  }

  // ------------------------------------------------------------------ theme
  const savedTheme = store.get("hs-theme");
  if (savedTheme) document.documentElement.dataset.theme = savedTheme;
  $("#themeBtn").addEventListener("click", () => {
    const cur = document.documentElement.dataset.theme || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    const next = cur === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    store.set("hs-theme", next);
  });

  // ------------------------------------------------------------------ mode toggle
  let modeSel = store.get("hs-mode") || "race";
  const modesFor = (m) => (m === "race" ? ["headstart", "baseline"] : [m]);
  function paintSeg() {
    for (const b of $("#modeSeg").querySelectorAll("button")) {
      const on = b.dataset.mode === modeSel;
      b.classList.toggle("on", on);
      b.setAttribute("aria-checked", on ? "true" : "false");
    }
  }
  $("#modeSeg").addEventListener("click", (e) => {
    const b = e.target.closest("button[data-mode]");
    if (!b) return;
    modeSel = b.dataset.mode;
    store.set("hs-mode", modeSel);
    paintSeg();
  });
  paintSeg();

  // ------------------------------------------------------------------ run / question state
  let run = null;       // {modes, kind, running, clientT0, base, q}
  function newQuestion(axStart = 0) {
    return {
      axStart, axEnd: 3000,
      words: [], spans: [], frame: null, frameMode: null,
      fires: new Map(), order: [],
      speechEnd: null, se: {}, epBy: {}, old: [], superseded: {}, endpointMs: {}, sourcesMs: {}, sources: {}, answer: {}, firstMs: {}, metric: {}, error: {},
      wordsDirty: true, finalWords: 0,
    };
  }
  function startRun(kind, modes) {
    run = { modes, kind, running: true, clientT0: performance.now(), base: null, q: newQuestion(), prevQ: null, history: [] };
    paintHistory();
    resetView();
    pipeReset();
    loop();
  }
  const primary = () => (run ? run.modes[0] : "headstart");
  function serverNow() {
    if (!run) return 0;
    return performance.now() - (run.base ?? run.clientT0);
  }
  function syncClock(ms) {
    if (!run || typeof ms !== "number") return;
    const cand = performance.now() - ms;
    run.base = run.base == null ? cand : Math.min(run.base, cand);
  }

  // ------------------------------------------------------------------ websocket
  let ws = null, wsOpen = false, warm = false;
  function connect() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${proto}://${location.host}/ws`);
    ws.binaryType = "arraybuffer";
    ws.onopen = () => { wsOpen = true; paintStatus(); };
    ws.onclose = () => {
      wsOpen = false; paintStatus();
      if (mic.on) stopMic(false);
      if (run) { run.running = false; }
      setTimeout(connect, 1500);
    };
    ws.onerror = () => { /* onclose follows */ };
    ws.onmessage = (m) => {
      let e;
      try { e = JSON.parse(m.data); } catch { return; }
      try { onEvent(e); } catch (err) { console.error("event handler failed", e, err); }
    };
  }
  function send(obj) {
    if (!ws || ws.readyState !== 1) { toast("Not connected to the server yet", true); return false; }
    ws.send(typeof obj === "string" || obj instanceof ArrayBuffer ? obj : JSON.stringify(obj));
    return true;
  }

  // ------------------------------------------------------------------ status / health
  let health = null;
  function paintStatus() {
    const s = $("#status");
    let cls = "down", txt = "offline";
    if (wsOpen && !warm) { cls = "cold"; txt = "warming models…"; }
    else if (wsOpen) {
      cls = "warm";
      txt = "ready" + (health ? ` · ${health.web_backend} · ${String(health.answer_model || "").replace(/^claude-/, "")}` : "");
    }
    s.className = "status " + cls;
    s.title = txt;
    s.querySelector("b").textContent = txt;
  }
  async function pollHealth() {
    try {
      const r = await fetch("/api/health", { cache: "no-store" });
      health = await r.json();
      warm = !!health.warm;
    } catch { warm = false; }
    paintStatus();
    setTimeout(pollHealth, warm ? 15000 : 2000);
  }

  // ------------------------------------------------------------------ events
  function onEvent(e) {
    if (e.mode === "sys") {
      if (e.t === "ready") { if (e.warm !== undefined) warm = !!e.warm; paintStatus(); }
      else if (e.t === "error") toast(e.error || "server error", true);
      else if (e.t === "run_done" && run) { run.running = false; finishRun(); settlePipe(run.q); }
      return;
    }
    if (!run) return;
    if (typeof e.ms === "number") syncClock(e.ms);
    if (e.mode === "ear") return onEar(e);
    const mode = e.mode;
    let q = run.q;
    // the previous question's answer can still be streaming when the next question starts (conversations, mic):
    // its late events update that turn's history entry, not the new panes
    const pq = run.prevQ;
    if (pq && q.endpointMs[mode] == null && pq.endpointMs[mode] != null && !pq.metric[mode] &&
        (["sources", "answer", "answer_error", "metric", "used"].includes(e.t) || (e.t === "spec" && String(e.lane).startsWith("final")))) {
      if (e.t === "answer") pq.answer[mode] = (pq.answer[mode] || "") + (e.delta || "");
      else if (e.t === "metric") { pq.metric[mode] = e; recordTurn(pq, mode); }
      return;
    }
    switch (e.t) {
      case "words": {
        if (!acceptWords(q, mode, e)) return;
        // a new question in the same run (live mic): start a fresh view once the previous one has answered
        // (words that start after the last answered speech end; not the same utterance continuing)
        const t0w = e.words && e.words.length ? (e.words[0].t0 ?? 0) * 1000 : null;
        if (q.speechEnd != null && t0w != null && t0w > q.speechEnd + 50 && Object.keys(q.endpointMs).length) {
          run.prevQ = q;
          run.q = q = newQuestion(Math.max(0, t0w - 400));
          resetView(true);
          paintHistory();
        }
        if (e.final) q.finalWords++;
        q.prevWords = q.words;
        q.words = e.words || [];
        pipe("ear", "“" + q.words.slice(-4).map((w) => w.w).join(" ") + "”");
        q.revFrom = e.from;
        renderWords(q);
        break;
      }
      case "frame":
        if (mode !== q.wordsMode) return;
        q.frame = e;
        q.spans = e.spans || [];
        if (e.corrected && Array.isArray(e.slots)) {
          // a corrected-away entity ("Canada, actually make that Turkey") stays in spans but not in slots
          const keep = new Set(e.slots.map((x) => String(x[1]).toLowerCase()));
          q.spans = q.spans.filter((sp) => !/^entity|^number/.test(sp.type) || keep.has(String(sp.text).toLowerCase().replace(/[^\w' -]/g, "").trim()));
        }
        renderSpans(q);
        renderFrame(e, !!e.final);
        if (e.carried) pipe("tagger", `memory: ${pronounIn(q.words) || "it"} → ${titleCase(e.carried)}`);
        else pipe("tagger", (e.intent ? stripIntent(e.intent) : "frame") + (e.slots && e.slots.length ? ` · ${e.slots.map((x) => x[1]).join(", ")}` : e.query ? ` · ${e.query}` : ""));
        break;
      case "endpoint":
        q.endpointMs[mode] = e.ms;
        if (e.speech_end_ms != null) { q.se[mode] = e.speech_end_ms; q.speechEnd = Math.max(q.speechEnd ?? 0, e.speech_end_ms); }
        q.epBy[mode] = e.by || null;
        if (e.frame) renderFrame(e.frame, true);     // the latest endpoint's frame wins
        paneState(mode);
        break;
      case "spec": {
        const key = mode + ":" + e.id;
        let f = q.fires.get(key);
        if (!f) {
          f = { key, id: e.id, mode, lane: e.lane, query: e.query, state: "fired", t0: e.ms, t1: null };
          q.fires.set(key, f);
          q.order.push(key);
        }
        if (e.ids) f.ids = e.ids;
        pipeSpec(q, e);
        if (e.state === "fired") { f.t0 = e.ms; break; }
        if (e.state === "stale") {
          if (f.state === "fired") f.t1 = e.ms;     // in flight: cancelled by the stale mark
          f.staleAt = e.ms;
          f.state = "stale";
        } else if (e.state === "reused") {
          f.state = "reused"; f.t1 = e.ms;
        } else {
          if (f.state !== "stale") f.state = e.state;
          if (f.t1 == null) f.t1 = e.ms;
        }
        if (e.took != null) f.took = e.took;
        if (e.n != null) f.n = e.n;
        if (e.top) f.top = e.top;
        if (e.error) f.error = e.error;
        break;
      }
      case "sources":
        q.sourcesMs[mode] = e.ms;
        q.sources[mode] = e.docs || [];
        renderSources(mode);
        break;
      case "used": {
        const ids = new Set(e.fires || []);
        for (const f of q.fires.values()) if (f.mode === mode) f.used = ids.has(f.id);
        break;
      }
      case "answer":
        if (q.firstMs[mode] == null) q.firstMs[mode] = e.ms;
        q.answer[mode] = (q.answer[mode] || "") + (e.delta || "");
        q.lastAnsMs = e.ms;
        renderAnswer(mode);
        if (Object.keys(q.metric).length) settlePipe(q); else pipe("answer", `${MODE_NAME[mode]} streaming…`);
        break;
      case "answer_error":
        q.error[mode] = e.error || "answer failed";
        renderAnswer(mode);
        break;
      case "extended": {
        // the speaker kept going after an early (semantic) endpoint: that answer is superseded, a new one follows
        q.old.push({ mode, ep: q.endpointMs[mode], by: q.epBy[mode], t0: q.sourcesMs[mode] ?? q.endpointMs[mode],
          t1: q.metric[mode] && q.se[mode] != null ? q.se[mode] + q.metric[mode].done_ms : e.ms, first: q.firstMs[mode],
          answer: q.answer[mode] || "", ready: q.metric[mode]?.ready_ms, path: q.metric[mode]?.path });
        q.superseded[mode] = q.old.at(-1);
        for (const k of ["endpointMs", "sourcesMs", "sources", "answer", "firstMs", "metric", "error", "se", "epBy"]) delete q[k][mode];
        q.speechEnd = Math.max(0, ...Object.values(q.se));
        if (!Object.keys(q.se).length) q.speechEnd = null;
        q.lastAnsMs = null;
        for (const k of ["answer", "stats", "sources", "path", "delta"]) pk(mode, k).innerHTML = "";
        renderAnswer(mode);
        toast(`${MODE_NAME[mode]}: you kept talking, re-answering the longer question`);
        break;
      }
      case "metric":
        q.metric[mode] = e;
        if (q.se[mode] == null && q.endpointMs[mode] != null) { q.se[mode] = q.endpointMs[mode] - (e.endpoint_lag_ms || 0); q.speechEnd = Math.max(q.speechEnd ?? 0, q.se[mode]); }
        // drafts still open at the endpoint were never used
        for (const f of q.fires.values()) {
          if (f.mode === mode && f.lane === "draft" && f.state === "fired") { f.state = "unused"; f.t1 = q.endpointMs[mode] ?? serverNow(); }
        }
        renderAnswer(mode);
        renderMetric(mode);
        recordTurn(q, mode);
        settlePipe(q);
        if (mode === "headstart" || !run.modes.includes("headstart")) pipe("answer", `ready ${fmtS(readyAdj(q, mode))} after you stopped`);
        break;
    }
  }

  function onEar(e) {
    const s = $("#earState");
    if (e.kind === "partial") { s.textContent = "hearing…"; s.className = "ear live"; pipe("ear", "“" + String(e.text || "").split(/\s+/).slice(-4).join(" ") + "”"); }
    else if (e.kind === "final") {
      s.textContent = `utterance closed${e.reason ? " · " + e.reason : ""}`; s.className = "ear";
      pipe("ear", "endpoint" + (e.reason ? " · " + e.reason : ""));
      // sessions send a final words event at their endpoint; if none comes (an earlier semantic endpoint already
      // answered this utterance), fall back to the ear's own text
      const q = run.q, before = q.finalWords;
      setTimeout(() => {
        if (run.q !== q || q.finalWords !== before) return;
        const toks = String(e.text || "").split(/\s+/).map((w) => w.replace(/[.?!,]+$/, "")).filter(Boolean);
        if (toks.length && toks.join(" ") !== q.words.map((w) => w.w).join(" ")) {
          q.words = toks.map((w, i) => ({ w, t0: q.words[i]?.t0 ?? null, t1: q.words[i]?.t1 ?? null }));
          q.spans = [];                // tagger spans index the session's words, not this re-tokenized text
          renderWords(q);
        }
      }, 400);
    }
    else if (e.kind === "arc_command_complete") { s.textContent = `Arc: question complete (p ${(+e.p || 0).toFixed(2)})`; s.className = "ear"; }
  }

  // ------------------------------------------------------------------ "how it works" strip
  const pipeTimers = {};
  function pipe(stage, sub) {
    const li = document.querySelector(`#pipe li[data-st="${stage}"]`);
    if (!li) return;
    li.classList.add("on");
    if (sub != null) li.querySelector("small").textContent = sub;
    li.classList.remove("hit");
    void li.offsetWidth;
    li.classList.add("hit");
    clearTimeout(pipeTimers[stage]);
    pipeTimers[stage] = setTimeout(() => li.classList.remove("hit"), 700);
  }
  const PIPE_IDLE = {};
  function pipeReset() {
    document.querySelectorAll("#pipe li").forEach((li) => {
      const st = li.dataset.st;
      if (!(st in PIPE_IDLE)) PIPE_IDLE[st] = li.querySelector("small").textContent;
      li.classList.remove("on", "hit");
      li.querySelector("small").textContent = PIPE_IDLE[st];
    });
  }
  // once answers are in, replace transient states ("drafting…", "streaming…") with what actually happened
  function settlePipe(q) {
    if (!run) return;
    const done = run.modes.filter((m) => q.metric[m]);
    if (!done.length) return;
    const SHORT = { headstart: "Head Start", baseline: "Baseline" };
    setPipe("answer", run.modes.map((m) => q.metric[m] ? `${SHORT[m]} ${fmtS(readyAdj(q, m))}` : `${SHORT[m]} streaming…`).join(" · "));
    const hm = q.metric.headstart;
    if (hm) {
      const bits = [`${hm.spec_fires ?? 0} early searches`];
      if (hm.stale) bits.push(`${hm.stale} stale`);
      if (hm.drafts) bits.push(hm.draft_reused ? "draft reused" : `${hm.drafts} draft${hm.drafts > 1 ? "s" : ""} unused`);
      setPipe("spec", bits.join(" · "));
    } else if (!run.modes.includes("headstart")) setPipe("spec", "off (baseline only)");
  }
  function setPipe(stage, sub) {
    const li = document.querySelector(`#pipe li[data-st="${stage}"]`);
    if (li) li.querySelector("small").textContent = sub;
  }

  function pipeSpec(q, e) {
    const fires = Array.from(q.fires.values());
    if (e.lane === "draft") {
      if (q.metric.headstart) return;
      if (e.state === "fired") pipe("spec", "drafting an answer early");
      else if (e.state === "reused") pipe("spec", "draft reused at endpoint");
      return;
    }
    const spec = fires.filter((f) => !f.lane.startsWith("final") && f.lane !== "draft");
    if (!e.lane.startsWith("final") && !q.metric.headstart) {
      const stale = spec.filter((f) => f.state === "stale" || f.state === "cancelled").length;
      const mem = q.frame && q.frame.carried ? `memory: ${titleCase(q.frame.carried)} · ` : "";
      pipe("spec", `${mem}${spec.length} early search${spec.length === 1 ? "" : "es"}${stale ? ` · ${stale} stale` : ""}`);
    }
    const cnt = {};
    for (const f of fires) if (f.lane !== "draft") { const l = f.lane.replace("final-", ""); cnt[l] = (cnt[l] || 0) + 1; }
    pipe("lanes", Object.entries(cnt).map(([l, n]) => `${l} ${n}`).join(" · "));
  }

  // both sessions emit the same words; follow whichever is still listening (Head Start stops after an early endpoint)
  function acceptWords(q, mode, e) {
    const sig = (e.words || []).map((w) => w.w).join(" ");
    if (mode !== primary() && q.wordsMode === primary() && q.endpointMs[primary()] == null) return false;
    if (sig === q.wordsSig && q.wordsMode && mode !== q.wordsMode) return false;
    q.wordsSig = sig;
    q.wordsMode = mode;
    return true;
  }

  // ------------------------------------------------------------------ conversation history (previous turns)
  function recordTurn(q, mode) {
    if (!run) return;
    let h = run.history.find((x) => x.q === q);
    if (!h) { h = { q, question: "", answers: {} }; run.history.push(h); }
    const m = q.metric[mode] || {};
    h.question = h.question || m.question || q.words.map((w) => w.w).join(" ");
    h.answers[mode] = q.answer[mode] || m.answer || "";
    h.carried = h.carried || (q.frame && q.frame.carried) || "";
    paintHistory();
  }
  function shortAnswer(t) {
    t = String(t || "").replace(/\s*\[\d+(?:\s*[,–-]\s*\d+)*\]/g, "").trim();
    const first = (t.match(/^.*?[.!?](\s|$)/) || [t])[0].trim();
    return first.length > 90 ? first.slice(0, 88) + "…" : first;
  }
  function paintHistory() {
    const box = $("#history");
    const prev = run ? run.history.filter((h) => h.q !== run.q).slice(-3) : [];
    box.hidden = !prev.length;
    box.innerHTML = prev.length ? `<span class="hl-lab">Earlier in this conversation</span>` + prev.map((h) => {
      const a = h.answers.headstart || h.answers.baseline || "";
      return `<span class="turn"><span class="tq">${esc(h.question)}</span><span class="ta">→ ${esc(shortAnswer(a) || "…")}</span></span>`;
    }).join("") : "";
  }

  function finishRun() {
    $("#earState").textContent = "done";
    $("#earState").className = "ear";
    if (mic.on) stopMic(false);
    for (const m of run.modes) paneState(m);
  }

  // ------------------------------------------------------------------ transcript
  function resetView(keepEar) {
    const tr = $("#transcript");
    tr.innerHTML = "";
    tr.dataset.empty = "1";
    if (!keepEar) { $("#earState").textContent = run && run.kind === "mic" ? "listening" : "starting…"; $("#earState").className = "ear live"; }
    renderFrame(null);
    for (const m of ["headstart", "baseline"]) {
      const p = $("#pane-" + m);
      p.classList.toggle("off", !!run && !run.modes.includes(m));
      p.classList.remove("winner");
      for (const k of ["answer", "stats", "sources", "path", "delta"]) { const n = p.querySelector(`[data-k="${k}"]`); n.innerHTML = ""; n.className = n.className.replace(/ ?(p-\S+|slow|err)/g, ""); }
      p.querySelector('[data-k="delta"]').className = "delta";
      p.querySelector('[data-k="big"]').textContent = "—";
      p.querySelector('[data-k="big"]').className = "big";
      p.querySelector('[data-k="sub"]').textContent = run && !run.modes.includes(m) ? "not in this run" : "after you stopped talking";
    }
    $("#tlBody").innerHTML = "";
    tlNodes.clear();
  }

  function renderWords(q) {
    const tr = $("#transcript");
    if (tr.dataset.empty) { tr.innerHTML = ""; delete tr.dataset.empty; }
    const nodes = tr.children;
    q.words.forEach((wd, i) => {
      const txt = wd.w;
      let n = nodes[i];
      if (!n) {
        n = el("span", "w new");
        n.appendChild(el("span", "tx", txt));
        tr.appendChild(n);
        tr.appendChild(document.createTextNode(" "));
        // keep word nodes at even indices: use a wrapper approach instead
      } else if (n.querySelector(".tx").textContent !== txt) {
        const old = n.querySelector(".tx").textContent;
        n.querySelector(".tx").textContent = txt;
        n.classList.remove("revised", "new");
        void n.offsetWidth;          // restart the animation
        n.classList.add("revised");
        n.querySelector(".was")?.remove();
        const was = el("span", "was", old);
        n.appendChild(was);
        setTimeout(() => was.remove(), 1700);
      }
    });
    tr.classList.toggle("long", q.words.map((w) => w.w).join(" ").length > 42);
    // words removed by a revision
    while (wordNodes(tr).length > q.words.length) wordNodes(tr).at(-1).remove();
    renderSpans(q);
  }
  function wordNodes(tr) { return Array.from(tr.querySelectorAll(":scope > .w")); }

  function renderSpans(q) {
    const nodes = wordNodes($("#transcript"));
    nodes.forEach((n) => { n.classList.remove("slot-entity", "slot-number"); n.querySelector(".lab")?.remove(); });
    for (const s of q.spans || []) {
      const fam = s.type.startsWith("number") ? "number" : s.type.startsWith("entity") ? "entity" : null;
      if (!fam || s.type === "entity.numeral") continue;
      for (let i = s.start; i < s.end && i < nodes.length; i++) {
        nodes[i].classList.add("slot-" + fam);
        if (i === s.start) nodes[i].appendChild(el("span", "lab", s.type.split(".").pop().replace(/_/g, " ")));
      }
    }
  }

  // ------------------------------------------------------------------ frame card
  const PRONOUNS = ["it", "its", "it's", "he", "she", "his", "her", "him", "they", "them", "their", "there", "that", "this"];
  const pronounIn = (words) => { for (const w of words || []) { const t = String(w.w || w).toLowerCase().replace(/[^a-z']/g, ""); if (PRONOUNS.includes(t)) return t; } return null; };
  const titleCase = (s) => String(s || "").replace(/\b\w/g, (c) => c.toUpperCase());

  function renderFrame(f, final = false) {
    const badges = $("#frameBadges");
    badges.innerHTML = "";
    if (!f) {
      $("#fIntent").textContent = "—"; $("#fSlots").textContent = "—"; $("#fRoute").textContent = "—"; $("#fQuery").textContent = "—";
      return;
    }
    const p = Math.max(0, Math.min(1, +f.p || 0));
    $("#fIntent").innerHTML = f.intent
      ? `<span class="nm" title="${esc(f.intent)}">${esc(stripIntent(f.intent))}</span><span class="pbar" title="p ${p.toFixed(2)}"><i style="width:${Math.round(p * 100)}%"></i></span>`
      : `<span class="muted">listening for intent…</span>`;
    const slots = f.slots || [];
    $("#fSlots").innerHTML = slots.length
      ? slots.map(([t, v]) => `<span class="chip ${/^(cardinal|ordinal|money|percent|date|time|quantity|numeral|year|number)/.test(t) ? "number" : ""}"><small>${esc(t.replace(/_/g, " "))}</small>${esc(v)}</span>`).join("")
      : `<span class="muted">none yet</span>`;
    const r = f.route || "web";
    $("#fRoute").innerHTML = `<span class="route"><span class="web ${r === "web" || r === "both" ? "on" : ""}">web</span><span class="local ${r === "local" || r === "both" ? "on" : ""}">local</span></span>`;
    $("#fQuery").textContent = f.query || "—";
    if (f.carried) {
      const m = el("span", "badge mem");
      m.innerHTML = `<i>memory</i> ${esc(pronounIn(run && run.q ? run.q.words : []) || "it")} → ${esc(titleCase(f.carried))}`;
      m.title = "Resolved from the previous turn of this conversation";
      badges.appendChild(m);
    }
    if (f.corrected) badges.appendChild(el("span", "badge corr", "corrected"));
    if (f.command_complete && !f.carried) badges.appendChild(el("span", "badge cc", "complete"));
    if (final) badges.appendChild(el("span", "badge cc", "final"));
  }

  // ------------------------------------------------------------------ answer panes
  const pane = (m) => $("#pane-" + m);
  const pk = (m, k) => pane(m).querySelector(`[data-k="${k}"]`);

  function renderAnswer(mode) {
    const q = run.q;
    const box = pk(mode, "answer");
    const txt = q.answer[mode] || "";
    if (q.error[mode] && !txt) { box.className = "answer err"; box.textContent = "Answer failed: " + q.error[mode]; return; }
    box.className = "answer";
    const n = (q.sources[mode] || []).length;
    let html = esc(txt).replace(/\[(\d+(?:\s*[,–-]\s*\d+)*)\]/g, (_, grp) =>
      grp.split(/\s*[,–-]\s*/).map((d) => `<a class="cite" data-n="${d}" ${+d > n ? 'title="not in the source list"' : ""}>${d}</a>`).join(""));
    if (!q.metric[mode] && !q.error[mode]) html += txt ? '<span class="caret"></span>' : (q.endpointMs[mode] != null ? '<span class="wait">searching…</span>' : "");
    const old = q.superseded[mode];
    if (old && old.answer) html = `<div class="old"><span class="badge corr">superseded</span> answered early (${fmtS(old.ready)}), then you kept talking: <s>${esc(old.answer.slice(0, 140))}${old.answer.length > 140 ? "…" : ""}</s></div>` + html;
    box.innerHTML = html;
  }

  function renderSources(mode) {
    const box = pk(mode, "sources");
    const docs = run.q.sources[mode] || [];
    box.innerHTML = "";
    docs.forEach((d, i) => {
      const a = el(d.url ? "a" : "div", "src");
      if (d.url) { a.href = d.url; a.target = "_blank"; a.rel = "noopener"; }
      a.dataset.n = String(i + 1);
      a.title = (d.title || "") + (d.text ? "\n\n" + d.text : "");
      const lane = String(d.lane || "").replace("final-", "");
      const meta = [d.source, d.score != null ? `cos ${(+d.score).toFixed(2)}` : null].filter(Boolean).join(" · ");
      const ts = d.t != null ? `<span class="ts" title="moment starts at ${Math.round(d.t)} s">▶ ${mmss(d.t)}</span>` : "";
      const title = d.t != null ? String(d.title || "").replace(/\s*·\s*\d+:\d+$/, "") : d.title;
      a.innerHTML = `<div class="st"><span class="sn">[${i + 1}]</span><span class="tt">${esc(title || "(untitled)")}</span></div>` +
        `<div class="sm">${lane ? `<span class="ln ${esc(lane)}">${esc(lane)}</span>` : ""}${ts}<span class="mt">${esc(meta)}</span></div>`;
      box.appendChild(a);
    });
  }

  const mmss = (t) => { t = Math.max(0, Math.round(+t)); return `${String(Math.floor(t / 60)).padStart(2, "0")}:${String(t % 60).padStart(2, "0")}`; };

  function paneState(mode) {
    const q = run.q;
    const big = pk(mode, "big");
    if (q.metric[mode]) return;
    if (q.speechEnd == null && q.endpointMs[mode] == null) { big.textContent = run.running ? "…" : "—"; big.className = "big"; }
  }

  function renderMetric(mode) {
    const q = run.q, m = q.metric[mode];
    const big = pk(mode, "big");
    big.textContent = fmtS(readyAdj(q, mode));
    big.className = "big final";
    if (readyAdj(q, mode) < 0) pk(mode, "sub").textContent = "before you stopped talking";
    const path = pk(mode, "path");
    path.textContent = m.path || "";
    path.className = "path" + (m.path ? " p-" + m.path : "");
    const st = [];
    if (mode === "headstart") {
      st.push(`fires <b>${m.spec_fires ?? 0}</b>`, `fed <b>${m.spec_used ?? 0}</b>`, `stale <b>${m.stale ?? 0}</b>`,
        `drafts <b>${m.drafts ?? 0}</b>${m.draft_reused ? "↺" : ""}`);
    }
    st.push(`web <b>${m.web_calls ?? 0}</b>`, `sources <b>${fmtS(m.sources_ms)}</b>`);
    if (mode !== "headstart") st.push(`done <b>${fmtS(m.done_ms)}</b>`);
    const box = pk(mode, "stats");
    box.innerHTML = st.map((s) => `<span>${s}</span>`).join("");
    box.title = `spec fires ${m.spec_fires ?? 0} · fed the answer ${m.spec_used ?? 0} · stale/cancelled ${m.stale ?? 0} · drafts ${m.drafts ?? 0}${m.draft_reused ? " (one reused)" : ""} · web calls ${m.web_calls ?? 0} · sources ${fmtS(m.sources_ms)} · done ${fmtS(m.done_ms)} · endpoint lag ${m.endpoint_lag_ms ?? "—"} ms (all after you stopped talking)`;
    if (m.error && !q.answer[mode]) { q.error[mode] = m.error; renderAnswer(mode); }
    // delta when both modes have finished
    const hs = q.metric.headstart, bl = q.metric.baseline;
    if (hs && bl) {
      for (const o of ["headstart", "baseline"]) { if (o !== mode) { pk(o, "big").textContent = fmtS(readyAdj(q, o)); } }
      const d = readyAdj(q, "baseline") - readyAdj(q, "headstart");
      const dh = pk("headstart", "delta"), db = pk("baseline", "delta");
      const [win, lose] = d >= 0 ? [dh, db] : [db, dh];
      const ad = Math.abs(d), ratio = Math.max(readyAdj(q, "baseline"), readyAdj(q, "headstart")) / Math.max(1, Math.min(readyAdj(q, "baseline"), readyAdj(q, "headstart")));
      if (ad < 100) {
        dh.innerHTML = db.innerHTML = `<span class="dl">dead heat (${Math.round(ad)} ms)</span>`;
        dh.className = db.className = "delta slow";
        return;
      }
      win.innerHTML = `<span class="dv">${fmtS(ad)}</span><span class="dl">faster${isFinite(ratio) && ratio >= 1.15 && Math.min(readyAdj(q, "baseline"), readyAdj(q, "headstart")) > 100 ? ` · ${ratio.toFixed(1)}× sooner` : ""}</span>`;
      win.className = "delta hero";
      lose.innerHTML = `<span class="dl">${fmtS(ad)} behind</span>`;
      lose.className = "delta slow";
      pane(d >= 0 ? "headstart" : "baseline").classList.add("winner");
    }
  }

  // ready time measured from when the speaker really stopped (the latest last-word end any mode saw). Each session's
  // own ready_ms counts from ITS endpoint frame's last word, and an early semantic endpoint can see an earlier one.
  function readyAdj(q, m) {
    const met = q.metric[m];
    if (!met) return null;
    const se = q.se[m] ?? q.speechEnd;
    return se + met.ready_ms - (q.speechEnd ?? se);
  }

  // live stopwatch tick: counts from "you stopped talking" until the first answer token
  function tickWatches() {
    if (!run) return;
    const q = run.q;
    for (const m of run.modes) {
      if (q.metric[m]) continue;
      const big = pk(m, "big");
      if (q.se[m] == null || q.speechEnd == null) continue;
      const end = q.firstMs[m] != null ? q.firstMs[m] : serverNow();
      big.textContent = fmtS(end - q.speechEnd);
      big.className = "big running";
    }
  }

  // cite hover / click -> source card
  $("#answers").addEventListener("mouseover", (e) => {
    const c = e.target.closest(".cite");
    const p = e.target.closest(".pane");
    if (!p) return;
    p.querySelectorAll(".src.hl").forEach((n) => n.classList.remove("hl"));
    if (c) p.querySelector(`.src[data-n="${c.dataset.n}"]`)?.classList.add("hl");
  });
  $("#answers").addEventListener("click", (e) => {
    const c = e.target.closest(".cite");
    if (!c) return;
    const src = c.closest(".pane").querySelector(`.src[data-n="${c.dataset.n}"]`);
    if (src && src.href) window.open(src.href, "_blank", "noopener");
  });

  // ------------------------------------------------------------------ timeline
  const tlNodes = new Map();      // key -> row element
  const NICE = [100, 200, 250, 500, 1000, 2000, 2500, 5000, 10000];

  function tlRows(q) {
    const rows = [];
    const now = run.running ? serverNow() : null;
    rows.push({ key: "speech", kind: "speech" });
    for (const mode of run.modes) {
      const startIdx = rows.length;
      for (const key of q.order) {
        const f = q.fires.get(key);
        if (f.mode !== mode) continue;
        rows.push({ key, kind: "fire", f, t0: f.t0, t1: f.t1 ?? now ?? f.t0 });
      }
      q.old.forEach((o, i) => {
        if (o.mode === mode && o.t0 != null) rows.push({ key: `old:${mode}:${i}`, kind: "answer", mode, t0: o.t0, t1: Math.max(o.t1 ?? o.t0, o.t0), first: o.first, superseded: true });
      });
      if (q.sourcesMs[mode] != null) {
        const m = q.metric[mode];
        const end = m && q.se[mode] != null ? q.se[mode] + m.done_ms : (q.lastAnsMs ?? now ?? q.sourcesMs[mode]);
        rows.push({ key: "ans:" + mode, kind: "answer", mode, t0: q.sourcesMs[mode], t1: Math.max(end, q.sourcesMs[mode]),
          first: q.firstMs[mode], live: !m });
      } else if (q.endpointMs[mode] != null) {
        rows.push({ key: "ans:" + mode, kind: "answer", mode, t0: q.endpointMs[mode], t1: now ?? q.endpointMs[mode], waiting: true, live: true });
      }
      if (rows[startIdx]) rows[startIdx].first = true;
      for (let i = startIdx; i < rows.length; i++) rows[i].mode = rows[i].mode || mode;
    }
    return rows;
  }

  function renderTimeline() {
    if (!run) return;
    const q = run.q;
    const rows = tlRows(q);
    // axis range (grows monotonically during a run so bars don't jitter)
    let need = q.axStart + 2500;
    const lastWord = q.words.at(-1);
    if (lastWord && lastWord.t1 != null) need = Math.max(need, lastWord.t1 * 1000);
    for (const r of rows) if (r.t1 != null) need = Math.max(need, r.t1);
    if (q.speechEnd != null) need = Math.max(need, q.speechEnd);
    for (const m of run.modes) if (q.metric[m] && q.se[m] != null) need = Math.max(need, q.se[m] + q.metric[m].done_ms);
    if (run.running) need = Math.max(need, serverNow());
    const target = q.axStart + (need - q.axStart) * 1.16 + 200;
    q.axEnd = run.running ? Math.max(q.axEnd, target) : target;
    const a0 = q.axStart, span = Math.max(500, q.axEnd - a0);
    const X = (ms) => ((ms - a0) / span) * 100;

    // ticks
    const axis = $("#tlAxis");
    const w = axis.clientWidth || 800;
    const step = NICE.find((s) => (span / s) * 64 <= w) || 10000;
    let tickHtml = "";
    const SPK = "you stopped talking";
    let spkA = null, spkB = null;
    if (q.speechEnd != null) {
      const sp = (X(q.speechEnd) / 100) * w, sw = SPK.length * 6.3 + 14;
      [spkA, spkB] = X(q.speechEnd) > 78 ? [sp - sw, sp] : [sp, sp + sw];
    }
    for (let t = Math.ceil(a0 / step) * step; t <= a0 + span; t += step) {
      const lab = t >= 1000 ? (t / 1000).toFixed(step % 1000 ? (step % 500 ? 2 : 1) : 0) + " s" : t + " ms";
      const tp = (X(t) / 100) * w, la = tp + 2, lb = tp + 6 + lab.length * 6.6;
      const hid = spkA != null && lb > spkA - 2 && la < spkB + 2;
      tickHtml += `<div class="tick" style="left:${X(t)}%"><span${hid ? ' class="hid"' : ""}>${lab}</span></div>`;
    }
    if (axis._h !== tickHtml) { axis.innerHTML = tickHtml; axis._h = tickHtml; }

    // rows
    const body = $("#tlBody");
    const want = rows.map((r) => r.key);
    for (const [k, n] of tlNodes) if (!want.includes(k)) { n.remove(); tlNodes.delete(k); }
    rows.forEach((r) => {
      let n = tlNodes.get(r.key);
      if (!n) { n = makeRow(r); tlNodes.set(r.key, n); }
      updateRow(n, r, X, q);
    });
    const cur = Array.from(body.children);
    if (cur.length !== rows.length || cur.some((c, i) => c !== tlNodes.get(want[i]))) {
      want.forEach((k) => body.appendChild(tlNodes.get(k)));
    }

    // markers
    const marks = [];
    if (run.running) marks.push({ cls: "now", at: serverNow() });
    if (q.speechEnd != null) marks.push({ cls: "spk", at: q.speechEnd, label: "you stopped talking", top: true });
    // endpoint markers: one dashed line each, one shared label when they're close (saves a label lane)
    const eps = [];
    for (const m of run.modes) {
      const at = q.endpointMs[m];
      if (at == null) continue;
      const lab = q.epBy[m] === "semantic" ? "semantic" : "Arc";
      if (!eps.some((x) => x.lab === lab && Math.abs(x.at - at) < 60)) eps.push({ at, lab });
    }
    eps.sort((a, b) => a.at - b.at);
    const close = eps.length > 1 && eps.at(-1).at - eps[0].at < 400;
    eps.forEach((x, i) => marks.push({ cls: "ep dash", at: x.at,
      label: close ? (i === 0 ? eps.map((y) => y.lab).join(" + ") + " endpoint" : null) : x.lab + " endpoint" }));
    for (const o of q.old) if (o.ep != null) marks.push({ cls: "ep dash", at: o.ep, label: o.by === "semantic" ? "semantic endpoint (early)" : "endpoint (early)" });
    for (const m of run.modes) {
      const met = q.metric[m];
      let at = null;
      const se = q.se[m];
      if (met && se != null) at = se + met.ready_ms;
      else if (q.firstMs[m] != null) at = q.firstMs[m];
      if (at != null) marks.push({ cls: m === "headstart" ? "hs" : "bl", at, label: `${MODE_NAME[m]} ready ${at - (q.speechEnd ?? at) >= 0 ? "+" : ""}${fmtS(at - (q.speechEnd ?? at))}`, mode: m });
    }
    const mk = $("#tlMarks");
    const tw = mk.clientWidth || 800;
    // de-collide bottom labels
    const bottoms = marks.filter((m) => m.label && !m.top).sort((a, b) => a.at - b.at);
    // place each label right or left of its marker, on the lowest level where it doesn't overlap another label
    const occ = [];                  // per level: [[a, b], ...] occupied px intervals
    const fits = (lv, a, b) => !(occ[lv] || []).some(([x, y]) => a < y + 6 && b > x - 6);
    for (const m of bottoms) {
      const px = (X(m.at) / 100) * tw;
      const w = m.label.length * 6 + 14;
      const opts = [[px, px + w, false], [px - w, px, true]].filter(([a, b]) => a >= -4 && b <= tw + 4);
      if (!opts.length) opts.push(px > tw / 2 ? [px - w, px, true] : [px, px + w, false]);
      let lv = 0, pick = null;
      for (; lv < 6 && !pick; lv++) pick = opts.find(([a, b]) => fits(lv, a, b));
      lv--;
      pick = pick || opts[0];
      (occ[lv] = occ[lv] || []).push([pick[0], pick[1]]);
      m.level = lv;
      m.flip = pick[2];
    }
    const ends = occ;
    const band = 12 + Math.max(1, ends.length) * 16;
    const tlEl = $("#tl");
    if (tlEl._band !== band) { tlEl.style.paddingBottom = band + "px"; tlEl._band = band; }
    let html = "";
    for (const m of marks) {
      const left = X(m.at);
      if (left < -1 || left > 101) continue;
      const right = (m.top ? left > 78 : m.flip) ? " right" : "";
      const lab = m.label ? `<span style="${m.top ? "" : `top:auto;bottom:${8 + (m.level || 0) * 16}px`}">${esc(m.label)}</span>` : "";
      html += `<div class="mark ${m.cls}${right}" style="left:${left}%">${lab}</div>`;
    }
    // "stopped -> ready" gap lines along the bottom
    run.modes.forEach((m, i) => {
      const se = q.speechEnd;
      if (q.se[m] == null || se == null) return;
      const met = q.metric[m];
      const end = met ? q.se[m] + met.ready_ms : (q.firstMs[m] ?? (run.running ? serverNow() : null));
      if (end == null) return;
      html += `<div class="gap ${m === "headstart" ? "hs" : "bl"}" style="left:${X(se)}%;width:${Math.max(0, X(end) - X(se))}%;bottom:${1 + i * 3}px"></div>`;
    });
    if (mk._h !== html) { mk.innerHTML = html; mk._h = html; }
  }

  function makeRow(r) {
    if (r.kind === "sec") {
      const n = el("div", "tl-sec " + (r.mode === "headstart" ? "hs" : "bl"));
      n.innerHTML = `<div class="tl-label"><i class="dot ${r.mode === "headstart" ? "hs" : "bl"}"></i>${MODE_NAME[r.mode]}</div><div class="tl-track"></div>`;
      return n;
    }
    const n = el("div", "tl-row" + (r.kind === "speech" ? " speech" : ""));
    const lab = el("div", "tl-label");
    const track = el("div", "tl-track");
    n.append(lab, track);
    if (r.kind === "speech") { lab.textContent = "speech"; }
    else {
      const bar = el("div", "bar");
      bar.appendChild(el("span", "bt"));
      track.appendChild(bar);
    }
    return n;
  }

  function updateRow(n, r, X, q) {
    if (r.kind === "sec") return;
    if (r.kind === "speech") {
      const track = n.lastChild;
      const tw = track.clientWidth || 600;
      const sig = q.words.map((w) => w.w + w.t0 + w.t1).join("|") + "@" + q.axStart + ":" + Math.round(q.axEnd) + ":" + tw;
      if (track._sig === sig) return;
      track._sig = sig;
      // words are placed at their spoken times but never overlap: each chip starts at least 3 px after the previous
      // one ends, is clipped to the next word's start, and drops its text when too narrow to read
      const ws = q.words.filter((w) => w.t0 != null);
      const px = (ms) => (X(ms) / 100) * tw;
      let prevEnd = -1e9, html = "";
      ws.forEach((w, i) => {
        const a = Math.max(px(w.t0 * 1000), prevEnd + 3);
        const nextA = i + 1 < ws.length ? px(ws[i + 1].t0 * 1000) : Infinity;
        let b = Math.max(px((w.t1 ?? w.t0 + 0.2) * 1000), a + 4);
        b = Math.min(b, Math.max(a + 4, nextA - 3));
        const wd = b - a;
        html += `<div class="sw${wd < 16 ? " tiny" : ""}" style="left:${a.toFixed(1)}px;width:${wd.toFixed(1)}px" title="${esc(w.w)}">${wd < 16 ? "" : esc(w.w)}</div>`;
        prevEnd = b;
      });
      track.innerHTML = html;
      return;
    }
    const lab = n.firstChild, bar = n.lastChild.firstChild, bt = bar.firstChild;
    let laneCls, labelLane, labelQ, state, info = "";
    if (r.kind === "answer") {
      const base = r.mode === "baseline";
      laneCls = "ans" + (base ? " final" : "");
      labelLane = "answer";
      labelQ = r.waiting ? (base ? "searching…" : "ranking…") : (r.live ? "streaming" : "answer");
      state = r.live ? "live" : "done";
      if (!r.waiting && r.first != null) info = `first token +${Math.round(r.first - r.t0)} ms`;
      if (r.superseded) { state = "stale"; labelQ = "early answer, superseded"; info = "speaker kept talking"; }
    } else {
      const f = r.f;
      const isFinal = f.lane.startsWith("final-");
      laneCls = f.lane.replace("final-", "") + (isFinal ? " final" : "");
      labelLane = f.lane.replace("final-", "");
      labelQ = f.query;
      state = f.state === "fired" ? "live" : f.state;
      if (f.state === "fired") info = `${Math.round((r.t1 ?? f.t0) - f.t0)} ms…`;
      else if (f.state === "reused") info = "reused at endpoint";
      else if (f.state === "unused") info = "not needed";
      else if (f.state === "stale") info = f.staleAt != null && f.t1 != null && f.staleAt > f.t1 + 5 ? `stale @${f.staleAt}` : "cancelled: meaning changed";
      else if (f.state === "cancelled") info = "cancelled";
      else if (f.state === "error") info = "error";
      else if (f.took != null) info = `${f.took} ms · ${f.n ?? 0} docs`;
      if (f.used) info += " · fed answer";
      lab.title = [f.query, f.top && f.top.length ? "top: " + f.top.join(" | ") : "", f.error || ""].filter(Boolean).join("\n");
    }
    const labHtml = `<span class="lane ${laneCls}">${esc(labelLane)}</span><span class="q">${esc(labelQ)}</span>`;
    if (lab._h !== labHtml) { lab.innerHTML = labHtml; lab._h = labHtml; }
    n.className = `tl-row st-${state} m-${r.mode}${r.first ? " first" : ""}`;
    const cls = `bar ${laneCls} ${state}${r.f && r.f.used ? " used" : ""}`;
    if (bar.className !== cls) bar.className = cls;
    const left = X(r.t0), width = Math.max(0.35, X(r.t1) - X(r.t0));
    bar.style.left = left + "%";
    bar.style.width = width + "%";
    if (bt.textContent !== info) bt.textContent = info;
    const flip = left + width > 80;
    if (bt._flip !== flip) { bt.style.left = flip ? "auto" : ""; bt.style.right = flip ? "calc(100% + 6px)" : ""; bt._flip = flip; }
    // first-token tick on answer rows
    let tm = bar.querySelector(".tickmark");
    if (r.kind === "answer" && r.first != null && r.t1 > r.t0) {
      if (!tm) { tm = el("i", "tickmark"); bar.appendChild(tm); }
      tm.style.left = ((r.first - r.t0) / (r.t1 - r.t0)) * 100 + "%";
    } else if (tm) tm.remove();
  }

  // ------------------------------------------------------------------ animation loop
  let looping = false;
  function loop() {
    if (looping) return;
    looping = true;
    let tail = 0;
    const step = () => {
      renderTimeline();
      tickWatches();
      if (run && run.running) tail = 0; else tail++;
      if (tail < 30) requestAnimationFrame(step); else looping = false;
    };
    requestAnimationFrame(step);
  }
  addEventListener("resize", () => { if (run) loop(); });

  // ------------------------------------------------------------------ replay
  let questions = [], lastReplay = null;
  const player = $("#player");
  // /api/questions only serves eval/questions.jsonl; the local-moments set (index/questions_local.jsonl) is mirrored
  // here and shown when its audio exists.
  const LOCAL_QS = [
    ["local_01.wav", "When did the announcer call Hunter's three in the game?"],
    ["local_02.wav", "What did the judge say about the objection in the hearing?"],
    ["local_03.wav", "In the call, where did they agree to meet halfway on the discount?"],
    ["local_04.wav", "What did the commentator say about Lonzo Ball in the game?"],
    ["local_05.wav", "In the hearing, where was the gun found?"],
    ["local_06.wav", "What did the judge ask about the officers waiting fifteen seconds?"],
    ["local_07.wav", "In the meeting, what did the teacher say Samuel won?"],
    ["local_08.wav", "What did the commentator say about who won the debate?"],
    ["local_09.wav", "In the recording, what did Chris Christie say about debate prep with Trump?"],
    ["local_10.wav", "In the clip, when did they say team USA won bronze?"],
  ].map(([file, text]) => ({ file, text, kind: "local" }));
  const qsets = { web: [], local: [], conv: [] };
  let qset = store.get("hs-qset") || "web";
  async function loadQuestions() {
    let all = [];
    try { all = await (await fetch("/api/questions", { cache: "no-store" })).json(); } catch { all = []; }
    if (!Array.isArray(all)) all = [];
    qsets.web = all.filter((q) => q.kind !== "local");
    qsets.local = all.filter((q) => q.kind === "local");
    if (!qsets.local.length) {
      try {
        const r = await fetch("/audio/" + LOCAL_QS[0].file, { headers: { Range: "bytes=0-1" }, cache: "no-store" });
        if (r.ok) qsets.local = LOCAL_QS;
      } catch { /* no local set */ }
    }
    try {
      const c = await (await fetch("/api/conversations", { cache: "no-store" })).json();
      qsets.conv = Array.isArray(c) ? c.filter((r) => Array.isArray(r.files) && r.files.length) : [];
    } catch { qsets.conv = []; }
    for (const b of $("#qtabs").querySelectorAll("button")) b.hidden = !(qsets[b.dataset.set] || []).length;
    $("#qtabs").hidden = !qsets.local.length && !qsets.conv.length;
    if (!(qsets[qset] || []).length) qset = "web";
    paintQuestions();
  }
  $("#qtabs").addEventListener("click", (e) => {
    const b = e.target.closest("button[data-set]");
    if (!b) return;
    qset = b.dataset.set;
    store.set("hs-qset", qset);
    paintQuestions();
  });
  function paintQuestions() {
    const list = $("#qlist");
    questions = qsets[qset] || [];
    for (const b of $("#qtabs").querySelectorAll("button")) b.classList.toggle("on", b.dataset.set === qset);
    list.innerHTML = "";
    if (!questions.length) {
      list.innerHTML = '<li class="empty">No spoken questions yet. The eval set (<code>eval/questions.jsonl</code>) will appear here; or type a file name below.</li>';
      return;
    }
    questions.forEach((q, i) => {
      const li = el("li");
      const b = el("button");
      b.dataset.i = String(i);
      const isConv = Array.isArray(q.files);
      const meta = isConv ? `${q.files.length} turns · one run` : [q.category || q.kind || q.type, q.route, q.dur_s ? `${(+q.dur_s).toFixed(1)} s` : null].filter(Boolean).join(" · ");
      const text = isConv ? (q.turns || q.files).map((t) => esc(t)).join('<span class="turn-sep">then</span>') : esc(q.text || q.question || q.file);
      b.innerHTML = `<span class="n">${i < 9 ? i + 1 : "·"}</span><span class="qt">${text}${meta ? `<span class="qm">${esc(meta)}</span>` : ""}</span>`;
      const key = isConv ? q.id || q.files.join("+") : q.file;
      b.dataset.file = key;
      if (lastReplay && lastReplay.key === key) b.classList.add("on");
      b.addEventListener("click", () => playItem(q, b));
      li.appendChild(b);
      list.appendChild(li);
    });
  }

  function playItem(item, btn) {
    if (Array.isArray(item.files)) replayConversation(item, btn);
    else replay(item.file, btn);
  }

  // A conversation is several wavs in ONE run (the server keeps memory within a run). The server's replay takes one
  // file, so the browser streams the PCM itself, exactly like the mic: start, real-time 100 ms Float32 frames, stop.
  let streamToken = 0;
  async function decode16k(file) {
    const buf = await (await fetch("/audio/" + encodeURIComponent(file), { cache: "force-cache" })).arrayBuffer();
    const ctx = new OfflineAudioContext(1, 16000, 16000);
    const ab = await ctx.decodeAudioData(buf);             // resampled to 16 kHz by the context
    if (ab.numberOfChannels === 1) return ab.getChannelData(0);
    const out = new Float32Array(ab.length);
    for (let c = 0; c < ab.numberOfChannels; c++) { const d = ab.getChannelData(c); for (let i = 0; i < d.length; i++) out[i] += d[i] / ab.numberOfChannels; }
    return out;
  }
  async function replayConversation(conv, btn, gapS = 2.5) {
    if (mic.on) stopMic(true);
    const my = ++streamToken;
    const modes = modesFor(modeSel);
    document.querySelectorAll(".qlist button.on").forEach((n) => n.classList.remove("on"));
    if (btn) btn.classList.add("on");
    lastReplay = { item: conv, btn, key: conv.id || conv.files.join("+") };
    try { player.pause(); } catch { /* ignore */ }
    let pcm;
    try { pcm = await Promise.all(conv.files.map(decode16k)); } catch (err) { toast("Could not load conversation audio: " + err.message, true); return; }
    if (my !== streamToken) return;
    if (!send({ t: "start", modes })) return;
    startRun("conversation", modes);
    run.history = [];
    paintHistory();
    const F = 1600, t0 = performance.now();
    let sent = 0;                                             // samples sent so far
    const pace = async (n) => {                               // wait until n samples' worth of real time has passed
      const target = t0 + (n / 16000) * 1000;
      while (performance.now() < target - 2) await new Promise((r) => setTimeout(r, Math.min(20, target - performance.now())));
    };
    const sendFrame = (f) => { if (ws && ws.readyState === 1) ws.send(f.buffer); sent += f.length; };
    const silence = async (sec) => { for (let k = 0; k < Math.round(sec * 10); k++) { if (my !== streamToken) return; await pace(sent + F); sendFrame(new Float32Array(F)); } };
    for (let ti = 0; ti < pcm.length; ti++) {
      if (my !== streamToken) return;
      if (ti > 0) await silence(gapS);
      if (my !== streamToken) return;
      player.src = "/audio/" + encodeURIComponent(conv.files[ti]);
      player.play().catch(() => {});
      const x = pcm[ti];
      for (let i = 0; i < x.length; i += F) {
        if (my !== streamToken) return;
        await pace(sent + F);
        sendFrame(x.slice(i, i + F));
      }
    }
    await silence(1.2);                                       // trailing pause so Arc endpoints like a live mic
    if (my !== streamToken) return;
    send({ t: "stop" });
  }

  function replay(file, btn) {
    if (!file) return;
    streamToken++;                                            // abort any conversation still streaming
    if (mic.on) stopMic(true);
    const modes = modesFor(modeSel);
    document.querySelectorAll(".qlist button.on").forEach((n) => n.classList.remove("on"));
    if (btn) btn.classList.add("on");
    lastReplay = { item: { file }, btn, key: file };
    try { player.pause(); } catch { /* ignore */ }
    player.src = "/audio/" + encodeURIComponent(file);
    if (!send({ t: "replay", file, modes })) return;
    startRun("replay", modes);
    player.currentTime = 0;
    player.play().catch(() => toast("Audio playback blocked by the browser; the race still runs", false));
  }
  $("#adhocForm").addEventListener("submit", (e) => {
    e.preventDefault();
    const f = $("#adhocFile").value.trim();
    if (f) replay(f, null);
  });

  // ------------------------------------------------------------------ microphone
  const mic = { on: false, ctx: null, stream: null, node: null, buf: [], acc: 0, cnt: 0, frac: 0, ratio: 3, level: 0 };
  const WORKLET = `class Cap extends AudioWorkletProcessor{process(i){const c=i[0]&&i[0][0];if(c)this.port.postMessage(c.slice(0));return true}}registerProcessor('hs-cap',Cap);`;

  async function startMic() {
    streamToken++;
    if (!navigator.mediaDevices?.getUserMedia) { toast("Microphone not available here; use Replay", true); return; }
    if (!wsOpen) { toast("Not connected to the server yet", true); return; }
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true } });
    } catch (err) { toast("Microphone permission denied; use Replay", true); return; }
    try { player.pause(); } catch { /* ignore */ }
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const src = ctx.createMediaStreamSource(stream);
    const sink = ctx.createGain();
    sink.gain.value = 0;
    sink.connect(ctx.destination);
    Object.assign(mic, { on: true, ctx, stream, buf: [], acc: 0, cnt: 0, frac: 0, ratio: ctx.sampleRate / 16000 });
    const modes = modesFor(modeSel);
    send({ t: "start", modes });
    startRun("mic", modes);
    document.querySelectorAll(".qlist button.on").forEach((n) => n.classList.remove("on"));
    try {
      const url = URL.createObjectURL(new Blob([WORKLET], { type: "application/javascript" }));
      await ctx.audioWorklet.addModule(url);
      const node = new AudioWorkletNode(ctx, "hs-cap");
      node.port.onmessage = (e) => pushPcm(e.data);
      src.connect(node); node.connect(sink);
      mic.node = node;
    } catch {
      const node = ctx.createScriptProcessor(4096, 1, 1);
      node.onaudioprocess = (e) => pushPcm(e.inputBuffer.getChannelData(0).slice(0));
      src.connect(node); node.connect(sink);
      mic.node = node;
    }
    paintMic();
  }

  // box-filter decimation to 16 kHz (works for 44.1k and 48k), then 100 ms Float32 frames
  function pushPcm(x) {
    if (!mic.on) return;
    let sq = 0;
    for (let i = 0; i < x.length; i++) {
      const s = x[i];
      sq += s * s;
      mic.acc += s; mic.cnt++; mic.frac += 1;
      if (mic.frac >= mic.ratio) {
        mic.buf.push(mic.acc / mic.cnt);
        mic.acc = 0; mic.cnt = 0; mic.frac -= mic.ratio;
      }
    }
    mic.level = 0.8 * mic.level + 0.2 * Math.sqrt(sq / Math.max(1, x.length));
    while (mic.buf.length >= 1600) {
      const f = new Float32Array(mic.buf.splice(0, 1600));
      if (ws && ws.readyState === 1) ws.send(f.buffer);
    }
    const ring = $("#micRing");
    ring.style.transform = `scale(${1 + Math.min(0.35, mic.level * 6)})`;
  }

  function stopMic(sendStop = true) {
    if (!mic.on) return;
    mic.on = false;
    if (mic.buf.length && ws && ws.readyState === 1) ws.send(new Float32Array(mic.buf.splice(0)).buffer);
    try { mic.node && mic.node.disconnect(); } catch { /* ignore */ }
    try { mic.stream.getTracks().forEach((t) => t.stop()); } catch { /* ignore */ }
    try { mic.ctx.close(); } catch { /* ignore */ }
    if (sendStop) send({ t: "stop" });
    $("#micRing").style.transform = "";
    paintMic();
  }
  function paintMic() {
    $("#micBtn").classList.toggle("live", mic.on);
    $("#micLabel").textContent = mic.on ? "Stop" : "Ask";
  }
  $("#micBtn").addEventListener("click", () => (mic.on ? stopMic(true) : startMic()));

  // ------------------------------------------------------------------ keyboard
  addEventListener("keydown", (e) => {
    if (e.target.closest("input, textarea") || e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.code === "Space") { e.preventDefault(); mic.on ? stopMic(true) : startMic(); }
    else if (/^[1-9]$/.test(e.key)) {
      const i = +e.key - 1;
      const b = document.querySelector(`.qlist button[data-i="${i}"]`);
      if (b && questions[i]) playItem(questions[i], b);
    } else if (e.key === "r" || e.key === "R") { if (lastReplay) playItem(lastReplay.item, lastReplay.btn); }
  });

  // ------------------------------------------------------------------ results (runs/ab_latest.json)
  const num = (v) => (typeof v === "number" && isFinite(v) ? v : null);
  function readyOf(x) {
    if (x == null) return null;
    if (typeof x === "number") return x;
    return num(x.ready_ms) ?? num(x.ready) ?? null;
  }
  function pct(arr, p) {
    const a = arr.filter((v) => v != null).sort((x, y) => x - y);
    if (!a.length) return null;
    const i = (a.length - 1) * p, lo = Math.floor(i), hi = Math.ceil(i);
    return a[lo] + (a[hi] - a[lo]) * (i - lo);
  }
  function normResults(d) {
    if (!d || (typeof d === "object" && !Array.isArray(d) && !Object.keys(d).length)) return null;
    const rows = Array.isArray(d) ? d : (d.rows || d.results || d.questions || d.items || []);
    const by = new Map();
    const get = (id, text) => { if (!by.has(id)) by.set(id, { id, text: text || id, hs: [], bl: [] }); const r = by.get(id); if (text && r.text === id) r.text = text; return r; };
    for (const r of rows) {
      if (!r || typeof r !== "object") continue;
      const id = r.id ?? r.qid ?? r.file ?? r.question ?? r.text;
      const text = r.text ?? r.question ?? r.transcript ?? null;
      const it = get(String(id), text);
      if (r.headstart !== undefined || r.baseline !== undefined) {
        const h = readyOf(r.headstart), b = readyOf(r.baseline);
        if (h != null) it.hs.push(h); if (b != null) it.bl.push(b);
      } else if (r.mode) {
        const v = readyOf(r);
        if (v != null) (r.mode === "baseline" ? it.bl : it.hs).push(v);
        if (r.correct != null) it[r.mode === "baseline" ? "blOk" : "hsOk"] = !!r.correct;
        if (r.path) it[r.mode === "baseline" ? "blPath" : "hsPath"] = r.path;
      } else {
        const h = num(r.headstart_ready_ms) ?? num(r.hs_ready_ms), b = num(r.baseline_ready_ms) ?? num(r.bl_ready_ms);
        if (h != null) it.hs.push(h); if (b != null) it.bl.push(b);
      }
    }
    const pairs = Array.from(by.values()).map((r) => ({ ...r, h: pct(r.hs, 0.5), b: pct(r.bl, 0.5) })).filter((r) => r.h != null || r.b != null);
    const sum = (d && d.summary) || {};
    const S = (mode, key, arr, p) => num(sum[mode]?.[key]) ?? pct(arr, p);
    const hsAll = pairs.map((r) => r.h), blAll = pairs.map((r) => r.b);
    return {
      pairs,
      hs50: S("headstart", "p50_ready_ms", hsAll, 0.5), hs90: S("headstart", "p90_ready_ms", hsAll, 0.9),
      bl50: S("baseline", "p50_ready_ms", blAll, 0.5), bl90: S("baseline", "p90_ready_ms", blAll, 0.9),
      summary: sum, raw: d,
    };
  }

  async function loadResults() {
    const body = $("#resBody");
    let d;
    try { d = await (await fetch("/api/results", { cache: "no-store" })).json(); } catch { d = null; }
    const R = normResults(d);
    if (!R) { body.innerHTML = '<p class="muted">No results yet. The eval harness writes <code>runs/ab_latest.json</code>.</p>'; $("#resSub").textContent = ""; return; }
    $("#resSub").textContent = R.pairs.length ? `${R.pairs.length} questions · ready ms after the speaker stopped` : "";
    const tile = (lab, v, cls = "", extra = "") => `<div class="tile ${cls}"><label>${lab}</label><div class="v">${v == null ? "—" : fmtS(v)}${extra}</div></div>`;
    // the saving is the PAIRED median (median over questions of baseline − head start), not p50(BL) − p50(HS)
    const P = R.raw && R.raw.paired && typeof R.raw.paired === "object" ? R.raw.paired : {};
    const both = R.pairs.filter((r) => r.h != null && r.b != null);
    const saved = num(P.p50_saved_ms) ?? pct(both.map((r) => r.b - r.h), 0.5);
    const nPair = num(P.n) ?? num(P.n_paired) ?? both.length;
    let fasterN = num(P.headstart_faster) ?? num(P.n_headstart_faster);
    if (fasterN != null && fasterN > 0 && fasterN <= 1 && nPair > 1) fasterN = Math.round(fasterN * nPair);   // a rate
    if (fasterN == null && both.length) fasterN = both.filter((r) => r.h < r.b).length;
    const fasterTxt = fasterN != null && nPair ? ` <small>faster on ${fasterN}/${nPair}</small>` : "";
    let html = `<div class="res-tiles">${tile("Head Start p50", R.hs50, "hs")}${tile("Baseline p50", R.bl50)}${tile("median saving per question", saved, "hs", fasterTxt)}${tile("p90 HS / BL", R.hs90, "", ` <small>/ ${fmtS(R.bl90)}</small>`)}</div>`;
    const acc = (m) => num(R.summary[m]?.accuracy);
    if (acc("headstart") != null || acc("baseline") != null) {
      const a = (v) => (v == null ? "—" : Math.round(v * 100) + "%");
      const n = R.summary.headstart?.n ?? R.summary.baseline?.n ?? R.pairs.length;
      html += `<p class="res-note">answer correct: Head Start <b>${a(acc("headstart"))}</b> · Baseline <b>${a(acc("baseline"))}</b> (n=${n})` +
        (R.summary.headstart?.draft_reuse_rate != null ? ` · draft reuse <b>${Math.round(R.summary.headstart.draft_reuse_rate * 100)}%</b>` : "") +
        (R.summary.headstart?.mean_web_calls != null ? ` · web calls/question <b>${R.summary.headstart.mean_web_calls}</b> vs <b>${R.summary.baseline?.mean_web_calls ?? "—"}</b>` : "") +
        (R.raw?.created ? ` · <span class="muted">${esc(R.raw.created)}</span>` : "") + `</p>`;
    }
    if (R.pairs.length) html += chart(R.pairs);
    else html += `<details><summary class="muted">raw ab_latest.json</summary><pre class="raw">${esc(JSON.stringify(R.raw, null, 2).slice(0, 6000))}</pre></details>`;
    body.innerHTML = html;
  }

  // dumbbell chart: one row per question, Head Start vs Baseline ready ms
  function chart(pairs) {
    const rows = pairs.slice().sort((a, b) => (b.b ?? b.h) - (a.b ?? a.h));
    const W = 960, L = 330, R = 70, rh = 24, top = 26, H = top + rows.length * rh + 8;
    const maxV = Math.max(...rows.flatMap((r) => [r.h, r.b]).filter((v) => v != null), 500);
    const step = NICE.find((s) => maxV / s <= 7) || 10000;
    const xmax = Math.ceil(maxV / step) * step;
    const x = (v) => L + (v / xmax) * (W - L - R);
    let g = "";
    for (let t = 0; t <= xmax; t += step) {
      g += `<line class="grid" x1="${x(t)}" x2="${x(t)}" y1="${top - 6}" y2="${H - 4}"/><text x="${x(t)}" y="${top - 12}" text-anchor="middle" class="num">${t >= 1000 ? (t / 1000).toFixed(step < 1000 ? 1 : 0) + " s" : t + " ms"}</text>`;
    }
    rows.forEach((r, i) => {
      const y = top + i * rh + rh / 2;
      const label = r.text.length > 52 ? r.text.slice(0, 50) + "…" : r.text;
      g += `<g class="row"><title>${esc(r.text)}\nHead Start ${fmtS(r.h)}${r.hsPath ? " (" + r.hsPath + ")" : ""} · Baseline ${fmtS(r.b)}</title>`;
      g += `<text x="${L - 12}" y="${y + 4}" text-anchor="end" class="lbl">${esc(label)}</text>`;
      if (r.h != null && r.b != null) g += `<line class="conn" x1="${x(r.h)}" x2="${x(r.b)}" y1="${y}" y2="${y}"/>`;
      if (r.b != null) g += `<circle class="pb${r.blOk === false ? " bad" : ""}" cx="${x(r.b)}" cy="${y}" r="5.5"/>`;
      if (r.h != null) g += `<circle class="ph${r.hsOk === false ? " bad" : ""}" cx="${x(r.h)}" cy="${y}" r="5.5"/>`;
      const far = Math.max(r.h ?? 0, r.b ?? 0);
      if (r.h != null && r.b != null) g += `<text x="${x(far) + 10}" y="${y + 4}" class="num">${r.b >= r.h ? "saved " + fmtS(r.b - r.h) : "lost " + fmtS(r.h - r.b)}</text>`;
      g += `</g>`;
    });
    return `<div class="res-legend"><span><i style="background:var(--hs)"></i>Head Start</span><span><i style="background:var(--bl)"></i>Baseline</span><span class="muted">answer ready after the last spoken word · hollow = wrong answer</span></div>
      <div class="res-chart"><svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Ready time per question, Head Start vs Baseline">${g}</svg></div>`;
  }
  $("#resReload").addEventListener("click", () => { loadResults(); loadQuestions(); });

  // ------------------------------------------------------------------ boot
  connect();
  pollHealth();
  loadQuestions();
  loadResults();
  setInterval(loadResults, 30000);
  window.__hs = { get run() { return run; } };   // for debugging in the console
})();
