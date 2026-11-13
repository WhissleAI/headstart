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
      wordsDirty: true,
    };
  }
  function startRun(kind, modes) {
    run = { modes, kind, running: true, clientT0: performance.now(), base: null, q: newQuestion() };
    resetView();
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
      else if (e.t === "run_done" && run) { run.running = false; finishRun(); }
      return;
    }
    if (!run) return;
    if (typeof e.ms === "number") syncClock(e.ms);
    if (e.mode === "ear") return onEar(e);
    const mode = e.mode;
    let q = run.q;
    switch (e.t) {
      case "words": {
        if (!acceptWords(q, mode, e)) return;
        // a new question in the same run (live mic): start a fresh view once the previous one has answered
        // (words that start after the last answered speech end; not the same utterance continuing)
        const t0w = e.words && e.words.length ? (e.words[0].t0 ?? 0) * 1000 : null;
        if (q.speechEnd != null && t0w != null && t0w > q.speechEnd + 50 && Object.keys(q.metric).length) {
          run.q = q = newQuestion(Math.max(0, t0w - 400));
          resetView(true);
        }
        q.prevWords = q.words;
        q.words = e.words || [];
        q.revFrom = e.from;
        renderWords(q);
        break;
      }
      case "frame":
        if (mode !== q.wordsMode) return;
        q.frame = e;
        q.spans = e.spans || [];
        renderSpans(q);
        renderFrame(e);
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
        markUsed(q);
        break;
      case "answer":
        if (q.firstMs[mode] == null) q.firstMs[mode] = e.ms;
        q.answer[mode] = (q.answer[mode] || "") + (e.delta || "");
        q.lastAnsMs = e.ms;
        renderAnswer(mode);
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
        markUsed(q);
        break;
    }
  }

  function onEar(e) {
    const s = $("#earState");
    if (e.kind === "partial") { s.textContent = "hearing…"; s.className = "ear live"; }
    else if (e.kind === "final") {
      s.textContent = `utterance closed${e.reason ? " · " + e.reason : ""}`; s.className = "ear";
      // sessions don't emit a words event for the ear's final revision; show it from the ear's own text
      const q = run.q;
      const toks = String(e.text || "").split(/\s+/).map((w) => w.replace(/[.?!,]+$/, "")).filter(Boolean);
      if (toks.length && toks.join(" ") !== q.words.map((w) => w.w).join(" ")) {
        q.words = toks.map((w, i) => ({ w, t0: q.words[i]?.t0 ?? null, t1: q.words[i]?.t1 ?? null }));
        q.spans = [];                  // tagger spans index the session's words, not this re-tokenized text
        renderWords(q);
      }
    }
    else if (e.kind === "arc_command_complete") { s.textContent = `Arc: question complete (p ${(+e.p || 0).toFixed(2)})`; s.className = "ear"; }
  }

  // a speculative fire "fed the answer" if one of its top titles is among the final sources (spec events carry
  // titles, not doc ids, so this is title-based)
  function markUsed(q) {
    for (const f of q.fires.values()) {
      const src = q.sources[f.mode];
      if (!src || !f.top || f.lane === "draft") continue;
      const titles = new Set(src.map((d) => d.title));
      f.used = f.top.some((t) => titles.has(t)) && (f.state === "done");
    }
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
  function renderFrame(f, final = false) {
    const badges = $("#frameBadges");
    badges.innerHTML = "";
    if (!f) {
      $("#fIntent").textContent = "—"; $("#fSlots").textContent = "—"; $("#fRoute").textContent = "—"; $("#fQuery").textContent = "—";
      return;
    }
    const p = Math.max(0, Math.min(1, +f.p || 0));
    $("#fIntent").innerHTML = f.intent
      ? `<span class="nm" title="${esc(f.intent)}">${esc(stripIntent(f.intent))}</span><span class="pbar"><i style="width:${Math.round(p * 100)}%"></i></span><span class="pv">${p.toFixed(2)}</span>`
      : `<span class="muted">listening for intent…</span>`;
    const slots = f.slots || [];
    $("#fSlots").innerHTML = slots.length
      ? slots.map(([t, v]) => `<span class="chip ${/^(cardinal|ordinal|money|percent|date|time|quantity|numeral|year|number)/.test(t) ? "number" : ""}"><small>${esc(t.replace(/_/g, " "))}</small>${esc(v)}</span>`).join("")
      : `<span class="muted">none yet</span>`;
    const r = f.route || "web";
    $("#fRoute").innerHTML = `<span class="route"><span class="web ${r === "web" || r === "both" ? "on" : ""}">web</span><span class="local ${r === "local" || r === "both" ? "on" : ""}">local</span></span>`;
    $("#fQuery").textContent = f.query || "—";
    if (f.corrected) badges.appendChild(el("span", "badge corr", "corrected"));
    if (f.command_complete) badges.appendChild(el("span", "badge cc", "complete"));
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
      const meta = [d.lane, d.source, d.t != null ? `@${Math.round(d.t)}s` : null, d.score != null ? `cos ${(+d.score).toFixed(2)}` : null].filter(Boolean).join(" · ");
      a.innerHTML = `<div class="st"><span class="sn">[${i + 1}]</span><span class="tt">${esc(d.title || "(untitled)")}</span></div><div class="sm">${esc(meta)}</div>`;
      box.appendChild(a);
    });
  }

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
      st.push(`spec fires <b>${m.spec_fires ?? 0}</b>`, `fed answer <b>${m.spec_used ?? 0}</b>`, `stale <b>${m.stale ?? 0}</b>`,
        `drafts <b>${m.drafts ?? 0}</b>${m.draft_reused ? " (reused)" : ""}`);
    }
    st.push(`web calls <b>${m.web_calls ?? 0}</b>`, `sources <b>${fmtS(m.sources_ms)}</b>`, `done <b>${fmtS(m.done_ms)}</b>`);
    if (m.endpoint_lag_ms != null) st.push(`endpoint lag <b>${m.endpoint_lag_ms} ms</b>`);
    pk(mode, "stats").innerHTML = st.map((s) => `<span>${s}</span>`).join("");
    if (m.error && !q.answer[mode]) { q.error[mode] = m.error; renderAnswer(mode); }
    // delta when both modes have finished
    const hs = q.metric.headstart, bl = q.metric.baseline;
    if (hs && bl) {
      for (const o of ["headstart", "baseline"]) { if (o !== mode) { pk(o, "big").textContent = fmtS(readyAdj(q, o)); } }
      const d = readyAdj(q, "baseline") - readyAdj(q, "headstart");
      const dh = pk("headstart", "delta"), db = pk("baseline", "delta");
      if (d >= 0) {
        dh.textContent = `${fmtS(d)} faster`;
        dh.className = "delta";
        db.textContent = `+${fmtS(d)} behind`;
        db.className = "delta slow";
        pane("headstart").classList.add("winner");
      } else {
        db.textContent = `${fmtS(-d)} faster`;
        db.className = "delta";
        dh.textContent = `+${fmtS(-d)} behind`;
        dh.className = "delta slow";
      }
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
      rows.push({ key: "sec:" + mode, kind: "sec", mode });
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
    for (let t = Math.ceil(a0 / step) * step; t <= a0 + span; t += step) {
      tickHtml += `<div class="tick" style="left:${X(t)}%"><span>${t >= 1000 ? (t / 1000).toFixed(step % 1000 ? (step % 500 ? 2 : 1) : 0) + " s" : t + " ms"}</span></div>`;
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
    const seen = new Set();
    for (const m of run.modes) {
      const at = q.endpointMs[m];
      if (at == null) continue;
      const lab = q.epBy[m] === "semantic" ? "semantic endpoint" : "Arc endpoint";
      const k = lab + Math.round(at / 60);
      if (seen.has(k)) continue;
      seen.add(k);
      marks.push({ cls: "ep dash", at, label: lab });
    }
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
    const ends = [];                 // right edge (px) of the last label per level
    for (const m of bottoms) {
      const px = (X(m.at) / 100) * tw;
      const w = m.label.length * 6 + 14;
      const flip = X(m.at) > 78;
      const a = flip ? px - w : px, b = flip ? px : px + w;
      let lv = 0;
      while (ends[lv] != null && ends[lv] > a - 4) lv++;
      ends[lv] = b;
      m.level = lv;
    }
    let html = "";
    for (const m of marks) {
      const left = X(m.at);
      if (left < -1 || left > 101) continue;
      const right = left > 78 ? " right" : "";
      const lab = m.label ? `<span style="${m.top ? "" : `top:auto;bottom:${8 + (m.level || 0) * 18}px`}">${esc(m.label)}</span>` : "";
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
      const sig = q.words.map((w) => w.w + w.t0 + w.t1).join("|") + "@" + q.axStart + ":" + Math.round(q.axEnd);
      if (track._sig === sig) return;
      track._sig = sig;
      track.innerHTML = q.words.filter((w) => w.t0 != null).map((w) => {
        const a = X(w.t0 * 1000), b = X((w.t1 ?? w.t0 + 0.2) * 1000);
        return `<div class="sw" style="left:${a}%;width:${Math.max(0.6, b - a)}%" title="${esc(w.w)}">${esc(w.w)}</div>`;
      }).join("");
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
    n.className = "tl-row st-" + state;
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
  async function loadQuestions() {
    const list = $("#qlist");
    try {
      const r = await fetch("/api/questions", { cache: "no-store" });
      questions = await r.json();
    } catch { questions = []; }
    if (!Array.isArray(questions)) questions = [];
    list.innerHTML = "";
    if (!questions.length) {
      list.innerHTML = '<li class="empty">No spoken questions yet. The eval set (<code>eval/questions.jsonl</code>) will appear here; or type a file name below.</li>';
      return;
    }
    questions.forEach((q, i) => {
      const li = el("li");
      const b = el("button");
      b.dataset.i = String(i);
      const meta = [q.category || q.kind || q.type, q.route, q.dur_s ? `${(+q.dur_s).toFixed(1)} s` : null].filter(Boolean).join(" · ");
      b.innerHTML = `<span class="n">${i < 9 ? i + 1 : "·"}</span><span class="qt">${esc(q.text || q.question || q.file)}${meta ? `<span class="qm">${esc(meta)}</span>` : ""}</span>`;
      b.addEventListener("click", () => replay(q.file, b));
      li.appendChild(b);
      list.appendChild(li);
    });
  }

  function replay(file, btn) {
    if (!file) return;
    if (mic.on) stopMic(true);
    const modes = modesFor(modeSel);
    document.querySelectorAll(".qlist button.on").forEach((n) => n.classList.remove("on"));
    if (btn) btn.classList.add("on");
    lastReplay = { file, btn };
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
      if (b && questions[i]) replay(questions[i].file, b);
    } else if (e.key === "r" || e.key === "R") { if (lastReplay) replay(lastReplay.file, lastReplay.btn); }
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
    const speed = R.hs50 != null && R.bl50 != null ? R.bl50 - R.hs50 : null;
    let html = `<div class="res-tiles">${tile("Head Start p50", R.hs50, "hs")}${tile("Baseline p50", R.bl50)}${tile("p50 saved", speed, "hs", R.hs50 && R.bl50 ? ` <small>${(R.bl50 / Math.max(1, R.hs50)).toFixed(1)}×</small>` : "")}${tile("p90 HS / BL", R.hs90, "", ` <small>/ ${fmtS(R.bl90)}</small>`)}</div>`;
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
