/* Live search results, shared by the main UI (app.js) and film mode (film.js).
   Each side shows the results of its latest landed search, Google-style. While you speak the list changes: a new
   search's results slide in, dropped ones fade out, kept ones move to their new rank (FLIP); a search whose meaning
   went stale strikes its results. At the endpoint the final ranked sources lock in.
   spec "done"/"stale" events carry docs[] (top-5 {id,title,text,url,source,t,lane}); older servers only ids + top titles. */
(() => {
  "use strict";
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const host = (u) => { try { return new URL(u).hostname.replace(/^www\./, ""); } catch { return ""; } };
  function specDocs(e) {
    if (Array.isArray(e.docs) && e.docs.length) return e.docs;
    if (!Array.isArray(e.ids)) return null;
    return e.ids.map((id, i) => ({ id, url: /^https?:\/\//.test(id) ? id : null, title: (e.top || [])[i] || null })).filter((d) => d.title);
  }
  const clip = (t, n) => { t = String(t || "").replace(/\s+/g, " ").trim(); return t.length <= n ? t : t.slice(0, n).replace(/\s+\S*$/, "").replace(/[\s,.;:|–—-]+$/, "") + "…"; };
  const cleanTitle = (t) => String(t || "").replace(/\s+/g, " ").trim().replace(/\s+[|–—-]\s+[^|–—-]{2,40}$/, "").trim();
  const HUES = [158, 210, 32, 265, 340, 190, 12, 120];
  const hue = (s) => { let h = 0; for (const c of String(s)) h = (h * 31 + c.charCodeAt(0)) >>> 0; return HUES[h % HUES.length]; };
  const ctx2d = document.createElement("canvas").getContext("2d");
  const textW = (t, px) => { ctx2d.font = `400 ${px}px Inter, sans-serif`; return ctx2d.measureText(t).width; };
  // one line of text at `px` that fits `w` pixels, cut at a word with an ellipsis (measured, never CSS-clipped)
  function oneLine(t, px, w) {
    if (textW(t, px) * 0.93 <= w) return t;
    let lo = 0, hi = t.length;
    while (lo < hi) { const mid = (lo + hi + 1) >> 1; if (textW(t.slice(0, mid) + "…", px) * 0.93 <= w) lo = mid; else hi = mid - 1; }
    return t.slice(0, lo).replace(/\s+\S*$/, "").replace(/[\s,.;:|–—-]+$/, "") + "…";
  }
  const mmss = (t) => { t = Math.max(0, Math.round(+t)); return `${String(Math.floor(t / 60)).padStart(2, "0")}:${String(t % 60).padStart(2, "0")}`; };
  function dateOf(d) {
    const v = d.published_date || d.published || d.date || (d.meta && (d.meta.published_date || d.meta.date));
    if (!v) return "";
    const t = new Date(v);
    return isNaN(t) ? String(v).slice(0, 10) : t.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });
  }

  // fires: this side's searches [{lane, query, state, t0, t1, docs}] in fire order (drafts are ignored)
  function view({ fires, sourcesMs, sources, query }) {
    const fs = (fires || []).filter((f) => f.lane !== "draft");
    if (sourcesMs != null) return { kind: "final", docs: sources || [], q: query || "" };
    const landed = fs.filter((f) => f.docs && f.docs.length && f.t1 != null);
    const last = landed.at(-1);
    const pending = fs.filter((f) => f.state === "fired" && f !== last).at(-1);
    if (!last) {
      if (pending) return { kind: "searching", docs: [], q: pending.query };
      return { kind: "empty", docs: [] };
    }
    const stale = last.state === "stale" || last.state === "cancelled";
    return { kind: stale ? "stale" : "live", docs: last.docs, q: last.query, next: pending && pending.t0 > last.t0 ? pending.query : null, fire: last };
  }

  // o: {cls: class prefix ("ri" film, "rr" main UI), px: snippet font px, numbered, max, snipLines}
  function itemHtml(d, i, o) {
    const p = o.cls;
    const h = d.url ? host(d.url) : (d.source || d.lane || "");
    const title = clip(cleanTitle(d.title) || h || "(untitled)", o.titleMax || 72);
    const raw = d.text ? String(d.text).replace(/^#+\s*/gm, "").replace(/\s+/g, " ") : "";
    const snip = raw ? (o.snipLines > 1 ? clip(raw, o.snipChars || 220) : oneLine(raw, o.px || 17, o.snipW || 760)) : "";
    const num = o.numbered ? `<span class="${p}-n">[${i + 1}]</span>` : "";
    const ts = d.t != null ? `<span class="${p}-ts">▶ ${mmss(d.t)}</span>` : "";
    const dt = o.dates ? dateOf(d) : "";
    return `<div class="${p}-h">${num}<span class="fav" style="--h:${hue(h)}">${esc((h[0] || "?").toUpperCase())}</span><span class="dom">${esc(h)}</span>${ts}${dt ? `<span class="${p}-d">${esc(dt)}</span>` : ""}</div>` +
      `<div class="${p}-t">${d.url && o.links ? `<a href="${esc(d.url)}" target="_blank" rel="noopener">${esc(title)}</a>` : esc(title)}</div>` + (snip ? `<div class="${p}-s">${esc(snip)}</div>` : "");
  }

  // FLIP: measure old positions, re-render keyed by doc id, animate from old to new. Only whole items that fit are kept.
  function flip(box, docs, opts = {}) {
    const o = { cls: "ri", px: 17, ...opts };
    const sig = docs.map((d) => d.id + "|" + (d.text ? 1 : 0)).join(",") + "|" + box.className + "|" + (o.max || "") + "|" + box.clientHeight + "|" + box.clientWidth;
    if (box._sig === sig) return;
    box._sig = sig;
    const old = new Map();
    for (const n of Array.from(box.children)) {
      if (n.classList.contains("leaving")) { n.remove(); continue; }
      old.set(n.dataset.id, { n, top: n.offsetTop });
    }
    const avail = box.clientHeight || 300;
    if (box.clientWidth) o.snipW = box.clientWidth - 32;
    const want = [];
    for (const d of docs) {
      if (!d || !d.id) continue;
      // same page twice (tracking params, mirrors): one row
      const dk = (d.url ? host(d.url) : "") + "|" + cleanTitle(d.title).toLowerCase();
      if (want.some((x) => x.id === d.id || x._dk === dk)) continue;
      want.push({ ...d, _dk: dk, _i: docs.indexOf(d) });
      if (o.max && want.length >= o.max) break;
    }
    const frag = [];
    for (const d of want) {
      let n = old.get(d.id)?.n;
      const isNew = !n;
      if (isNew) { n = document.createElement("div"); n.className = o.cls; n.dataset.id = d.id; }
      n.dataset.n = String(d._i + 1);
      const html = itemHtml(d, d._i, o);
      if (n._h !== html) { n.innerHTML = html; n._h = html; }
      frag.push({ n, d });
    }
    for (const { n } of frag) box.appendChild(n);
    const keep = [], shown = new Set();
    let used = 0;
    for (const { n, d } of frag) {
      const cs = getComputedStyle(n);
      const h = n.offsetHeight + (parseFloat(cs.marginBottom) || 0);
      if (used + h > avail + 1) { n.remove(); continue; }
      used += h; shown.add(d.id); keep.push(n);
    }
    for (const n of keep) box.appendChild(n);
    for (const [id, x] of old) {
      if (shown.has(id)) continue;
      const n = x.n;
      n.classList.add("leaving");
      n.style.top = x.top + "px";
      box.appendChild(n);
      setTimeout(() => n.remove(), 450);
    }
    for (const n of keep) {
      const x = old.get(n.dataset.id);
      n.classList.remove("entering");
      if (x) {
        const dy = x.top - n.offsetTop;
        if (dy) {
          n.style.transition = "none";
          n.style.transform = `translateY(${dy}px)`;
          void n.offsetWidth;
          n.style.transition = "";
          n.style.transform = "";
        }
      } else {
        n.classList.add("entering");
        void n.offsetWidth;
        requestAnimationFrame(() => n.classList.remove("entering"));
      }
    }
  }

  // the header line above a results list
  function head(v, o = {}) {
    const side = o.mode;
    if (v.kind === "final") return `<span class="lk">✓ ${o.answered ? "answer from" : "ranked"}</span> top results for <b>“${esc(v.q)}”</b>`;
    if (v.kind === "searching") return `<span class="sp"></span>searching <b>“${esc(v.q)}”</b>`;
    if (v.kind === "stale") return `<span class="st">meaning changed</span> <s>“${esc(v.q)}”</s>` + (v.next ? ` → searching <b>“${esc(v.next)}”</b>` : "");
    if (v.kind === "live") return `results for <b>“${esc(v.q)}”</b>` + (v.next ? ` <span class="nx"><span class="sp"></span>now searching “${esc(v.next)}”</span>` : "");
    return side === "baseline" ? (o.endpointed ? "" : "waiting for you to finish…") : "listening…";
  }

  window.HSResults = { specDocs, view, flip, head, host, clip, cleanTitle, hue, oneLine, esc };
})();
