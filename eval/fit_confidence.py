"""Fit the confidence-fusion models on the 80 real Flux traces and measure them with question-grouped cross-validation.

Input: runs/conf_features.jsonl (eval/conf_features.py). Output: server/conf_model.json (with --write).

1) Word model: P(word survives unchanged into the final transcript) from Flux word confidence, revision history
   (changes / updates held), last-word position and the tagger span covering it (score margin, ver). Reported:
   out-of-fold AUC / Brier vs Flux confidence alone.
2) End-of-turn model: P(these words are the whole question) at every Flux TurnInfo message (Update or EagerEndOfTurn)
   from Flux end_of_turn_confidence, pause since the last word, tagger command_complete at the last word (+ score),
   gate / act / slot-end / content, Arc audio cc_p / cc event / for-me, word count and the last word's confidence.
   A policy answers (soft) at the first message whose score >= thr. Replayed per question exactly like
   eval/flux_gate_eval.py: trigger = answer start - last word end (EndOfTurn when no complete message fires),
   premature = soft answers on words that are not the final question (per 100 questions).
   Thresholds are chosen INSIDE the training folds (nested): the reported numbers are out-of-sample.

Run: .venv/bin/python eval/fit_confidence.py [--write]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from server import confidence, frame  # noqa: E402

FEAT = ROOT / "runs" / "conf_features.jsonl"
WORD_F = ["ear", "revs", "age", "is_last", "in_span", "span_margin", "span_ver"]
EOT_F = ["eot", "pause", "nw", "cc_last", "cc_score", "gate", "act_q", "slot_end", "content", "last_content",
         "arc_cc_p", "arc_cc_missing", "arc_cc_recent", "arc_for_me", "eager", "last_conf"]
K = 5


def folds(ids, k=K, seed=0):
    u = sorted(set(ids))
    rng = np.random.default_rng(seed)
    rng.shuffle(u)
    return {q: i % k for i, q in enumerate(u)}


def fit_lr(X, y, C=1.0):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    m = LogisticRegression(C=C, max_iter=2000).fit((X - mu) / sd, y)
    return m, mu, sd


def pred(m, mu, sd, X):
    return m.predict_proba((X - mu) / sd)[:, 1]


def spec_of(m, mu, sd, names):
    return {"features": ["bias"] + names, "w": [float(m.intercept_[0])] + [float(x) for x in m.coef_[0]],
            "mu": [0.0] + [float(x) for x in mu], "sd": [1.0] + [float(x) for x in sd]}


# ------------------------------------------------------------------------------------------------ word model
def word_rows(R):
    X, y, g = [], [], []
    for r in R:
        if r["turn"] != r["final_turn"]:
            continue
        n = len(r["words"])
        for i in range(n):
            f = confidence.word_features(i, n, r["conf"][i], r["revs"][i], r["age"][i], r["spans"])
            X.append([f[k] for k in WORD_F])
            y.append(int(r["final_ok"][i]))
            g.append(r["id"])
    return np.array(X, float), np.array(y), g


def word_model(R):
    X, y, g = word_rows(R)
    fo = folds(g)
    fi = np.array([fo[q] for q in g])
    oof = np.zeros(len(y))
    for k in range(K):
        m, mu, sd = fit_lr(X[fi != k], y[fi != k])
        oof[fi == k] = pred(m, mu, sd, X[fi == k])
    ear = X[:, 0]
    print(f"WORD model: {len(y)} word observations (final turn), {100 * (1 - y.mean()):.1f}% later revised")
    print(f"  Flux confidence alone : AUC {roc_auc_score(y, ear):.3f}  Brier {brier_score_loss(y, ear):.4f}")
    print(f"  fused (out-of-fold)   : AUC {roc_auc_score(y, oof):.3f}  Brier {brier_score_loss(y, oof):.4f}")
    for th in (0.5, 0.8, 0.9):
        lo = oof < th
        lo_e = ear < th
        print(f"  flagged < {th}: fused {lo.mean() * 100:4.1f}% of words, {100 * (1 - y[lo]).mean() if lo.any() else 0:4.1f}%"
              f" of them revised | Flux {lo_e.mean() * 100:4.1f}%, {100 * (1 - y[lo_e]).mean() if lo_e.any() else 0:4.1f}%")
    m, mu, sd = fit_lr(X, y)
    spec = spec_of(m, mu, sd, WORD_F)
    print("  weights (standardized):", {k: round(w, 2) for k, w in zip(WORD_F, spec["w"][1:])})
    return spec, {"n": int(len(y)), "revised_pct": round(100 * (1 - y.mean()), 2),
                  "auc_flux": round(roc_auc_score(y, ear), 4), "auc_fused": round(roc_auc_score(y, oof), 4),
                  "brier_flux": round(brier_score_loss(y, ear), 5), "brier_fused": round(brier_score_loss(y, oof), 5)}


# ------------------------------------------------------------------------------------------------ eot model
def eot_feats(r):
    n = len(r["words"])
    slot_end = any(s.get("family") in ("entity", "number") and s["type"] != "entity.numeral" and s["end"] == n
                   for s in r["spans"])
    toks = [frame.norm(w) for w in r["words"]]
    w_end = r["ends"][-1]
    arc_recent = r.get("arc_cc_t") is not None and r["arc_cc_t"] >= w_end - 0.6
    return confidence.eot_features(
        eot=r["eot"], pause_s=r["pause"], n_words=n, cc_last=r["cc_at"] is not None and r["cc_at"] == n - 1,
        cc_score=r["cc_score"] or 0.0 if (r["cc_at"] == n - 1) else 0.0, gate=r["gate"],
        act_question=r["act_question"], slot_end=slot_end, has_content=r["has_content"],
        last_content=bool(toks) and toks[-1] not in frame.STOP, arc_cc_p=r.get("arc_cc_p"),
        arc_cc_recent=arc_recent, arc_for_me=r.get("arc_for_me"), is_eager=r["event"] == "EagerEndOfTurn",
        last_conf=r["conf"][-1] if r["conf"] else None)


def by_question(R):
    Q = {}
    for r in R:
        Q.setdefault(r["id"], []).append(r)
    for q in Q.values():
        q.sort(key=lambda r: r["t"])
    return Q


def replay(Q, decide):
    """decide(r) -> True (answer now) | float t (answer at t, a wait) | False. One soft answer per distinct words."""
    lat, prem, fires = [], 0, 0
    per_q = {}
    for qid, rs in Q.items():
        last_end, final_t = rs[0]["last_end"], rs[0]["final_t"]
        fired_at, done_words = None, set()
        for r in rs:
            key = (r["turn"], tuple(r["words"]))
            if key in done_words:
                continue
            d = decide(r)
            if d is None or d is False or isinstance(d, (bool, np.bool_)) and not d:
                continue
            at = r["t"] if isinstance(d, (bool, np.bool_)) else float(d)
            done_words.add(key)
            fires += 1
            if r["complete"]:
                fired_at = at
                break
            prem += 1
        t = fired_at if fired_at is not None else final_t
        lat.append((t - last_end) * 1000)
        per_q[qid] = round((t - last_end) * 1000)
    x = np.array(lat)
    n = len(Q)
    return {"p50": float(np.median(x)), "mean": float(x.mean()), "p90": float(np.percentile(x, 90)),
            "premature": 100.0 * prem / n, "fires": fires, "per_q": per_q}


BASE = {
    "EndOfTurn only": lambda r: False,
    "eager (default)": lambda r: r["event"] == "EagerEndOfTurn",
    "eager+arc": lambda r: r["event"] == "EagerEndOfTurn" and (
        True if (r.get("arc_cc_t") is not None and r["arc_cc_t"] >= r["ends"][-1] - 0.6) else (r.get("arc_wait_t") or False)),
    "eager+gate": lambda r: r["event"] == "EagerEndOfTurn" and r["gate"],
}


def eot_model(R, target_prem=None):
    Q = by_question(R)
    ids = sorted(Q)
    fo = folds(ids)
    X = {id(r): np.array([eot_feats(r)[k] for k in EOT_F], float) for r in R}
    print(f"\nEOT model: {len(R)} TurnInfo messages with words, {len(Q)} questions, "
          f"{sum(r['complete'] for r in R)} complete")
    res = {k: replay(Q, f) for k, f in BASE.items()}
    for k, v in res.items():
        print(f"  {k:34s} p50 {v['p50']:5.0f} mean {v['mean']:5.0f} p90 {v['p90']:5.0f}  premature/100q {v['premature']:5.1f}")
    eager = res["eager (default)"]
    variants = {"fused@eager": lambda r: r["event"] == "EagerEndOfTurn",
                "fused@any": lambda r: r["event"] in ("EagerEndOfTurn", "Update", "TurnResumed", "StartOfTurn")}
    out = {"baselines": {k: {kk: vv for kk, vv in v.items() if kk != "per_q"} for k, v in res.items()}}
    thr_grid = np.round(np.arange(0.05, 0.96, 0.025), 3)
    oof = {}
    for k in range(K):
        tr = [r for r in R if fo[r["id"]] != k]
        te = [r for r in R if fo[r["id"]] == k]
        m, mu, sd = fit_lr(np.stack([X[id(r)] for r in tr]), np.array([int(r["complete"]) for r in tr]))
        for r in te:
            oof[id(r)] = float(pred(m, mu, sd, X[id(r)][None])[0])
        for r in tr:
            r.setdefault("_in", {})[k] = float(pred(m, mu, sd, X[id(r)][None])[0])
    try:
        from sklearn.metrics import roc_auc_score as auc
        y = np.array([int(r["complete"]) for r in R])
        print(f"  out-of-fold AUC (all messages): fused {auc(y, [oof[id(r)] for r in R]):.3f} | Flux eot conf "
              f"{auc(y, [r['eot'] for r in R]):.3f}")
    except Exception:
        pass
    for vname, where in variants.items():
        curve = []
        for th in thr_grid:
            v = replay(Q, lambda r, th=th: where(r) and oof[id(r)] >= th)
            curve.append((float(th), v))
        # nested: in each fold choose the thr on the TRAINING questions (in-sample scores), apply to the test fold
        rules = {"match eager+arc premature, min p50": None, "min p50 with premature <= eager's / 2": None}
        nested = {}
        for rule in rules:
            lat_all, prem_all, n_all = {}, 0.0, 0
            for k in range(K):
                Qtr = {q: rs for q, rs in Q.items() if fo[q] != k}
                Qte = {q: rs for q, rs in Q.items() if fo[q] == k}
                ref = replay(Qtr, BASE["eager+arc"])["premature"] if rule.startswith("match") else \
                    replay(Qtr, BASE["eager (default)"])["premature"] / 2
                best = None
                for th in thr_grid:
                    v = replay(Qtr, lambda r, th=th: where(r) and r["_in"][k] >= th)
                    if v["premature"] <= ref + 1e-9 and (best is None or v["p50"] < best[1]["p50"] or
                                                         (v["p50"] == best[1]["p50"] and v["premature"] < best[1]["premature"])):
                        best = (th, v)
                th = best[0] if best else 0.95
                v = replay(Qte, lambda r, th=th: where(r) and oof[id(r)] >= th)
                lat_all.update(v["per_q"])
                prem_all += v["premature"] * len(Qte) / 100
                n_all += len(Qte)
            x = np.array(list(lat_all.values()))
            nested[rule] = {"p50": float(np.median(x)), "mean": float(x.mean()), "p90": float(np.percentile(x, 90)),
                            "premature": 100 * prem_all / n_all}
        print(f"  --- {vname} (out-of-fold scores; threshold sweep)")
        for th, v in curve:
            if abs((th * 40) - round(th * 40)) < 1e-6 and round(th * 40) % 2 == 0:
                dom = " <- dominates eager" if (v["p50"] <= eager["p50"] and v["premature"] < eager["premature"]) else ""
                print(f"    thr {th:.2f}  p50 {v['p50']:5.0f} mean {v['mean']:5.0f} p90 {v['p90']:5.0f}  "
                      f"premature/100q {v['premature']:5.1f}{dom}")
        for rule, v in nested.items():
            print(f"    NESTED ({rule}): p50 {v['p50']:5.0f} mean {v['mean']:5.0f} p90 {v['p90']:5.0f} premature/100q "
                  f"{v['premature']:5.1f}")
        out[vname] = {"curve": [{"thr": th, **{kk: vv for kk, vv in v.items() if kk != "per_q"}} for th, v in curve],
                      "nested": nested}
    # two thresholds: one on EagerEndOfTurn, a (usually higher) one on plain updates; nested selection as above
    grid2 = np.round(np.arange(0.3, 0.96, 0.05), 3)

    def two(r, te, tu, sc):
        if r["event"] == "EagerEndOfTurn":
            return sc >= te
        return r["event"] in ("Update", "TurnResumed", "StartOfTurn") and sc >= tu

    for rule in ("match eager+arc premature", "premature <= 8"):
        lat_all, prem_all, chosen = {}, 0.0, []
        for k in range(K):
            Qtr = {q: rs for q, rs in Q.items() if fo[q] != k}
            Qte = {q: rs for q, rs in Q.items() if fo[q] == k}
            ref = replay(Qtr, BASE["eager+arc"])["premature"] if rule.startswith("match") else 8.0
            best = None
            for te in grid2:
                for tu in grid2:
                    if tu < te:
                        continue
                    v = replay(Qtr, lambda r, te=te, tu=tu: two(r, te, tu, r["_in"][k]))
                    if v["premature"] <= ref + 1e-9 and (best is None or (v["p50"], v["mean"]) < (best[1]["p50"], best[1]["mean"])):
                        best = ((te, tu), v)
            te, tu = best[0] if best else (0.95, 0.95)
            chosen.append((float(te), float(tu)))
            v = replay(Qte, lambda r, te=te, tu=tu: two(r, te, tu, oof[id(r)]))
            lat_all.update(v["per_q"])
            prem_all += v["premature"] * len(Qte) / 100
        x = np.array(list(lat_all.values()))
        nv = {"p50": float(np.median(x)), "mean": float(x.mean()), "p90": float(np.percentile(x, 90)),
              "premature": 100 * prem_all / len(Q), "chosen": chosen}
        print(f"  --- fused, eager thr + update thr, NESTED ({rule}): p50 {nv['p50']:5.0f} mean {nv['mean']:5.0f} "
              f"p90 {nv['p90']:5.0f} premature/100q {nv['premature']:5.1f}  per-fold (thr_eager, thr_update) {chosen}")
        out.setdefault("fused@2thr", {})[rule] = nv
    for te, tu in ((0.65, 0.9), (0.6, 0.9), (0.7, 0.9), (0.65, 0.85), (0.65, 0.95)):
        v = replay(Q, lambda r, te=te, tu=tu: two(r, te, tu, oof[id(r)]))
        print(f"  fixed (thr_eager {te}, thr_update {tu}), out-of-fold scores: p50 {v['p50']:5.0f} mean {v['mean']:5.0f} "
              f"p90 {v['p90']:5.0f} premature/100q {v['premature']:5.1f}")
        out.setdefault("fused@2thr_fixed", {})[f"{te},{tu}"] = {kk: vv for kk, vv in v.items() if kk != "per_q"}
    for r in R:
        r.pop("_in", None)
    return Q, X, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="write server/conf_model.json")
    ap.add_argument("--thr-eager", type=float, default=None)
    ap.add_argument("--thr-update", type=float, default=None)
    ap.add_argument("--drop", default="", help="ablation: comma list of EOT features to leave out")
    a = ap.parse_args()
    global EOT_F
    EOT_F = [f for f in EOT_F if f not in set(a.drop.split(","))]
    R = [json.loads(x) for x in FEAT.read_text().splitlines() if x.strip()]
    wspec, wstats = word_model(R)
    Q, X, eot_stats = eot_model(R)
    m, mu, sd = fit_lr(np.stack([X[id(r)] for r in R]), np.array([int(r["complete"]) for r in R]))
    espec = spec_of(m, mu, sd, EOT_F)
    print("\n  EOT weights (standardized):", {k: round(w, 2) for k, w in zip(EOT_F, espec["w"][1:])})
    if a.thr_eager is not None:
        espec["thr_eager"] = a.thr_eager
    if a.thr_update is not None:
        espec["thr_update"] = a.thr_update
    if a.write:
        out = {"word": wspec, "eot": espec, "fitted_on": "runs/flux_traces.jsonl (80 real Flux traces)",
               "stats": {"word": wstats, "eot": eot_stats}}
        confidence.MODEL_PATH.write_text(json.dumps(out, indent=1))
        print("wrote", confidence.MODEL_PATH)


if __name__ == "__main__":
    main()
