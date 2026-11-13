## wikipedia  (2500 configs × 2 seeds × 48 questions)

Baseline (sim): p50 1596 / p90 1796 ms, web/q 0.79, llm/q 1.1

**Current config**

| config | p50 | p90 | mean | risky | pool | top1=BL | web/q | llm/q | waste/q | reuse |
|---|---|---|---|---|---|---|---|---|---|---|
| stab=1 web=6 draft=6 sil=250 semst=420 cover=0.3 hedge=800 cap=1.0 | 1182 | 1650 | 1171 | 0.042 | 0.021 | 0.938 | 2.94 | 2.41 | 1.22 | 0.479 |

**Recommended** (accuracy proxy no worse than current; min p50 + 0.5·p90 + 25 ms/web call + 15 ms/LLM request)

| config | p50 | p90 | mean | risky | pool | top1=BL | web/q | llm/q | waste/q | reuse |
|---|---|---|---|---|---|---|---|---|---|---|
| stab=1 web=6 draft=6 sil=350 semst=550 cover=0.2 hedge=1000 cap=0.6 | 1160 | 1639 | 1160 | 0.042 | 0.021 | 0.938 | 2.94 | 2.27 | 1.17 | 0.521 |

`SPEC_STABLE_UPDATES=1 SPEC_WEB_BUDGET=6 SPEC_DRAFT_BUDGET=6 SEM_SILENCE_MS=350 SEM_STABLE_MS=550 SEM_COVER_S=0.2 HEDGE_MS=1000 PENDING_CAP_S=0.6`

**Cost-efficient alternative** (within 25 ms p50 / 50 ms p90 of the recommendation, fewest calls)

| config | p50 | p90 | mean | risky | pool | top1=BL | web/q | llm/q | waste/q | reuse |
|---|---|---|---|---|---|---|---|---|---|---|
| stab=1 web=6 draft=2 sil=350 semst=550 cover=0.3 hedge=1000 cap=0.6 | 1172 | 1669 | 1184 | 0.042 | 0.021 | 0.938 | 2.92 | 1.97 | 0.89 | 0.479 |

`SPEC_STABLE_UPDATES=1 SPEC_WEB_BUDGET=6 SPEC_DRAFT_BUDGET=2 SEM_SILENCE_MS=350 SEM_STABLE_MS=550 SEM_COVER_S=0.3 HEDGE_MS=1000 PENDING_CAP_S=0.6`

**Pareto front** over (p50, p90, risky, web/q, llm/q): 163 configs, fastest 25 by p50

| config | p50 | p90 | mean | risky | pool | top1=BL | web/q | llm/q | waste/q | reuse |
|---|---|---|---|---|---|---|---|---|---|---|
| ▶ stab=1 web=6 draft=6 sil=350 semst=550 cover=0.2 hedge=1000 cap=0.6 | 1160 | 1639 | 1160 | 0.042 | 0.021 | 0.938 | 2.94 | 2.27 | 1.17 | 0.521 |
| stab=1 web=6 draft=6 sil=350 semst=550 cover=0.2 hedge=1000 cap=1.0 | 1160 | 1639 | 1160 | 0.042 | 0.021 | 0.938 | 2.94 | 2.27 | 1.17 | 0.521 |
| stab=1 web=8 draft=4 sil=350 semst=550 cover=0.2 hedge=1000 cap=1.0 | 1160 | 1660 | 1164 | 0.042 | 0.021 | 0.938 | 2.94 | 2.21 | 1.06 | 0.521 |
| stab=1 web=8 draft=4 sil=350 semst=550 cover=0.2 hedge=1000 cap=0.6 | 1160 | 1660 | 1164 | 0.042 | 0.021 | 0.938 | 2.94 | 2.21 | 1.06 | 0.521 |
| stab=1 web=6 draft=4 sil=200 semst=550 cover=0.3 hedge=1000 cap=1.0 | 1160 | 1671 | 1173 | 0.042 | 0.021 | 0.938 | 2.94 | 2.18 | 1.06 | 0.531 |
| stab=1 web=6 draft=4 sil=250 semst=550 cover=0.3 hedge=1000 cap=1.0 | 1160 | 1671 | 1173 | 0.042 | 0.021 | 0.938 | 2.94 | 2.18 | 1.06 | 0.531 |
| stab=1 web=6 draft=4 sil=150 semst=550 cover=0.3 hedge=1000 cap=0.6 | 1160 | 1671 | 1173 | 0.042 | 0.021 | 0.938 | 2.94 | 2.18 | 1.06 | 0.531 |
| stab=1 web=6 draft=4 sil=200 semst=550 cover=0.3 hedge=1000 cap=0.6 | 1160 | 1671 | 1173 | 0.042 | 0.021 | 0.938 | 2.94 | 2.18 | 1.06 | 0.531 |
| stab=1 web=6 draft=4 sil=150 semst=550 cover=0.3 hedge=1000 cap=1.0 | 1160 | 1671 | 1173 | 0.042 | 0.021 | 0.938 | 2.94 | 2.18 | 1.06 | 0.531 |
| stab=1 web=6 draft=4 sil=350 semst=550 cover=0.3 hedge=1000 cap=1.0 | 1162 | 1650 | 1162 | 0.042 | 0.021 | 0.938 | 2.92 | 2.15 | 1.06 | 0.531 |
| stab=1 web=8 draft=2 sil=350 semst=550 cover=0.2 hedge=1000 cap=0.6 | 1167 | 1665 | 1182 | 0.042 | 0.021 | 0.938 | 2.94 | 2.03 | 0.89 | 0.469 |
| stab=1 web=6 draft=2 sil=250 semst=550 cover=0.3 hedge=1000 cap=1.0 | 1167 | 1672 | 1191 | 0.042 | 0.021 | 0.938 | 2.94 | 2.0 | 0.89 | 0.479 |
| stab=1 web=6 draft=2 sil=200 semst=550 cover=0.3 hedge=1000 cap=1.0 | 1167 | 1672 | 1191 | 0.042 | 0.021 | 0.938 | 2.94 | 2.0 | 0.89 | 0.479 |
| stab=1 web=6 draft=2 sil=250 semst=550 cover=0.3 hedge=1000 cap=0.6 | 1167 | 1672 | 1191 | 0.042 | 0.021 | 0.938 | 2.94 | 2.0 | 0.89 | 0.479 |
| stab=1 web=6 draft=2 sil=350 semst=550 cover=0.3 hedge=1000 cap=0.6 | 1172 | 1669 | 1184 | 0.042 | 0.021 | 0.938 | 2.92 | 1.97 | 0.89 | 0.479 |
| stab=2 web=6 draft=6 sil=350 semst=550 cover=0.2 hedge=1000 cap=1.0 | 1175 | 1640 | 1186 | 0.042 | 0.021 | 0.917 | 2.93 | 2.19 | 1.08 | 0.479 |
| stab=2 web=6 draft=4 sil=350 semst=550 cover=0.3 hedge=1000 cap=0.6 | 1175 | 1654 | 1192 | 0.042 | 0.021 | 0.917 | 2.92 | 2.06 | 0.98 | 0.49 |
| stab=1 web=4 draft=6 sil=150 semst=550 cover=0.2 hedge=1000 cap=0.6 | 1177 | 1698 | 1212 | 0.042 | 0.021 | 0.938 | 2.88 | 2.35 | 1.18 | 0.385 |
| stab=1 web=4 draft=6 sil=200 semst=550 cover=0.2 hedge=1000 cap=1.0 | 1177 | 1698 | 1212 | 0.042 | 0.021 | 0.938 | 2.88 | 2.35 | 1.18 | 0.385 |
| stab=1 web=4 draft=6 sil=200 semst=550 cover=0.2 hedge=1000 cap=0.6 | 1177 | 1698 | 1212 | 0.042 | 0.021 | 0.938 | 2.88 | 2.35 | 1.18 | 0.385 |
| stab=1 web=4 draft=4 sil=150 semst=550 cover=0.2 hedge=1000 cap=0.6 | 1177 | 1717 | 1215 | 0.042 | 0.021 | 0.938 | 2.88 | 2.25 | 1.07 | 0.385 |
| stab=1 web=4 draft=6 sil=150 semst=420 cover=0.2 hedge=1000 cap=1.0 | 1180 | 1697 | 1201 | 0.042 | 0.021 | 0.938 | 2.88 | 2.42 | 1.23 | 0.333 |
| stab=1 web=4 draft=6 sil=150 semst=420 cover=0.2 hedge=1000 cap=0.6 | 1180 | 1697 | 1201 | 0.042 | 0.021 | 0.938 | 2.88 | 2.42 | 1.23 | 0.333 |
| stab=1 web=4 draft=6 sil=200 semst=420 cover=0.2 hedge=1000 cap=0.6 | 1180 | 1697 | 1201 | 0.042 | 0.021 | 0.938 | 2.88 | 2.42 | 1.23 | 0.333 |
| stab=1 web=4 draft=6 sil=250 semst=550 cover=0.2 hedge=1000 cap=0.6 | 1180 | 1698 | 1216 | 0.042 | 0.021 | 0.938 | 2.85 | 2.3 | 1.17 | 0.417 |

**Top 10 by the selection score**

| config | p50 | p90 | mean | risky | pool | top1=BL | web/q | llm/q | waste/q | reuse |
|---|---|---|---|---|---|---|---|---|---|---|
| stab=1 web=6 draft=6 sil=350 semst=550 cover=0.2 hedge=1000 cap=0.6 | 1160 | 1639 | 1160 | 0.042 | 0.021 | 0.938 | 2.94 | 2.27 | 1.17 | 0.521 |
| stab=1 web=6 draft=6 sil=350 semst=550 cover=0.2 hedge=1000 cap=1.0 | 1160 | 1639 | 1160 | 0.042 | 0.021 | 0.938 | 2.94 | 2.27 | 1.17 | 0.521 |
| stab=1 web=6 draft=6 sil=250 semst=550 cover=0.2 hedge=1000 cap=0.6 | 1160 | 1639 | 1169 | 0.042 | 0.021 | 0.938 | 2.94 | 2.34 | 1.21 | 0.479 |
| stab=1 web=6 draft=6 sil=250 semst=550 cover=0.2 hedge=800 cap=0.6 | 1160 | 1639 | 1167 | 0.042 | 0.021 | 0.938 | 2.94 | 2.41 | 1.21 | 0.479 |
| stab=1 web=6 draft=6 sil=150 semst=550 cover=0.2 hedge=1000 cap=0.6 | 1160 | 1639 | 1166 | 0.042 | 0.021 | 0.938 | 2.96 | 2.4 | 1.22 | 0.448 |
| stab=1 web=6 draft=6 sil=200 semst=550 cover=0.2 hedge=1000 cap=0.6 | 1160 | 1639 | 1166 | 0.042 | 0.021 | 0.938 | 2.96 | 2.4 | 1.22 | 0.448 |
| stab=1 web=6 draft=6 sil=200 semst=550 cover=0.2 hedge=1000 cap=1.0 | 1160 | 1639 | 1166 | 0.042 | 0.021 | 0.938 | 2.96 | 2.4 | 1.22 | 0.448 |
| stab=1 web=6 draft=6 sil=150 semst=550 cover=0.2 hedge=800 cap=0.6 | 1160 | 1639 | 1164 | 0.042 | 0.021 | 0.938 | 2.96 | 2.46 | 1.22 | 0.448 |
| stab=1 web=6 draft=6 sil=200 semst=550 cover=0.2 hedge=800 cap=0.6 | 1160 | 1639 | 1164 | 0.042 | 0.021 | 0.938 | 2.96 | 2.46 | 1.22 | 0.448 |
| stab=1 web=6 draft=6 sil=150 semst=550 cover=0.2 hedge=800 cap=1.0 | 1160 | 1639 | 1164 | 0.042 | 0.021 | 0.938 | 2.96 | 2.46 | 1.22 | 0.448 |

**Knob marginals** (mean over configs with that value: p50 / p90 / risky / web per q)

- `SPEC_STABLE_UPDATES` — 1: 1207 / 1719 / 0.037 / 2.87; 2: 1220 / 1714 / 0.037 / 2.87
- `SPEC_WEB_BUDGET` — 3: 1282 / 1791 / 0.021 / 2.73; 4: 1202 / 1744 / 0.042 / 2.86; 6: 1184 / 1664 / 0.042 / 2.94; 8: 1184 / 1666 / 0.042 / 2.95
- `SPEC_DRAFT_BUDGET` — 2: 1217 / 1717 / 0.037 / 2.87; 4: 1211 / 1723 / 0.037 / 2.87; 6: 1211 / 1711 / 0.037 / 2.87
- `SEM_SILENCE_MS` — 150: 1208 / 1714 / 0.037 / 2.88; 200: 1211 / 1719 / 0.036 / 2.88; 250: 1218 / 1717 / 0.037 / 2.86; 350: 1215 / 1717 / 0.037 / 2.85
- `SEM_STABLE_MS` — 300: 1215 / 1711 / 0.037 / 2.88; 420: 1206 / 1718 / 0.037 / 2.86; 550: 1218 / 1721 / 0.037 / 2.86
- `SEM_COVER_S` — 0.2: 1202 / 1706 / 0.037 / 2.88; 0.3: 1212 / 1721 / 0.037 / 2.87; 0.4: 1227 / 1724 / 0.037 / 2.86
- `HEDGE_MS` — 600: 1214 / 1718 / 0.037 / 2.87; 800: 1214 / 1719 / 0.036 / 2.87; 1000: 1211 / 1714 / 0.037 / 2.87
- `PENDING_CAP_S` — 0.6: 1213 / 1717 / 0.037 / 2.87; 1.0: 1213 / 1717 / 0.037 / 2.87


## tavily  (2500 configs × 2 seeds × 48 questions)

Baseline (sim): p50 2236 / p90 2580 ms, web/q 0.79, llm/q 1.1

**Current config**

| config | p50 | p90 | mean | risky | pool | top1=BL | web/q | llm/q | waste/q | reuse |
|---|---|---|---|---|---|---|---|---|---|---|
| stab=1 web=6 draft=6 sil=250 semst=420 cover=0.3 hedge=800 cap=1.0 | 1703 | 2298 | 1662 | 0.052 | 0.021 | 0.927 | 2.96 | 1.92 | 0.72 | 0.083 |

**Recommended** (accuracy proxy no worse than current; min p50 + 0.5·p90 + 25 ms/web call + 15 ms/LLM request)

| config | p50 | p90 | mean | risky | pool | top1=BL | web/q | llm/q | waste/q | reuse |
|---|---|---|---|---|---|---|---|---|---|---|
| stab=1 web=6 draft=2 sil=200 semst=550 cover=0.4 hedge=1000 cap=1.0 | 1627 | 2290 | 1685 | 0.021 | 0.021 | 0.958 | 2.94 | 1.57 | 0.46 | 0.052 |

`SPEC_STABLE_UPDATES=1 SPEC_WEB_BUDGET=6 SPEC_DRAFT_BUDGET=2 SEM_SILENCE_MS=200 SEM_STABLE_MS=550 SEM_COVER_S=0.4 HEDGE_MS=1000 PENDING_CAP_S=1.0`

**Pareto front** over (p50, p90, risky, web/q, llm/q): 138 configs, fastest 25 by p50

| config | p50 | p90 | mean | risky | pool | top1=BL | web/q | llm/q | waste/q | reuse |
|---|---|---|---|---|---|---|---|---|---|---|
| stab=1 web=6 draft=2 sil=150 semst=420 cover=0.2 hedge=1000 cap=0.6 | 1443 | 2278 | 1589 | 0.26 | 0.021 | 0.708 | 2.98 | 1.7 | 0.48 | 0.031 |
| stab=1 web=6 draft=2 sil=200 semst=420 cover=0.2 hedge=1000 cap=0.6 | 1443 | 2278 | 1589 | 0.26 | 0.021 | 0.708 | 2.98 | 1.7 | 0.48 | 0.031 |
| stab=1 web=8 draft=4 sil=150 semst=420 cover=0.2 hedge=1000 cap=0.6 | 1448 | 2276 | 1578 | 0.26 | 0.021 | 0.708 | 2.98 | 1.83 | 0.64 | 0.062 |
| stab=1 web=8 draft=4 sil=200 semst=420 cover=0.2 hedge=1000 cap=0.6 | 1448 | 2276 | 1578 | 0.26 | 0.021 | 0.708 | 2.98 | 1.83 | 0.64 | 0.062 |
| stab=2 web=6 draft=2 sil=150 semst=420 cover=0.2 hedge=1000 cap=0.6 | 1458 | 2280 | 1612 | 0.26 | 0.021 | 0.708 | 2.98 | 1.65 | 0.44 | 0.031 |
| stab=2 web=6 draft=2 sil=200 semst=420 cover=0.2 hedge=1000 cap=0.6 | 1458 | 2280 | 1612 | 0.26 | 0.021 | 0.708 | 2.98 | 1.65 | 0.44 | 0.031 |
| stab=2 web=8 draft=2 sil=200 semst=420 cover=0.2 hedge=1000 cap=0.6 | 1458 | 2280 | 1612 | 0.26 | 0.021 | 0.708 | 2.98 | 1.65 | 0.44 | 0.031 |
| stab=1 web=6 draft=2 sil=200 semst=550 cover=0.2 hedge=1000 cap=0.6 | 1462 | 2278 | 1602 | 0.198 | 0.021 | 0.771 | 2.98 | 1.68 | 0.48 | 0.031 |
| stab=1 web=6 draft=2 sil=150 semst=550 cover=0.2 hedge=1000 cap=0.6 | 1462 | 2278 | 1602 | 0.198 | 0.021 | 0.771 | 2.98 | 1.68 | 0.48 | 0.031 |
| stab=1 web=6 draft=4 sil=200 semst=550 cover=0.2 hedge=1000 cap=0.6 | 1488 | 2276 | 1590 | 0.198 | 0.021 | 0.771 | 2.98 | 1.83 | 0.64 | 0.062 |
| stab=2 web=6 draft=2 sil=250 semst=420 cover=0.2 hedge=1000 cap=0.6 | 1490 | 2278 | 1616 | 0.271 | 0.021 | 0.698 | 2.96 | 1.6 | 0.46 | 0.031 |
| stab=1 web=8 draft=6 sil=250 semst=420 cover=0.2 hedge=1000 cap=0.6 | 1490 | 2278 | 1588 | 0.26 | 0.021 | 0.708 | 2.96 | 1.85 | 0.72 | 0.083 |
| stab=2 web=8 draft=2 sil=250 semst=420 cover=0.2 hedge=1000 cap=0.6 | 1490 | 2278 | 1616 | 0.271 | 0.021 | 0.698 | 2.96 | 1.6 | 0.46 | 0.031 |
| stab=2 web=6 draft=4 sil=250 semst=420 cover=0.2 hedge=1000 cap=0.6 | 1493 | 2276 | 1606 | 0.271 | 0.021 | 0.698 | 2.96 | 1.74 | 0.61 | 0.062 |
| stab=1 web=6 draft=4 sil=250 semst=420 cover=0.2 hedge=1000 cap=0.6 | 1493 | 2276 | 1592 | 0.26 | 0.021 | 0.708 | 2.96 | 1.75 | 0.61 | 0.083 |
| stab=1 web=6 draft=2 sil=250 semst=550 cover=0.2 hedge=1000 cap=0.6 | 1512 | 2278 | 1614 | 0.198 | 0.021 | 0.771 | 2.96 | 1.59 | 0.46 | 0.052 |
| stab=1 web=6 draft=4 sil=250 semst=550 cover=0.2 hedge=1000 cap=0.6 | 1513 | 2276 | 1602 | 0.198 | 0.021 | 0.771 | 2.96 | 1.75 | 0.61 | 0.083 |
| stab=1 web=4 draft=2 sil=200 semst=420 cover=0.2 hedge=1000 cap=0.6 | 1514 | 2369 | 1658 | 0.219 | 0.021 | 0.76 | 2.9 | 1.7 | 0.48 | 0.031 |
| stab=1 web=4 draft=2 sil=250 semst=420 cover=0.2 hedge=1000 cap=0.6 | 1516 | 2401 | 1671 | 0.219 | 0.021 | 0.76 | 2.88 | 1.61 | 0.46 | 0.052 |
| stab=2 web=4 draft=2 sil=250 semst=420 cover=0.2 hedge=1000 cap=0.6 | 1516 | 2401 | 1684 | 0.229 | 0.021 | 0.75 | 2.88 | 1.6 | 0.46 | 0.031 |
| stab=2 web=8 draft=2 sil=250 semst=550 cover=0.2 hedge=1000 cap=0.6 | 1520 | 2278 | 1630 | 0.208 | 0.021 | 0.76 | 2.96 | 1.58 | 0.46 | 0.031 |
| stab=1 web=4 draft=6 sil=150 semst=550 cover=0.2 hedge=1000 cap=0.6 | 1548 | 2369 | 1663 | 0.156 | 0.021 | 0.823 | 2.9 | 1.94 | 0.74 | 0.062 |
| stab=1 web=4 draft=6 sil=200 semst=550 cover=0.2 hedge=1000 cap=0.6 | 1548 | 2369 | 1663 | 0.156 | 0.021 | 0.823 | 2.9 | 1.94 | 0.74 | 0.062 |
| stab=1 web=4 draft=2 sil=250 semst=550 cover=0.2 hedge=1000 cap=0.6 | 1548 | 2401 | 1691 | 0.156 | 0.021 | 0.823 | 2.88 | 1.59 | 0.46 | 0.052 |
| stab=2 web=4 draft=2 sil=200 semst=420 cover=0.2 hedge=1000 cap=0.6 | 1558 | 2369 | 1680 | 0.219 | 0.021 | 0.76 | 2.9 | 1.65 | 0.44 | 0.031 |

**Top 10 by the selection score**

| config | p50 | p90 | mean | risky | pool | top1=BL | web/q | llm/q | waste/q | reuse |
|---|---|---|---|---|---|---|---|---|---|---|
| stab=1 web=6 draft=2 sil=200 semst=550 cover=0.4 hedge=1000 cap=1.0 | 1627 | 2290 | 1685 | 0.021 | 0.021 | 0.958 | 2.94 | 1.57 | 0.46 | 0.052 |
| stab=1 web=6 draft=2 sil=150 semst=550 cover=0.4 hedge=1000 cap=1.0 | 1627 | 2290 | 1685 | 0.021 | 0.021 | 0.958 | 2.94 | 1.57 | 0.46 | 0.052 |
| stab=1 web=8 draft=2 sil=350 semst=550 cover=0.3 hedge=1000 cap=1.0 | 1627 | 2290 | 1669 | 0.042 | 0.021 | 0.938 | 2.96 | 1.55 | 0.46 | 0.052 |
| stab=1 web=8 draft=2 sil=350 semst=550 cover=0.4 hedge=1000 cap=1.0 | 1627 | 2290 | 1676 | 0.021 | 0.021 | 0.958 | 2.96 | 1.55 | 0.46 | 0.052 |
| stab=1 web=8 draft=2 sil=150 semst=550 cover=0.4 hedge=1000 cap=1.0 | 1627 | 2290 | 1685 | 0.021 | 0.021 | 0.958 | 2.96 | 1.57 | 0.46 | 0.052 |
| stab=1 web=6 draft=2 sil=350 semst=550 cover=0.4 hedge=800 cap=1.0 | 1627 | 2290 | 1674 | 0.021 | 0.021 | 0.958 | 2.94 | 1.61 | 0.46 | 0.052 |
| stab=1 web=6 draft=2 sil=150 semst=550 cover=0.4 hedge=800 cap=1.0 | 1627 | 2290 | 1683 | 0.021 | 0.021 | 0.958 | 2.94 | 1.64 | 0.46 | 0.052 |
| stab=1 web=6 draft=2 sil=200 semst=550 cover=0.4 hedge=800 cap=1.0 | 1627 | 2290 | 1683 | 0.021 | 0.021 | 0.958 | 2.94 | 1.64 | 0.46 | 0.052 |
| stab=1 web=6 draft=2 sil=250 semst=550 cover=0.4 hedge=800 cap=1.0 | 1627 | 2290 | 1683 | 0.021 | 0.021 | 0.958 | 2.94 | 1.64 | 0.46 | 0.052 |
| stab=1 web=8 draft=2 sil=350 semst=550 cover=0.4 hedge=800 cap=1.0 | 1627 | 2290 | 1674 | 0.021 | 0.021 | 0.958 | 2.96 | 1.61 | 0.46 | 0.052 |

**Knob marginals** (mean over configs with that value: p50 / p90 / risky / web per q)

- `SPEC_STABLE_UPDATES` — 1: 1711 / 2369 / 0.124 / 2.89; 2: 1732 / 2371 / 0.131 / 2.89
- `SPEC_WEB_BUDGET` — 3: 1844 / 2513 / 0.092 / 2.75; 4: 1722 / 2394 / 0.131 / 2.89; 6: 1657 / 2284 / 0.146 / 2.96; 8: 1660 / 2284 / 0.142 / 2.97
- `SPEC_DRAFT_BUDGET` — 2: 1718 / 2368 / 0.125 / 2.89; 4: 1726 / 2371 / 0.127 / 2.89; 6: 1719 / 2370 / 0.130 / 2.89
- `SEM_SILENCE_MS` — 150: 1716 / 2362 / 0.133 / 2.91; 200: 1720 / 2371 / 0.137 / 2.90; 250: 1725 / 2371 / 0.130 / 2.88; 350: 1724 / 2376 / 0.109 / 2.87
- `SEM_STABLE_MS` — 300: 1754 / 2364 / 0.172 / 2.91; 420: 1713 / 2374 / 0.123 / 2.88; 550: 1697 / 2371 / 0.088 / 2.88
- `SEM_COVER_S` — 0.2: 1691 / 2363 / 0.145 / 2.90; 0.3: 1720 / 2373 / 0.127 / 2.89; 0.4: 1755 / 2374 / 0.110 / 2.88
- `HEDGE_MS` — 600: 1724 / 2371 / 0.125 / 2.89; 800: 1724 / 2373 / 0.126 / 2.89; 1000: 1716 / 2366 / 0.132 / 2.90
- `PENDING_CAP_S` — 0.6: 1678 / 2367 / 0.210 / 2.89; 1.0: 1761 / 2373 / 0.052 / 2.89

