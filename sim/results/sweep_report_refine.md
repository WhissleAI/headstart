## wikipedia  (76 configs × 12 seeds × 48 questions)

Baseline (sim): p50 1602 / p90 1839 ms, web/q 0.79, llm/q 1.12

**Current config**

| config | p50 | p90 | mean | risky | pool | top1=BL | web/q | llm/q | waste/q | reuse |
|---|---|---|---|---|---|---|---|---|---|---|
| stab=1 web=6 draft=6 sil=250 semst=420 cover=0.3 hedge=800 cap=1.0 | 1197 | 1728 | 1201 | 0.049 | 0.021 | 0.918 | 2.94 | 2.4 | 1.18 | 0.523 |

**Recommended** (accuracy proxy no worse than current; min p50 + 0.5·p90 + 25 ms/web call + 15 ms/LLM request)

| config | p50 | p90 | mean | risky | pool | top1=BL | web/q | llm/q | waste/q | reuse |
|---|---|---|---|---|---|---|---|---|---|---|
| stab=1 web=6 draft=4 sil=200 semst=550 cover=0.2 hedge=800 cap=0.6 | 1164 | 1711 | 1194 | 0.05 | 0.021 | 0.917 | 2.97 | 2.45 | 1.17 | 0.431 |

`SPEC_STABLE_UPDATES=1 SPEC_WEB_BUDGET=6 SPEC_DRAFT_BUDGET=4 SEM_SILENCE_MS=200 SEM_STABLE_MS=550 SEM_COVER_S=0.2 HEDGE_MS=800 PENDING_CAP_S=0.6`

**Cost-efficient alternative** (within 25 ms p50 / 50 ms p90 of the recommendation, fewest calls)

| config | p50 | p90 | mean | risky | pool | top1=BL | web/q | llm/q | waste/q | reuse |
|---|---|---|---|---|---|---|---|---|---|---|
| stab=1 web=6 draft=2 sil=350 semst=550 cover=0.3 hedge=1000 cap=0.6 | 1182 | 1745 | 1215 | 0.049 | 0.021 | 0.918 | 2.92 | 1.98 | 0.87 | 0.479 |

`SPEC_STABLE_UPDATES=1 SPEC_WEB_BUDGET=6 SPEC_DRAFT_BUDGET=2 SEM_SILENCE_MS=350 SEM_STABLE_MS=550 SEM_COVER_S=0.3 HEDGE_MS=1000 PENDING_CAP_S=0.6`

**Pareto front** over (p50, p90, risky, web/q, llm/q): 42 configs, fastest 25 by p50

| config | p50 | p90 | mean | risky | pool | top1=BL | web/q | llm/q | waste/q | reuse |
|---|---|---|---|---|---|---|---|---|---|---|
| stab=1 web=6 draft=4 sil=150 semst=550 cover=0.2 hedge=800 cap=1.0 | 1164 | 1711 | 1194 | 0.049 | 0.021 | 0.918 | 2.97 | 2.45 | 1.17 | 0.431 |
| stab=1 web=8 draft=4 sil=200 semst=550 cover=0.2 hedge=1000 cap=1.0 | 1164 | 1718 | 1198 | 0.049 | 0.021 | 0.918 | 2.97 | 2.39 | 1.17 | 0.431 |
| stab=1 web=8 draft=4 sil=250 semst=550 cover=0.2 hedge=800 cap=1.0 | 1166 | 1726 | 1196 | 0.049 | 0.021 | 0.918 | 2.94 | 2.38 | 1.14 | 0.46 |
| stab=1 web=6 draft=4 sil=250 semst=550 cover=0.2 hedge=1000 cap=0.6 | 1166 | 1733 | 1201 | 0.049 | 0.021 | 0.918 | 2.94 | 2.32 | 1.14 | 0.46 |
| stab=1 web=6 draft=4 sil=250 semst=550 cover=0.2 hedge=1000 cap=1.0 | 1166 | 1733 | 1201 | 0.049 | 0.021 | 0.918 | 2.94 | 2.32 | 1.14 | 0.46 |
| stab=1 web=6 draft=6 sil=200 semst=550 cover=0.3 hedge=800 cap=0.6 | 1169 | 1732 | 1195 | 0.049 | 0.021 | 0.918 | 2.94 | 2.37 | 1.15 | 0.538 |
| stab=1 web=6 draft=6 sil=250 semst=550 cover=0.3 hedge=800 cap=1.0 | 1169 | 1732 | 1195 | 0.049 | 0.021 | 0.918 | 2.94 | 2.37 | 1.15 | 0.538 |
| stab=1 web=6 draft=6 sil=150 semst=550 cover=0.3 hedge=800 cap=0.6 | 1169 | 1732 | 1195 | 0.049 | 0.021 | 0.918 | 2.94 | 2.37 | 1.15 | 0.538 |
| stab=1 web=6 draft=6 sil=150 semst=550 cover=0.3 hedge=800 cap=1.0 | 1169 | 1732 | 1195 | 0.049 | 0.021 | 0.918 | 2.94 | 2.37 | 1.15 | 0.538 |
| stab=1 web=6 draft=6 sil=150 semst=550 cover=0.3 hedge=1000 cap=1.0 | 1169 | 1742 | 1200 | 0.049 | 0.021 | 0.918 | 2.94 | 2.31 | 1.15 | 0.538 |
| stab=1 web=6 draft=6 sil=150 semst=550 cover=0.3 hedge=1000 cap=0.6 | 1169 | 1742 | 1200 | 0.049 | 0.021 | 0.918 | 2.94 | 2.31 | 1.15 | 0.538 |
| stab=1 web=6 draft=6 sil=250 semst=550 cover=0.3 hedge=1000 cap=0.6 | 1169 | 1742 | 1200 | 0.049 | 0.021 | 0.918 | 2.94 | 2.31 | 1.15 | 0.538 |
| stab=1 web=6 draft=4 sil=200 semst=550 cover=0.3 hedge=1000 cap=1.0 | 1169 | 1746 | 1201 | 0.049 | 0.021 | 0.918 | 2.94 | 2.21 | 1.06 | 0.538 |
| stab=1 web=6 draft=4 sil=250 semst=550 cover=0.3 hedge=1000 cap=1.0 | 1169 | 1746 | 1201 | 0.049 | 0.021 | 0.918 | 2.94 | 2.21 | 1.06 | 0.538 |
| stab=1 web=6 draft=4 sil=150 semst=550 cover=0.3 hedge=1000 cap=0.6 | 1169 | 1746 | 1201 | 0.049 | 0.021 | 0.918 | 2.94 | 2.21 | 1.06 | 0.538 |
| stab=1 web=6 draft=4 sil=200 semst=550 cover=0.3 hedge=1000 cap=0.6 | 1169 | 1746 | 1201 | 0.049 | 0.021 | 0.918 | 2.94 | 2.21 | 1.06 | 0.538 |
| stab=1 web=6 draft=4 sil=150 semst=550 cover=0.3 hedge=1000 cap=1.0 | 1169 | 1746 | 1201 | 0.049 | 0.021 | 0.918 | 2.94 | 2.21 | 1.06 | 0.538 |
| stab=1 web=6 draft=6 sil=150 semst=420 cover=0.2 hedge=800 cap=1.0 | 1172 | 1707 | 1191 | 0.049 | 0.021 | 0.918 | 2.97 | 2.59 | 1.3 | 0.405 |
| stab=1 web=6 draft=6 sil=200 semst=420 cover=0.2 hedge=800 cap=1.0 | 1172 | 1707 | 1191 | 0.049 | 0.021 | 0.918 | 2.97 | 2.59 | 1.3 | 0.405 |
| stab=1 web=6 draft=4 sil=350 semst=550 cover=0.3 hedge=800 cap=1.0 | 1173 | 1730 | 1190 | 0.049 | 0.021 | 0.918 | 2.92 | 2.22 | 1.06 | 0.538 |
| stab=1 web=6 draft=4 sil=350 semst=550 cover=0.3 hedge=1000 cap=1.0 | 1173 | 1741 | 1195 | 0.049 | 0.021 | 0.918 | 2.92 | 2.17 | 1.06 | 0.538 |
| stab=1 web=6 draft=6 sil=250 semst=420 cover=0.2 hedge=800 cap=0.6 | 1175 | 1722 | 1195 | 0.049 | 0.021 | 0.918 | 2.94 | 2.51 | 1.26 | 0.441 |
| stab=1 web=6 draft=4 sil=350 semst=550 cover=0.2 hedge=800 cap=0.6 | 1175 | 1728 | 1189 | 0.049 | 0.021 | 0.918 | 2.94 | 2.3 | 1.08 | 0.517 |
| stab=1 web=8 draft=4 sil=350 semst=550 cover=0.2 hedge=800 cap=1.0 | 1175 | 1728 | 1189 | 0.049 | 0.021 | 0.918 | 2.94 | 2.3 | 1.08 | 0.517 |
| stab=1 web=6 draft=2 sil=250 semst=550 cover=0.3 hedge=1000 cap=1.0 | 1180 | 1745 | 1223 | 0.049 | 0.021 | 0.918 | 2.94 | 2.03 | 0.87 | 0.479 |

**Top 10 by the selection score**

| config | p50 | p90 | mean | risky | pool | top1=BL | web/q | llm/q | waste/q | reuse |
|---|---|---|---|---|---|---|---|---|---|---|
| stab=1 web=6 draft=4 sil=200 semst=550 cover=0.2 hedge=800 cap=0.6 | 1164 | 1711 | 1194 | 0.05 | 0.021 | 0.917 | 2.97 | 2.45 | 1.17 | 0.431 |
| stab=1 web=6 draft=4 sil=150 semst=550 cover=0.2 hedge=800 cap=1.0 | 1164 | 1711 | 1194 | 0.049 | 0.021 | 0.918 | 2.97 | 2.45 | 1.17 | 0.431 |
| stab=1 web=6 draft=6 sil=150 semst=550 cover=0.2 hedge=800 cap=0.6 | 1164 | 1711 | 1192 | 0.05 | 0.021 | 0.917 | 2.97 | 2.55 | 1.27 | 0.431 |
| stab=1 web=6 draft=6 sil=200 semst=550 cover=0.2 hedge=800 cap=0.6 | 1164 | 1711 | 1192 | 0.05 | 0.021 | 0.917 | 2.97 | 2.55 | 1.27 | 0.431 |
| stab=1 web=6 draft=6 sil=150 semst=550 cover=0.2 hedge=800 cap=1.0 | 1164 | 1711 | 1192 | 0.049 | 0.021 | 0.918 | 2.97 | 2.55 | 1.27 | 0.431 |
| stab=1 web=6 draft=6 sil=150 semst=550 cover=0.2 hedge=1000 cap=0.6 | 1164 | 1715 | 1197 | 0.05 | 0.021 | 0.917 | 2.97 | 2.49 | 1.27 | 0.431 |
| stab=1 web=6 draft=6 sil=200 semst=550 cover=0.2 hedge=1000 cap=0.6 | 1164 | 1715 | 1197 | 0.05 | 0.021 | 0.917 | 2.97 | 2.49 | 1.27 | 0.431 |
| stab=1 web=6 draft=6 sil=200 semst=550 cover=0.2 hedge=1000 cap=1.0 | 1164 | 1715 | 1197 | 0.049 | 0.021 | 0.918 | 2.97 | 2.49 | 1.27 | 0.431 |
| stab=1 web=6 draft=4 sil=200 semst=550 cover=0.2 hedge=1000 cap=0.6 | 1164 | 1718 | 1198 | 0.05 | 0.021 | 0.917 | 2.97 | 2.39 | 1.17 | 0.431 |
| stab=1 web=8 draft=4 sil=200 semst=550 cover=0.2 hedge=1000 cap=1.0 | 1164 | 1718 | 1198 | 0.049 | 0.021 | 0.918 | 2.97 | 2.39 | 1.17 | 0.431 |

**Knob marginals** (mean over configs with that value: p50 / p90 / risky / web per q)

- `SPEC_STABLE_UPDATES` — 1: 1181 / 1737 / 0.049 / 2.93; 2: 1186 / 1740 / 0.049 / 2.94
- `SPEC_WEB_BUDGET` — 4: 1210 / 1763 / 0.049 / 2.87; 6: 1173 / 1730 / 0.049 / 2.95; 8: 1173 / 1729 / 0.049 / 2.94
- `SPEC_DRAFT_BUDGET` — 2: 1193 / 1749 / 0.049 / 2.92; 4: 1177 / 1739 / 0.049 / 2.93; 6: 1178 / 1731 / 0.049 / 2.94
- `SEM_SILENCE_MS` — 150: 1179 / 1733 / 0.049 / 2.94; 200: 1178 / 1731 / 0.049 / 2.94; 250: 1187 / 1746 / 0.049 / 2.91; 350: 1180 / 1738 / 0.049 / 2.93
- `SEM_STABLE_MS` — 420: 1193 / 1732 / 0.049 / 2.92; 550: 1178 / 1739 / 0.049 / 2.93
- `SEM_COVER_S` — 0.2: 1184 / 1735 / 0.049 / 2.93; 0.3: 1176 / 1741 / 0.049 / 2.94
- `HEDGE_MS` — 800: 1171 / 1722 / 0.049 / 2.95; 1000: 1185 / 1743 / 0.049 / 2.92
- `PENDING_CAP_S` — 0.6: 1182 / 1739 / 0.049 / 2.93; 1.0: 1180 / 1735 / 0.049 / 2.93


## tavily  (81 configs × 12 seeds × 48 questions)

Baseline (sim): p50 2251 / p90 2707 ms, web/q 0.79, llm/q 1.12

**Current config**

| config | p50 | p90 | mean | risky | pool | top1=BL | web/q | llm/q | waste/q | reuse |
|---|---|---|---|---|---|---|---|---|---|---|
| stab=1 web=6 draft=6 sil=250 semst=420 cover=0.3 hedge=800 cap=1.0 | 1702 | 2327 | 1688 | 0.08 | 0.021 | 0.898 | 2.96 | 1.91 | 0.71 | 0.083 |

**Recommended** (accuracy proxy no worse than current; min p50 + 0.5·p90 + 25 ms/web call + 15 ms/LLM request)

| config | p50 | p90 | mean | risky | pool | top1=BL | web/q | llm/q | waste/q | reuse |
|---|---|---|---|---|---|---|---|---|---|---|
| stab=1 web=8 draft=6 sil=200 semst=550 cover=0.3 hedge=800 cap=1.0 | 1667 | 2320 | 1679 | 0.075 | 0.021 | 0.905 | 2.96 | 1.91 | 0.71 | 0.083 |

`SPEC_STABLE_UPDATES=1 SPEC_WEB_BUDGET=8 SPEC_DRAFT_BUDGET=6 SEM_SILENCE_MS=200 SEM_STABLE_MS=550 SEM_COVER_S=0.3 HEDGE_MS=800 PENDING_CAP_S=1.0`

**Cost-efficient alternative** (within 25 ms p50 / 50 ms p90 of the recommendation, fewest calls)

| config | p50 | p90 | mean | risky | pool | top1=BL | web/q | llm/q | waste/q | reuse |
|---|---|---|---|---|---|---|---|---|---|---|
| stab=1 web=6 draft=2 sil=200 semst=550 cover=0.4 hedge=1000 cap=1.0 | 1680 | 2347 | 1717 | 0.043 | 0.021 | 0.936 | 2.94 | 1.58 | 0.46 | 0.047 |

`SPEC_STABLE_UPDATES=1 SPEC_WEB_BUDGET=6 SPEC_DRAFT_BUDGET=2 SEM_SILENCE_MS=200 SEM_STABLE_MS=550 SEM_COVER_S=0.4 HEDGE_MS=1000 PENDING_CAP_S=1.0`

**Pareto front** over (p50, p90, risky, web/q, llm/q): 63 configs, fastest 25 by p50

| config | p50 | p90 | mean | risky | pool | top1=BL | web/q | llm/q | waste/q | reuse |
|---|---|---|---|---|---|---|---|---|---|---|
| stab=1 web=8 draft=4 sil=150 semst=420 cover=0.2 hedge=1000 cap=0.6 | 1481 | 2293 | 1593 | 0.25 | 0.021 | 0.717 | 2.98 | 1.85 | 0.64 | 0.064 |
| stab=1 web=8 draft=4 sil=200 semst=420 cover=0.2 hedge=1000 cap=0.6 | 1481 | 2293 | 1593 | 0.25 | 0.021 | 0.717 | 2.98 | 1.85 | 0.64 | 0.064 |
| stab=1 web=6 draft=2 sil=150 semst=420 cover=0.2 hedge=1000 cap=0.6 | 1483 | 2291 | 1609 | 0.25 | 0.021 | 0.717 | 2.98 | 1.72 | 0.48 | 0.028 |
| stab=1 web=6 draft=2 sil=200 semst=420 cover=0.2 hedge=1000 cap=0.6 | 1483 | 2291 | 1609 | 0.25 | 0.021 | 0.717 | 2.98 | 1.72 | 0.48 | 0.028 |
| stab=2 web=6 draft=2 sil=150 semst=420 cover=0.2 hedge=1000 cap=0.6 | 1498 | 2296 | 1628 | 0.269 | 0.021 | 0.693 | 2.98 | 1.69 | 0.46 | 0.028 |
| stab=2 web=6 draft=2 sil=200 semst=420 cover=0.2 hedge=1000 cap=0.6 | 1498 | 2296 | 1628 | 0.269 | 0.021 | 0.693 | 2.98 | 1.69 | 0.46 | 0.028 |
| stab=2 web=8 draft=2 sil=200 semst=420 cover=0.2 hedge=1000 cap=0.6 | 1498 | 2296 | 1628 | 0.269 | 0.021 | 0.693 | 2.98 | 1.69 | 0.46 | 0.028 |
| stab=1 web=6 draft=4 sil=200 semst=550 cover=0.2 hedge=1000 cap=0.6 | 1508 | 2293 | 1601 | 0.207 | 0.021 | 0.767 | 2.98 | 1.86 | 0.64 | 0.064 |
| stab=1 web=8 draft=6 sil=250 semst=420 cover=0.2 hedge=1000 cap=0.6 | 1509 | 2301 | 1608 | 0.245 | 0.021 | 0.722 | 2.96 | 1.87 | 0.71 | 0.082 |
| stab=1 web=6 draft=4 sil=250 semst=420 cover=0.2 hedge=1000 cap=0.6 | 1510 | 2312 | 1608 | 0.245 | 0.021 | 0.722 | 2.96 | 1.77 | 0.62 | 0.082 |
| stab=1 web=6 draft=2 sil=200 semst=550 cover=0.2 hedge=1000 cap=0.6 | 1511 | 2292 | 1620 | 0.207 | 0.021 | 0.767 | 2.98 | 1.7 | 0.48 | 0.028 |
| stab=1 web=6 draft=2 sil=150 semst=550 cover=0.2 hedge=1000 cap=0.6 | 1511 | 2292 | 1620 | 0.207 | 0.021 | 0.767 | 2.98 | 1.7 | 0.48 | 0.028 |
| stab=2 web=6 draft=2 sil=250 semst=420 cover=0.2 hedge=1000 cap=0.6 | 1519 | 2310 | 1635 | 0.266 | 0.021 | 0.696 | 2.96 | 1.64 | 0.47 | 0.028 |
| stab=2 web=8 draft=2 sil=250 semst=420 cover=0.2 hedge=1000 cap=0.6 | 1519 | 2310 | 1635 | 0.266 | 0.021 | 0.696 | 2.96 | 1.64 | 0.47 | 0.028 |
| stab=1 web=6 draft=4 sil=250 semst=550 cover=0.2 hedge=1000 cap=0.6 | 1529 | 2312 | 1615 | 0.205 | 0.021 | 0.769 | 2.96 | 1.77 | 0.62 | 0.082 |
| stab=1 web=6 draft=2 sil=250 semst=550 cover=0.2 hedge=1000 cap=0.6 | 1530 | 2309 | 1634 | 0.205 | 0.021 | 0.769 | 2.96 | 1.62 | 0.47 | 0.045 |
| stab=1 web=4 draft=2 sil=200 semst=420 cover=0.2 hedge=1000 cap=0.6 | 1538 | 2400 | 1680 | 0.205 | 0.021 | 0.773 | 2.9 | 1.72 | 0.48 | 0.028 |
| stab=1 web=4 draft=6 sil=150 semst=550 cover=0.2 hedge=1000 cap=0.6 | 1552 | 2426 | 1680 | 0.163 | 0.021 | 0.814 | 2.9 | 1.95 | 0.73 | 0.064 |
| stab=1 web=4 draft=6 sil=200 semst=550 cover=0.2 hedge=1000 cap=0.6 | 1552 | 2426 | 1680 | 0.163 | 0.021 | 0.814 | 2.9 | 1.95 | 0.73 | 0.064 |
| stab=1 web=4 draft=4 sil=150 semst=550 cover=0.2 hedge=1000 cap=0.6 | 1552 | 2431 | 1681 | 0.163 | 0.021 | 0.814 | 2.9 | 1.86 | 0.64 | 0.064 |
| stab=2 web=4 draft=2 sil=200 semst=420 cover=0.2 hedge=1000 cap=0.6 | 1564 | 2406 | 1700 | 0.224 | 0.021 | 0.748 | 2.9 | 1.69 | 0.46 | 0.028 |
| stab=1 web=6 draft=4 sil=350 semst=420 cover=0.2 hedge=1000 cap=0.6 | 1567 | 2322 | 1628 | 0.196 | 0.021 | 0.771 | 2.95 | 1.74 | 0.61 | 0.083 |
| stab=1 web=8 draft=4 sil=350 semst=550 cover=0.2 hedge=1000 cap=0.6 | 1571 | 2322 | 1629 | 0.158 | 0.021 | 0.816 | 2.96 | 1.74 | 0.61 | 0.083 |
| stab=1 web=4 draft=2 sil=250 semst=420 cover=0.2 hedge=1000 cap=0.6 | 1572 | 2420 | 1695 | 0.2 | 0.021 | 0.778 | 2.88 | 1.63 | 0.47 | 0.045 |
| stab=1 web=8 draft=2 sil=350 semst=550 cover=0.2 hedge=1000 cap=0.6 | 1574 | 2319 | 1648 | 0.158 | 0.021 | 0.816 | 2.96 | 1.59 | 0.46 | 0.047 |

**Top 10 by the selection score**

| config | p50 | p90 | mean | risky | pool | top1=BL | web/q | llm/q | waste/q | reuse |
|---|---|---|---|---|---|---|---|---|---|---|
| stab=1 web=8 draft=6 sil=200 semst=550 cover=0.3 hedge=800 cap=1.0 | 1667 | 2320 | 1679 | 0.075 | 0.021 | 0.905 | 2.96 | 1.91 | 0.71 | 0.083 |
| stab=1 web=8 draft=6 sil=250 semst=550 cover=0.3 hedge=800 cap=1.0 | 1667 | 2320 | 1679 | 0.075 | 0.021 | 0.905 | 2.96 | 1.91 | 0.71 | 0.083 |
| stab=1 web=8 draft=6 sil=150 semst=550 cover=0.3 hedge=800 cap=1.0 | 1667 | 2320 | 1679 | 0.075 | 0.021 | 0.905 | 2.96 | 1.91 | 0.71 | 0.083 |
| stab=1 web=6 draft=6 sil=250 semst=550 cover=0.3 hedge=800 cap=1.0 | 1667 | 2320 | 1679 | 0.075 | 0.021 | 0.905 | 2.96 | 1.91 | 0.71 | 0.083 |
| stab=1 web=6 draft=6 sil=150 semst=550 cover=0.3 hedge=800 cap=1.0 | 1667 | 2320 | 1679 | 0.075 | 0.021 | 0.905 | 2.96 | 1.91 | 0.71 | 0.083 |
| stab=1 web=6 draft=2 sil=200 semst=550 cover=0.3 hedge=800 cap=1.0 | 1673 | 2328 | 1699 | 0.075 | 0.021 | 0.905 | 2.96 | 1.66 | 0.46 | 0.047 |
| stab=1 web=8 draft=2 sil=150 semst=550 cover=0.3 hedge=800 cap=1.0 | 1673 | 2328 | 1699 | 0.075 | 0.021 | 0.905 | 2.96 | 1.66 | 0.46 | 0.047 |
| stab=1 web=6 draft=2 sil=150 semst=550 cover=0.3 hedge=800 cap=1.0 | 1673 | 2328 | 1699 | 0.075 | 0.021 | 0.905 | 2.96 | 1.66 | 0.46 | 0.047 |
| stab=1 web=8 draft=2 sil=200 semst=550 cover=0.3 hedge=800 cap=1.0 | 1673 | 2328 | 1699 | 0.075 | 0.021 | 0.905 | 2.96 | 1.66 | 0.46 | 0.047 |
| stab=1 web=6 draft=2 sil=250 semst=550 cover=0.3 hedge=800 cap=1.0 | 1673 | 2328 | 1699 | 0.075 | 0.021 | 0.905 | 2.96 | 1.66 | 0.46 | 0.047 |

**Knob marginals** (mean over configs with that value: p50 / p90 / risky / web per q)

- `SPEC_STABLE_UPDATES` — 1: 1625 / 2336 / 0.112 / 2.95; 2: 1550 / 2338 / 0.232 / 2.95
- `SPEC_WEB_BUDGET` — 4: 1575 / 2430 / 0.181 / 2.89; 6: 1614 / 2323 / 0.129 / 2.96; 8: 1623 / 2323 / 0.118 / 2.96
- `SPEC_DRAFT_BUDGET` — 2: 1612 / 2338 / 0.134 / 2.95; 4: 1595 / 2332 / 0.144 / 2.95; 6: 1640 / 2340 / 0.102 / 2.95
- `SEM_SILENCE_MS` — 150: 1615 / 2335 / 0.118 / 2.95; 200: 1604 / 2333 / 0.137 / 2.95; 250: 1599 / 2341 / 0.155 / 2.94; 350: 1640 / 2337 / 0.105 / 2.95
- `SEM_STABLE_MS` — 420: 1532 / 2328 / 0.234 / 2.95; 550: 1638 / 2339 / 0.099 / 2.95
- `SEM_COVER_S` — 0.2: 1538 / 2343 / 0.214 / 2.94; 0.3: 1655 / 2329 / 0.091 / 2.96; 0.4: 1677 / 2337 / 0.048 / 2.95
- `HEDGE_MS` — 600: 1679 / 2314 / 0.049 / 2.96; 800: 1675 / 2330 / 0.061 / 2.95; 1000: 1583 / 2341 / 0.165 / 2.95
- `PENDING_CAP_S` — 0.6: 1549 / 2340 / 0.204 / 2.95; 1.0: 1675 / 2333 / 0.060 / 2.95

