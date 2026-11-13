# Head Start: demo checklist

One command does the technical part: `./run_demo.sh`. It checks prerequisites, frees the port, starts the server,
waits for the warm-up, replays q01 in Race mode against the real backends and prints **GO**, **GO (DEGRADED)** or
**NO-GO**, then opens Chrome. The rest of this page covers what a script can't do.

## T-30 min: the machine
- [ ] **Power:** charger plugged in, battery above 80 %. Battery settings: Low Power Mode **off** (it throttles the
      CPU and the Arc ear falls behind). Prevent sleep: `caffeinate -dis &` in a spare terminal.
- [ ] **Close CPU hogs:** Docker Desktop, other browsers, Slack, Xcode/Simulator, any `uvicorn`/`python` left from
      benchmarks (`ps aux | sort -nrk 3 | head`), Spotlight indexing on a new drive. Make sure
      `runs/.run_ab.lock` does not exist (it means a benchmark is running and every latency is off).
- [ ] **Notifications off:** Focus → Do Not Disturb (no banners over the race panes).
- [ ] **Display:** plug in the projector, System Settings → Displays → **Mirror**, resolution **1280×720** (or the
      closest "scaled" option). Chrome full screen (⌃⌘F) and zoom 100 % (⌘0). The layout is built for 1280×720.
- [ ] **Audio out:** mute system sounds; the replay plays the question audio through the speakers, so check the room
      volume once with a replay.

## T-15 min: start and preflight
```sh
cd ~/Desktop/work/whissle/headstart
./run_demo.sh            # live web (falls back to the recorded cache by itself if the lane fails)
./run_demo.sh --full     # also replays a local-moments question (local_05)
./run_demo.sh --offline  # venue Wi-Fi is bad: web lane serves the recorded cache from the start (labelled)
```
- [ ] Read the checklist it prints. **GO** = everything live. **GO (DEGRADED)** = a fallback is active, the UI shows
      a yellow "Degraded mode" banner; demo it and say so if asked. **NO-GO** = see Recovery below.
- [ ] Header status says **ready** (green dot). Yellow dot "ready · degraded" = a fallback is active.
- [ ] Mode **Race** selected.
- [ ] Replay q01 once more by hand (sidebar) so the browser, the audio element and the connection pools are warm.

## Mic permission (Chrome)
- [ ] Open http://localhost:8770, press **Space** (or Ask) once: Chrome asks for the microphone, click **Allow**.
      `localhost` counts as a secure origin, so no HTTPS is needed.
- [ ] If it was blocked before: click the icon left of the URL → Microphone → Allow → reload.
- [ ] macOS: System Settings → Privacy & Security → Microphone → Google Chrome **on** (Chrome must be restarted after
      turning it on).
- [ ] Input device: System Settings → Sound → Input. Prefer a headset or lavalier over the laptop mic in a loud room;
      check the input level meter moves while you talk.
- [ ] Say "who founded the company Nvidia" once: the transcript card should show the words within ~0.3 s.

## Room mic is bad: use replays (sidebar, keys 1–9, R)
Replays stream the recorded question through the same ear in real time; the race is identical to a live question.

| beat | replay | what it shows |
|---|---|---|
| live race | **q01** "who founded the company Nvidia" | searches fire while talking, answer first, big delta |
| self-correction | **q27** "capital of Canada, actually make that Turkey" | Canada search goes stale, Turkey fires |
| moments (Local tab) | **local_05** "In the hearing, where was the gun found?" | audio lane, cites the moment timestamp; works without web |
| spare factoids | q03 (Apollo 11), q02 (Australia), q05 (Everest) | backups if one misbehaves |

Local-moments questions (Local tab, local_01…local_10) need no web at all: retrieval is on the laptop. Only the
answer model is remote, and it falls back to an extractive answer if that is down.

## Degraded modes (what the audience sees, and what to say)
| failure | what the server does | UI | say |
|---|---|---|---|
| web lane unreachable / slow (>3 s) | serves `runs/web_cache.json` (57 recorded Wikipedia responses for the eval set), stops trying live for 30 s, retries after | yellow banner "Web: live web unreachable: serving recorded cache"; pane chip "cached web ×N"; sources marked CACHED | "Venue Wi-Fi is down, so the web results are a recording of real Wikipedia responses; the speculation, timing and answers are live." |
| `--offline` / `WEB_BACKEND=cache` | cache only, never touches the network for the web lane | banner "cached (... not live)" | same |
| question not in the cache | web lane returns no sources (local questions unaffected) | answer says the sources don't cover it | use a replay from the table above |
| Anthropic unreachable (or no first token in 4 s) | extractive answer: best-matching sentence of the top source, cited; stops trying for 30 s | banner "Answers: answer model unreachable: extractive answers"; badge "extractive · not generated" on the answer | "Without the network the answer is extracted from the top source instead of generated; the race still shows when each side had its sources." |

Both modes always use the same fallbacks (same process-wide state), so the race stays fair. Nothing cached or
extracted is ever shown as live: every such result carries a label.

## Recovery
| symptom | fix |
|---|---|
| NO-GO: prerequisite missing | the line says which file; models live outside git (`ARC_DIR`, HF cache for EmbeddingGemma 2, `models/arctag`, `vendor/arctag`). Without Wi-Fi a missing model cannot be downloaded: check before leaving. |
| NO-GO: server exited during warm-up | `tail -40 runs/server_8770.log`. Port taken by something else: `./run_demo.sh --port 8771`. |
| NO-GO: web lane down **and** no cache | `git pull` / copy `runs/web_cache.json`; or record on a working network: `.venv/bin/python eval/record_web_cache.py`. Demo the Local tab meanwhile. |
| NO-GO: answers empty | Anthropic key missing or invalid: check `.env` has `ANTHROPIC_API_KEY` (don't paste it on screen). |
| NO-GO: latency (Head Start > 2.5 s) | something is eating the CPU (benchmark lock, Docker, a second server); close it and rerun `./run_demo.sh`. |
| status dot red "offline" in the page | the server died: rerun `./run_demo.sh` (it kills the stale one), reload the page. |
| words don't appear when talking | mic permission or wrong input device (above); switch to replays, don't debug on stage. |
| a replay hangs or looks wrong | press another replay: each one starts a fresh run. Reload the page if the timeline is stuck. |
| Wi-Fi drops mid-demo | nothing to do: the first failed call opens the breaker and the next question is served from the cache within ~3 s; the banner appears. If it returns, the server notices on the next call after 30 s. |
| want to force offline mid-demo | `./run_demo.sh --offline` (restarts the server: ~40 s warm-up). |

## Before leaving for the venue
- [ ] `./run_demo.sh --full` on the home network: GO.
- [ ] `./run_demo.sh --offline --full --no-open`: GO (DEGRADED), q01 and local_05 answered.
- [ ] Turn Wi-Fi off, `./run_demo.sh --full --no-open`: GO (DEGRADED) with cached web + extractive answers. Wi-Fi on.
- [ ] `.venv/bin/python -m pytest -q tests` green.
