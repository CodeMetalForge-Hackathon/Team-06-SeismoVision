# Planned live demo runbook

**Status:** plan for the implementation owner to confirm and fill in. This document does not assert that the camera pipeline already works.

## What the judge should see

In one sentence: the camera observes an oscillating cardboard building and reports target motion while the team changes how much the camera itself moves. Show the live measurement and its reliability state, then explain what was built during the weekend.

## Before the judge arrives

- Confirm the robot, cardboard model, camera, Pi, display, and power work together.
- Fix camera distance, field of view, target features, and reference features; record any calibration values.
- Measure or mark a ground-truth oscillation amplitude or frequency if possible.
- Run the entire sequence three times. Record a backup video if useful, clearly labeled as recorded.
- Keep one teammate at the table during the Sunday 12:00–2:00 p.m. judging period.

## Live sequence

1. **Idle control:** model and camera still. Show the baseline reading and reliability state.
2. **Isolated camera:** move the model with the robot. Show measured target motion against the reference.
3. **Coupled camera:** let the camera share base motion. Compare raw apparent motion with the compensated result, if implemented.
4. **Camera disturbance:** add vibration at the camera. Show whether the result stays useful or correctly becomes **measurement unreliable**.
5. **Explain the boundary:** the reading is a prototype motion observation, not a real-building safe-entry or collapse verdict.

Do not present a replay as live. If the live pipeline fails, label the backup video and show the actual working pieces and failure honestly. Optional Boston building footage should be identified as observational footage, with no safety classification of that building.

## Result log to fill in during testing

For each condition, record date and operator, camera and sensor models, mounting, distance, frame rate, robot setting, ground-truth motion, measured motion, latency, reliability state, false alerts, and what failed. Link results or clips in this repository only if the team has permission to share them. Keep demo-calibrated thresholds separate from any real-world structural threshold.

## Judge questions

- What existed before the Saturday 10:00 a.m. build start, and what did the team build this weekend?
- Which of the three conditions worked live, and which did not?
- How does the system identify camera motion rather than target motion?
- What are the current power, latency, bandwidth, and offline constraints?
- What validation and qualified review would be required before field use?

The [mission brief](../logistics/archive/forge-2026-saturday-mission-brief.pdf) allows partial progress on a hard problem and recommends a focused live capability. It specifies about 15 minutes at the table and 10 minutes for finalist stage demos.
