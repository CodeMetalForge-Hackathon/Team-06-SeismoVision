# Earthquake Stability project brief

**Working name:** Earthquake Stability

**Team:** Code Metal Forge 2026, Team 6

**Status:** experimental prototype plan; implementation and validation in progress

## Problem and user

After an earthquake, fire, blast, or other damaging event, responders may have to work around a structure before its condition is known. Rubble removal, aftershocks, and ongoing activity can add motion. An operator needs timely, interpretable evidence about motion while keeping people away from an uncertain structure where possible.

The initial user is a first responder or special operations team assessing a hazardous search and rescue scene. The project focuses on detecting a possible **structural motion hazard** under Forge's Objective 02. It does not attempt to certify occupancy or replace a qualified building safety evaluation.

## Proposed system

The team currently has a Raspberry Pi 5, Camera Module 3, and AI HAT+ 26 TOPS. A teammate owns the implementation. An ultrasonic sensor or other complementary sensor is being considered; its model, placement, and contribution are not yet fixed.

The agreed starting point for the camera analysis is **640 × 480 at 30 frames per second**. The team will establish motion accuracy and processing speed at this baseline, then test higher resolution and frame rates one change at a time. The [live Project Draft](https://docs.google.com/document/d/1R1bV6s_dKIiCPTpNjPlFjH4C7jETn-MTjPRs5dSnEEA/edit?tab=t.cduxvcafdeir) and [versioned writeup](project-draft.md) contain the proposed camera settings and validation checks. These settings are a test plan, not a verified optimum.

**Implementation handoff:** `seismovision/config.example.json` currently uses 1280 × 720 at 30 fps, with regions defined in those coordinates. The implementation owner must revise the resolution and regions together before using the agreed baseline.

The intended field concept is a portable camera on a tripod, pointed at a structure from a suitable observation position. Processing should run locally on the Pi so the core measurement does not depend on cellular service or cloud access. The proposed analysis tracks visible features or selected regions of the target, estimates camera movement from a reference where possible, and reports target motion with a reliability indication. Sensor fusion and any use of the AI HAT+ remain implementation decisions to validate, not completed capabilities.

The output should distinguish **concerning motion detected**, **no concerning motion observed**, and **measurement unreliable**. These are experimental motion observations. No output may be interpreted as entry clearance or a reliable collapse forecast without building-specific validation and qualified review.

## Planned physical demo

An oscillating robot moves a cardboard building. The team plans to compare:

1. Camera isolated from the moving model: measure target motion against a stationary reference.
2. Camera mechanically coupled to the model or its base: show how shared vibration changes the raw image and whether compensation helps.
3. Added vibration at the camera: test whether camera noise can be identified or rejected while target motion remains visible.

An idle control, measured ground truth, and repeat runs would make the comparison interpretable. [The demo runbook](../demo/runbook.md) records the exact procedure and results once the implementation owner confirms them. Any footage of a Boston building is observational only; the team should not label a real building safe or unsafe from this prototype.

## Evidence and limits

Published research demonstrates video measurement of subtle motion and, in a controlled steel-building experiment, calibrated camera plus accelerometer measurements of interstory drift. That work used known target geometry, multiple sensors, building-specific damage thresholds, and laboratory ground truth. It does not supply a universal threshold for this external tripod setup, a cardboard model, or a Boston building. [Research and evidence](../research/evidence.md) explains the sources and the validation path.

Post-earthquake entry decisions also consider damage, nonstructural hazards, and other site conditions. Qualified evaluators use established procedures such as ATC-20. This prototype can support awareness and future research, not independently make those decisions.

## What the team must still establish

- Exact Camera Module 3 variant, optics, mounting, reference features, operating distance, and measured performance at the 640 × 480, 30 fps baseline.
- Exact complementary sensor model and whether it measures useful target or camera motion.
- Ground-truth motion measurements for each demo condition and a calibration procedure.
- Tested sensitivity, false alerts, latency, repeatability, and unreliable-data behavior.
- What the weekend implementation actually does, what was pre-existing, and what remains proposed.
- For any future real-building claim: relevant building class, calibrated physical units, suitable structural metrics, qualified review, and independent field validation.
