# Team 6 project writeup

Live source: [Project Draft subtab](https://docs.google.com/document/d/1R1bV6s_dKIiCPTpNjPlFjH4C7jETn-MTjPRs5dSnEEA/edit?tab=t.cduxvcafdeir). This is a versioned snapshot for the `project_description` branch; the Google Doc remains editable.

## Team 6 project summary (working draft)

**Problem.** After an earthquake or other damaging event, responders may need to approach a structure whose condition is unknown. Motion that is hard to see could add useful hazard information while crews plan a search or rescue.

**Mission fit.** Our focused Forge 2026 objective is Detect Life and Hazards, specifically structural motion that may matter to first responders and special operations teams.

**Hardware and approach.** We have a Raspberry Pi 5, Raspberry Pi camera module, and AI HAT+ 26 TOPS. A teammate is building the camera-first implementation. The proposed portable unit observes a structure from a tripod, tracks target features relative to a reference, estimates camera motion, and reports the measured motion locally without relying on cloud access. An ultrasonic or other complementary sensor is under consideration; its value has not yet been tested.

**Planned physical demo.** A robot oscillates a cardboard building. We compare an isolated camera, a camera coupled to the model or its base, and deliberate camera vibration. An idle baseline and an independent measure of model movement would help us show whether the result reflects building motion rather than camera shake. Possible Boston building footage would be observational only.

**Output and limits.** The prototype aims to show concerning motion, no concerning motion observed, or an unreliable measurement. It cannot yet clear a building for entry or predict collapse. Those decisions require validated building-specific criteria and qualified assessment.

**Work still to validate.** Record the exact sensor models, calibration, motion units, ground truth, sensitivity, false alerts, latency, and performance in each demo condition. Separate what works live from what remains proposed.

**Shared project record.** The private [Team 6 GitHub repo](https://github.com/CodeMetalForge-Hackathon/Team-6) contains the [versioned project brief](https://github.com/CodeMetalForge-Hackathon/Team-6/blob/main/docs/project/earthquake-stability.md), research notes, demo runbook, hackathon instructions, and teammate onboarding. Continue adding ideas here, then reconcile decisions and results into the repo before submission.

# Earthquake Stability — Project Draft

**Team 6 • Code Metal Forge 2026 • Working draft**

## Problem

After an earthquake, fire, blast, or other damaging event, first responders may need to work near a building whose condition is uncertain. Subtle motion could be relevant to their hazard picture, especially while aftershocks, rubble removal, or machinery may disturb the structure. A responder needs a fast, interpretable observation from a safe vantage point. Motion alone cannot establish whether a building is safe to enter or whether it will collapse.

## Proposed solution

We are developing a portable, camera-first structural motion indicator for Forge’s **Detect Life and Hazards** objective. A Raspberry Pi 5 and Raspberry Pi camera module would observe visible features on a structure from a tripod and process video locally, without cloud access. The team also has an AI HAT+ 26 TOPS; its role is still to be confirmed by the implementation owner. An ultrasonic or other complementary sensor is under consideration, but no contribution from it has yet been demonstrated.

The intended output is a measured motion trace and a clear status such as **concerning motion observed**, **no concerning motion observed**, or **measurement unreliable**. Any demo threshold must be labeled as specific to the model and test conditions. These statuses are experimental hazard indicators, not entry clearance or a reliable collapse forecast.

## Research foundation

**Camera measurements and sensor fusion.** Hsu and Kuo’s 2020 [stand-alone smart camera study](https://pmc.ncbi.nlm.nih.gov/articles/PMC7349837/) combined a camera, single-board computer, and two accelerometers to estimate interstory drift in shaking-table tests of a scaled six-story steel building. Their method tracked reference targets, corrected camera rotation, calibrated displacement, and fused video with acceleration. In that experiment, camera-only estimates sometimes understated damage; the fused estimates matched the reference damage categories. This supports our focus on camera motion compensation and independent measurement, but the study used an instrumented building, known geometry, and calibrated damage criteria that our external tripod prototype does not yet have.

**Seeing subtle motion efficiently.** Komati and Nagmode’s 2023 [Modified Eulerian video magnification study](https://iieta.org/journals/ts/paper/10.18280/ts.400607) processes a selected region of video rather than every pixel, then amplifies small variations. The authors evaluated vibration-frequency measurement on a car engine and compared it with an accelerometer. This gives us a candidate technique for visualizing small movement in the cardboard demo and a reason to focus processing on the structure. Its reported performance is for the authors’ experiments; it does not demonstrate building damage detection or validate our Pi implementation.

**What the papers mean for us.** They show plausible ways to measure and display subtle motion, and they expose two failure modes we should test: camera motion mistaken for target motion, and a camera-only result that misses or understates relevant movement. We will test these ideas rather than claim that the cited algorithms or results are already reproduced.

## Planned live demonstration

A robot oscillates a cardboard building while the camera watches it. We plan four conditions: an idle baseline; a camera isolated from the moving model; a camera coupled to the model or its base; and deliberately added vibration at the camera. For each condition, we should record the commanded motion, an independent measure of model motion if available, the camera’s measurement, and whether the result is reliable. Repeated runs, fixed lighting and distance, and a clearly visible stationary reference will make the comparison interpretable. If we film a Boston building, that footage will be observational only and will not be labeled as a safety assessment.

## What remains to validate

The implementation owner will determine the actual algorithm, camera settings, sensor placement, and any AI HAT+ use. Before reporting performance, we need measured sensitivity, false alerts, latency, repeatability, calibration, and behavior under blur, occlusion, lighting changes, and camera shake. We also need to state what the system measures—pixels, displacement, frequency, or another quantity—and compare it against independent ground truth. A future building-entry decision would require building-specific structural criteria and qualified assessment beyond this hackathon prototype.

## References

Hsu, T.-Y., and Kuo, X.-J. (2020). [A Stand-Alone Smart Camera System for Online Post-Earthquake Building Safety Assessment](https://pmc.ncbi.nlm.nih.gov/articles/PMC7349837/). *Sensors*, 20(12), 3374\. DOI: 10.3390/s20123374.

Komati, R. D., and Nagmode, M. S. (2023). [Amplifying Imperceptible Variations in Video Sequences for Time-Varying Process Analysis](https://iieta.org/journals/ts/paper/10.18280/ts.400607). *Traitement du Signal*, 40(6), 2409–2422. DOI: 10.18280/ts.400607.

## Camera and Raspberry Pi setup — proposed starting profile

**Scope.** These settings are for the brightly lit cardboard-building demo with a Raspberry Pi 5 and Camera Module 3. They are starting values, not measured optimum settings or evidence of real-building accuracy. The implementation owner should test and record the final configuration.

### Capture settings

**Video:** Start with the team's agreed baseline: a 640 × 480 analysis stream at 30 frames per second. This gives about 33.3 ms between frames. Confirm the actual sensor mode, field of view, delivered frame rate, and dropped frames on the Pi. If the building occupies too few pixels, test a larger stream or a closer view; if processing falls behind, process selected frames or reduce the work per frame. After the 640 × 480, 30 fps baseline is accurate and repeatable, test higher resolution and frame rates one change at a time; keep each only if measured accuracy and processing speed remain acceptable.

**Exposure:** Start near 1/500 second (2,000 microseconds) under bright, steady light. Shorten exposure if edges blur; add light before increasing gain substantially. Start analogue gain near 1× and adjust only enough to keep tracked features visible. Avoid clipped highlights and very dark regions. If room lights cause brightness bands or flicker, use steady lighting and retest exposure.

**Lock changing camera controls after setup:** Let autofocus, exposure, and white balance settle in preview; then fix focus position, exposure, analogue gain, and white balance for each measured run. Use manual focus at the actual model distance. Keep HDR off, keep framing and digital crop fixed, and do not change settings midway through a comparison. Compare fast versus off noise reduction on recorded frames; choose the one that preserves trackable detail without lowering delivered frame rate.

**Scene:** Keep the entire cardboard model and a stationary reference in view. Use a rigid tripod, consistent distance and lighting, and distinct visible features at several locations on the model. Keep the reference separate from the moving base. Camera Module 3 uses a rolling shutter, so strong camera vibration can distort a frame; mark such runs unreliable if the reference cannot correct the motion.

### Processing sequence

1. Show a live preview. Detect or let the operator select the target building and stationary reference. Confirm the outline before analysis starts.

2. Track selected building features and reference features in each delivered frame. Run full building detection only when the outline is lost or at a lower rate, rather than spending the whole frame budget on it. Time-stamp every frame and retain frame metadata such as exposure, gain, and frame duration.

3. Estimate camera movement from the stationary reference, subtract it from the apparent target movement, and calculate relative motion by region. Show a contour or heat map of **measured apparent motion**, with the measurement units and an unreliable-data indicator. The map cannot by itself label a region as structurally weak or predict collapse.

4. Compare the idle, isolated-camera, coupled-camera, and added-camera-noise runs against an independent motion measurement if available. Record target detection success, delivered and processed fps, dropped frames, delay, false motion during the idle run, and motion error. Label thresholds as demo-specific until independently validated.

**Picamera2 controls to investigate:** FrameDurationLimits or FrameRate for capture rate; ExposureTime (microseconds), AnalogueGain, and AeEnable for exposure; AfMode and LensPosition for focus; AwbEnable for white balance. Read back actual image metadata after changing controls, because updates can take several frames. These control names and camera limits are documented in the [official Picamera2 manual](https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf). See also [Camera Module 3 hardware modes](https://www.raspberrypi.com/documentation/accessories/camera.html) and [Raspberry Pi camera software guidance](https://www.raspberrypi.com/documentation/accessories/camera-software.html).
