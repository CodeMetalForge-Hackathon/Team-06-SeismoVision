# Earthquake Stability — SeismoVision

Team 6's Code Metal Forge 2026 prototype measures visible motion in a model structure. After an earthquake, responders need information about hazards before approaching a damaged building. Our proposed field unit uses a camera to show where motion is measurable and whether the observation is reliable. **It cannot determine that a building is stable, predict collapse, or clear anyone to enter.** A qualified building assessment remains necessary.

## Current prototype

The implemented live grid preview runs on a Raspberry Pi 5 with a Camera Module 3. After a camera preview, a scene-specific OpenCV detector identifies the cardboard stack and separates its upper and lower sections. This is **not a trained YOLO model** and is tuned to the unmarked cardboard demo. The lower section represents the comparatively stable base; the upper section is the part expected to move more.

The program lays out eight cells across the lower section and two across the upper section. It tracks image features inside the cells, preserving the green cell boxes and dots shown in the demo. A visible room background provides reference features for estimating camera movement. Motion of the upper section is compared with the lower base; this helps distinguish a responding structure from common table or camera movement. Tracking quality is checked continuously. Sustained loss of features or pressing `R` triggers automatic reacquisition of the outline and tracking points. When the scene or camera is obscured, measurements should be treated as unreliable until tracking recovers.

The camera baseline is **640 × 480 at 30 fps**. The software uses optical flow for motion tracking and a sliding-window FFT to show frequency content over time. The default grid configuration analyzes roughly 0.2–5 Hz with a 30-second window and at least 180 samples. Those settings are prototype parameters, not validated structural thresholds. Optional motion magnification is available but disabled by default.

### Reading the display

| Display element | Meaning |
| --- | --- |
| Yellow outlines | Detected upper and lower cardboard regions; reacquired after tracking loss or `R`. |
| Green boxes and dots | Analysis cells and the tracked image features inside them. Sparse or lost dots reduce confidence. |
| Motion indicators | Measured image displacement, with background and base motion used as references. |
| Colored FFT panels (“heat maps”) | Frequency amplitude over time for each analysis cell. Color indicates stronger or weaker measured frequency energy; these are **spectrograms**, not a map of physical damage or structural strength. |
| Highlighted or unreliable cells | Relative motion anomalies or insufficient tracking evidence. An anomaly is a prompt for closer inspection, not a collapse probability. |

The grid preview and the separate configured-target `seismovision` pipeline are distinct entry points. After following the [installation steps](seismovision/README.md), run the grid preview from its directory on the Pi with:

```bash
cd seismovision
cp grid_config.example.json grid_config.json
python -m seismovision.grid_preview --config grid_config.json --camera picamera2
```

See [seismovision/README.md](seismovision/README.md) for installation, controls, configuration, and the alternate pipeline. The prototype has not been validated on real buildings or calibrated to a safe-entry decision.

## Demo and evidence

The table moves and the cardboard stack responds; the table edge remains visible. The team is comparing an isolated camera, a camera coupled to the moving setup, and added camera vibration. The [demo runbook](docs/demo/runbook.md) describes the trial plan. [Documentation media](documentation/README.md) records the cardboard model and screen output. Those images and videos illustrate the prototype; they do not establish measurement accuracy or ground truth. The [research record](docs/research/evidence.md) separates sourced methods from claims the team has not validated.

## Hardware and indicative unit cost

| Component | Role | Indicative price (USD) |
| --- | --- | ---: |
| [Raspberry Pi 5, 16 GB](https://www.raspberrypi.com/news/price-rises-for-some-16gb-raspberry-pi-products/) | Local capture and analysis | $385 |
| [Camera Module 3](https://www.raspberrypi.com/products/camera-module-3/) | Video capture | From $25 |
| [AI HAT+ 26 TOPS](https://datasheets.raspberrypi.com/ai-hat-plus/raspberry-pi-ai-hat-plus-product-brief.pdf) | Available accelerator hardware; current OpenCV grid tracking does not require it | $110 |
| [5-inch Touch Display 2](https://www.raspberrypi.com/products/touch-display-2/) | Proposed local field display | $40 |
| [Active Cooler](https://www.raspberrypi.com/news/introducing-raspberry-pi-5/) | Pi cooling | $5 |
| Tripod, mount, storage, camera cable, enclosure, portable power | Field integration allowance; not yet selected or tested | $80–$180 estimate |

**Estimated one-off parts total: $645–$745 with the AI HAT+, or $535–$635 without it.** The allowance is a planning estimate, not a vendor quote. Prices exclude tax, shipping, assembly, and validation. A portable power system and display have not been integrated into the current demo; the Pi 5 requires an appropriate power supply.

## Intended field workflow — proposed

A responder would place a tripod-mounted camera and small display at a safe standoff, frame the building together with a stable reference, and review measured motion, frequency changes, and tracking confidence. An unreliable view would prompt repositioning or reacquisition. This is a **screening aid** for qualified responders, not an entry authorization. Camera motion, distance, lighting, occlusion, perspective, and differences between the cardboard model and a real building limit what can be inferred.

## Future work — not implemented

- Mount the sensing package on a drone to view multiple sides of a building. Flight vibration, moving-camera compensation, georeferencing, and operating rules would need separate engineering and validation.
- Save time-stamped observations and building identifiers in a city-scale database so repeated assessments can be compared. This would require an approved data model, access controls, retention rules, and a way to distinguish measurements from expert assessments.
- Validate against measured ground truth across camera placements, excitation conditions, and real structures before considering operational use.

## Project records

- [Project brief](docs/project/earthquake-stability.md) and [project draft](docs/project/project-draft.md)
- [Code and Pi handoff](code/README.md)
- [Research and evidence](docs/research/evidence.md)
- [Hackathon logistics](docs/logistics/hackathon.md)
- [Team collaboration](CONTRIBUTING.md)
