# Implementation

The Raspberry Pi prototype currently lives in [`../seismovision/`](../seismovision/). The `optimized_algorithm` branch adds automatic cardboard localization to the interactive map, preserving the lower 4 by 2 grid and adding two upper-box cells. Its outputs are experimental measurements, not a building-entry decision or collapse prediction.

Run and hardware notes are maintained in [`../seismovision/README.md`](../seismovision/README.md). Pi 5 performance and vibration accuracy require a camera recording and measured ground truth.

Keep this handoff updated with:

- Hardware and exact camera and sensor models.
- Operating system, dependencies, and installation commands.
- How to run the camera pipeline and the three demo conditions.
- Calibration, reference targets, output units, and failure states.
- Build or test commands and observed results.

The owner may choose a different code layout. If so, update this file and the root README rather than moving working code without coordination.
