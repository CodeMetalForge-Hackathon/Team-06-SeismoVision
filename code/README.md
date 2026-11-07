# Implementation

The teammate-owned Raspberry Pi application is in [`seismovision`](../seismovision/README.md).

The additive [`motion_prototype`](motion_prototype/README.md) experiment measures a grid
of image displacements relative to an independent reference, with camera translation,
rotation and scale compensation. It includes controlled-image tests and a video runner.
It does not change the teammate's device application and is not yet integrated into it
or a website. Hardware performance and field accuracy remain unverified.

See the [verification record](motion_prototype/VERIFICATION.md) for observed checks.

Device integration should document:

- Hardware and exact camera and sensor models.
- Operating system, dependencies, and installation commands.
- How to run the camera pipeline and the three demo conditions.
- Calibration, reference targets, output units, and failure states.
- Build or test commands and observed results.

The owner may choose a different code layout. If so, update this file and the root README rather than moving working code without coordination.
