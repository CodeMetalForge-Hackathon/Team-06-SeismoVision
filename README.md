# Earthquake Stability

An experimental, camera-first structural motion indicator for search and rescue teams. This is Team 6's project for [Code Metal Forge 2026](https://www.codemetal.ai/forge-hackathon), focused on **Objective 02: Detect Life and Hazards**.

After an earthquake or other damaging event, responders may need to work near a structure whose condition is uncertain. The proposed field unit observes motion from a distance and presents measurements that could help responders notice a possible structural hazard. It is designed around edge processing when normal infrastructure is unavailable.

**Status:** the project plan and demo scenarios are documented here. An experimental Raspberry Pi prototype is available in [`seismovision/`](seismovision/). Nothing in this repository establishes a building-entry decision or predicts collapse.

## Start here

- [Project brief](docs/project/earthquake-stability.md): problem, proposed system, hardware, and limits.
- [Project writeup](docs/project/project-draft.md): research-backed draft and 640 × 480, 30 fps camera baseline.
- [Demo runbook](docs/demo/runbook.md): planned physical demonstration and validation record.
- [Research and evidence](docs/research/evidence.md): relevant methods and what they do and do not establish.
- [Hackathon logistics](docs/logistics/hackathon.md): official rubric, timeline, and submission requirements.
- [Brainstorming sources](docs/brainstorm/README.md): live Google Doc and dated backup.
- [Team collaboration](CONTRIBUTING.md): GitHub and Codex setup for all four teammates.
- [Code folder](code/README.md): implementation overview and handoff notes.

## Planned live demonstration

A robot oscillates a cardboard structure while the camera observes it. The team plans to compare three conditions: an isolated camera, a camera coupled to the structure, and deliberately added camera vibration. The key question is whether the system can distinguish motion of the target from motion of the camera. A Boston building video may be used as observational footage, not as a safety assessment.

## Interpretation

The prototype may report measured or concerning motion, no concerning motion observed, or an unreliable measurement. **No observed motion is not proof that a structure is safe.** Entry clearance and collapse prediction require validated, building-specific methods and qualified assessment. See [research and evidence](docs/research/evidence.md).

## Submission

The [Saturday mission brief](docs/logistics/archive/forge-2026-saturday-mission-brief.pdf) says the team repository is the submission and all code must be pushed by **Sunday, November 8, 2026 at 12:00 p.m. Eastern**. A live table demo is recommended; a backup video is optional.
