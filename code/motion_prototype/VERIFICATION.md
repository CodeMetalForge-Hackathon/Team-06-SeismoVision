# Verification Report

## What was checked

Additive motion algorithm on branch `feature/reference-motion-algorithm`, based on
repository commit `5d15ae5`. Local Python 3.9, NumPy 2.0.2, OpenCV 5.0.0.
The existing `seismovision` application was read and left unchanged.

## Commands / evidence

Run from repository root with the virtual environment described in README.

| Check | Evidence | Result |
| --- | --- | --- |
| New image tests | `PYTHONPATH=code/motion_prototype .venv/bin/python -m unittest discover -s code/motion_prototype/tests -v`: 11 tests pass | PASS |
| Existing analysis tests | `PYTHONPATH=seismovision/src .venv/bin/python -m unittest discover -s seismovision/tests -v`: 3 tests pass | PASS |
| Synthetic sequence | `demo.py --output /private/tmp/team6-motion-demo`: 360 frames, four explicitly synthetic scenarios | PASS |
| Camera-only sequence | All 90 frames below configured motion threshold | PASS |
| Video runner | `run.py --video /private/tmp/team6-motion-demo/synthetic-input.avi --building 210 60 180 180 --reference 20 30 150 240 --output /private/tmp/team6-motion-replay`: 360 frames processed | PASS |
| Encoded replay behavior | Same overall state counts as synthetic source: 1 initialization/unreliable, 194 below threshold, 165 above threshold | PASS |
| Output video decode | Both motion-map AVI files decode to 360 frames of 440x300 | PASS |
| Preview | Inspected `preview.png`: top row about 3 px, remaining cells about 0.01–0.02 px during roof-only motion and camera shake; synthetic label visible | PASS |
| Patch formatting | `git diff --check` | PASS |

Tests cover idle frames, known X/Y displacement, camera translation plus rotation,
simultaneous target/camera motion, mechanically coupled apparent motion with an independent
reference, roof localization, reference loss and recovery, textureless images, target
occlusion, a 1.25 Hz signal and return to the anchor, and invalid inputs.

The synthetic demo measured about 6.12 ms mean processing time per frame on this local
host. This excludes capture, image generation, drawing and encoding. It is not a Pi benchmark.
Evidence files are local `/private/tmp` artifacts, not checked-in field measurements.

## Issues found and fixed

- Initial dependency installation was blocked by sandbox network access; installation
  succeeded after the approved elevated retry.
- The existing video source uses processing timestamps. The separate experimental replay
  runner uses media time (frame index / FPS) so processing speed cannot alter frequency.
- No changes were made to the teammate-owned application; these improvements remain a
  separate integration candidate.

## Remaining unverified areas

Pi CPU/thermal behavior, CSI camera path, real cardboard trials, lighting/blur robustness,
outdoor footage, physical calibration, sensor fusion and web integration. No automatic
building segmentation, reproduced MEVM implementation, validated modal identification,
or collapse/entry-safety decision is claimed. Default thresholds need setup-specific tests.

## Final status

PARTIAL — implemented and verified on controlled synthetic images and encoded replay;
device integration and physical-scene validation remain outstanding.
