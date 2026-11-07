# Experimental reference-compensated motion map

Additive algorithm experiment, separate from the teammate-owned `seismovision` application.
No cloud, model downloads, or network calls at runtime. Uses Python, NumPy and OpenCV.
Pi performance is untested. No hardware accelerator required.

## Run locally

From the repository root:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install numpy opencv-python
PYTHONPATH=code/motion_prototype .venv/bin/python -m unittest discover -s code/motion_prototype/tests -v
.venv/bin/python code/motion_prototype/demo.py --output /tmp/team6-demo-new
.venv/bin/python code/motion_prototype/run.py \
  --video /tmp/team6-demo-new/synthetic-input.avi \
  --building 210 60 180 180 --reference 20 30 150 240 \
  --output /tmp/team6-replay-new
```

Outputs must be new directories. The demo creates explicitly synthetic textured planar
scenes: idle, camera-only motion, target plus camera motion, then roof-only plus camera
motion. It saves input/overlay AVI files, a preview, JSONL measurements and local timing.
The runner saves measurements, effective arguments and a motion-map AVI without a GUI.
Use `--camera 0` instead of `--video` for an OpenCV-compatible USB camera. Native Pi CSI
capture is not wired into this runner. Supply original-resolution BGR frames directly
from the owner's capture adapter when integrating; no rescaling is performed here.

Choose your own rectangles for real video: `x y width height`. The example coordinates
only fit the generated demo. Start with a sharp, stationary frame. Reference and target
must not overlap. Use an independently mounted textured reference visible alongside the
model, with similar depth to reduce parallax. A 3x3 grid is the default.

## Math and behavior

1. `goodFeaturesToTrack` selects corners within the reference and each building cell.
2. Pyramidal Lucas–Kanade tracks each anchor-frame point `p` into the current frame `q`.
   Forward/backward checks and patch-error limits reject poor matches.
3. Fit the reference points with RANSAC: `q ≈ s R p + t`, where `s` is scale,
   `R` is 2D rotation and `t` is translation. Building points never enter this fit.
4. Each building point's residual is `r = q - (s R p + t)`, in current-image pixels.
   Take coordinate-wise medians within each cell; magnitude is `sqrt(dx² + dy²)`.
   These are offsets from the initial frame, not inter-frame velocities or integrated
   displacement. A displaced target can remain highlighted after it stops moving.
5. A rolling timestamped signal gives an exploratory dominant frequency using uniform
   interpolation, linear detrending, a Hann window and summed X/Y Fourier power.
   Require at least 90 samples, three seconds, three cycles in-window, measurable
   amplitude, and a spectral peak above the band median. Return null otherwise.

The default `0.75 px` display threshold is an **unvalidated demo setting**, not a learned
noise baseline or building damage threshold. Use `--threshold-px` after measuring idle
noise and known movement in your actual setup. Frequency is an exploratory diagnostic;
this implementation does not prove periodicity or estimate modal damage.

Cells return `MOTION_ABOVE_THRESHOLD`, `NO_MOTION_ABOVE_THRESHOLD`, or `UNRELIABLE`.
Unknown cells carry null measurements, never invented zeros. Reference loss invalidates
all cells and clears temporal history. Partial target loss stays visible per cell; a
measured moving cell can still raise the overall motion state while others are unknown.
Reference residual, inlier count, coverage, tracked-point survival and within-cell spread
are checked. These are consistency checks, not a probability of correctness.

The anchor never silently resets. Create a new `MotionMap` after moving the camera or
changing the scene. Small temporary occlusions can recover against the same anchor.
Large target shifts (20 px), rotation (5 degrees), scale changes (5 percent), and poor
fits are outside the intended operating range and produce unreliable measurements.

## Web / edge integration contract

```python
from motion import MotionMap
tracker = MotionMap(building=(210, 60, 180, 180), reference=(20, 30, 150, 240))
report = tracker.update(bgr_frame, capture_timestamp_seconds)
# Send report as JSON from the Pi to the local dashboard.
```

Use one tracker per capture stream, called serially. Timestamps must strictly increase.
Live frames need capture timestamps; constant-frame-rate replay uses frame index / FPS.
Variable-frame-rate media must first be converted to a known cadence. Invalid shapes,
overlapping selections and non-monotonic timestamps raise `ValueError`.
UI selection, streaming server, polygon segmentation, CSI capture and sensor fusion
are not implemented here. There is no automatic semantic building detection.

## Evidence and limits

See [research](../../docs/research/evidence.md) for the two requested papers and the
distinction between published methods and this implementation. This prototype is not
a reproduction of either paper. Measurements come from unamplified frames.

A coherent moving reference can look like camera movement. Common camera/target motion
is unobservable without an independent reference. Depth differences, out-of-plane camera
rotation, rolling shutter, changing illumination, repeating textures and small local
deformations can defeat or bias tracking. The grid summarizes coherent image movement;
it is not a continuous contour map, physical strain map or map of structural weakness.

No pixel output certifies safe entry, identifies the weakest part of a building, or
predicts collapse. Physical displacement requires geometric calibration; structural
inference requires building-specific evidence and qualified assessment. Synthetic tests
exercise implementation behavior, not real-world structural validity.
