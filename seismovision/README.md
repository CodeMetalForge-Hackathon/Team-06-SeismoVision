# SeismoVision

SeismoVision is a conservative, camera-based **structural-response screening** prototype. It tracks configured visual targets on a building, compensates for camera motion with a fixed reference target, estimates dominant vibration frequency and relative displacement, and provides a Modified Eulerian Video Magnification (MEVM) inspection view.

It is not a building-safety certification system. Its only alert is **inspection recommended**; a qualified structural engineer must establish the baseline, thresholds, mounting plan, and response procedure.

## What is implemented

- Pi Camera / USB / video-file capture behind a portable `FrameSource` interface.
- Fixed reference-region tracking to reject apparent motion from camera movement.
- Automatic feature reacquisition after camera relocation, soft-shadow normalization, and visible tracking points.
- Multiple building target regions tracked with Lucas-Kanade optical flow.
- Rolling, timestamp-aware frequency and peak-displacement estimates.
- Centered X/Y displacement arrows; low-pass displacement magnitude/rate; an extended live dashboard with raw/filtered magnitude, frequency, roof-to-mid-facade difference, and two color spectrograms.
- A calibrated roof-to-mid-facade frequency-difference indicator that can recommend inspection; it never certifies a structure safe or unstable.
- A conservative baseline comparison and data-quality gate.
- ROI-only Laplacian-pyramid temporal filtering for MEVM-style visualization.
- JSON-lines evidence log and optional normal/magnified MP4 files.
- Synthetic unit tests for the signal analysis and decision policy.

The system intentionally uses configured regions instead of automatically selecting the largest moving object. In outdoor building footage that object is often a tree, vehicle, or shadow.

## Pi 4 quick start

Use 64-bit Raspberry Pi OS, a cooled Pi 4, a stable supply, and current `rpicam`/libcamera software. Install the Python packages in a virtual environment:

```bash
sudo apt update
sudo apt install -y python3-venv python3-opencv python3-numpy
python3 -m venv .venv --system-site-packages
. .venv/bin/activate
pip install -e . --no-deps
```

Copy and edit the example configuration:

```bash
cp config.example.json config.json
python -m seismovision --config config.json --camera picamera2
```

For development without a Pi camera, use a video file:

```bash
python -m seismovision --config config.json --video path/to/recording.mp4
```

Press `q` to stop when the preview is enabled.

## Commissioning checklist

1. Rigidly mount the camera to an independent stable support, not to the measured façade.
2. Include a non-moving visual reference target in `reference_roi`.
3. Add high-contrast targets at known positions on the building and enter their rectangles in `targets`.
4. Lock focus, exposure, gain, and white balance after the scene is correctly exposed. Auto-exposure can resemble motion.
5. Record baseline data during known healthy operation and have an engineer set `baseline_hz`, tolerances, and any displacement limit.
6. Validate against an accelerometer/IMU and controlled excitation before interpreting alerts.

## Output interpretation

`NO_RESULT` means the data are not trustworthy (for example: camera movement, insufficient tracked points, or not enough samples). `NORMAL_RELATIVE_TO_BASELINE` means no configured anomaly was observed; it does **not** mean safe. `INSPECTION_RECOMMENDED` means the configured signal limits were exceeded with acceptable data quality.

If the camera is moved, the app waits for `camera_settle_frames` stable frames and then automatically reacquires all features, clears displacement/frequency histories, and starts a new measurement baseline. Press `r` in the local preview to do this manually. Dots show the active optical-flow features: cyan dots are the stationary reference and target dots match their status-box color.

The app writes a timestamped evidence folder containing `measurements.jsonl`, the effective configuration, and—when enabled—raw and MEVM-view video. Do not use the magnified video as a quantitative measurement; use the logged unamplified displacement signal.

## Porting to Jetson Orin Nano

Keep the configuration and analysis modules unchanged. Replace only the source factory with a CSI/GStreamer source and optionally replace OpenCV CPU primitives with CUDA implementations. The analysis interface accepts `frame: np.ndarray` and a monotonic timestamp, so recorded Pi footage can be replayed unchanged on Jetson for parity testing.

## Tests

```bash
python -m unittest discover -s tests -v
```
