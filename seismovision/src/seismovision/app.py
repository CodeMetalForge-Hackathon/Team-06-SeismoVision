from __future__ import annotations

import argparse
from collections import deque
from typing import Optional

import cv2
import numpy as np

from .analysis import LowPassVector, RollingVibrationAnalyzer, choose_stronger_axis, evaluate
from .capture import make_source
from .config import AppConfig, load_config
from .evidence import EvidenceWriter
from .magnification import RoiMotionMagnifier
from .tracking import MotionTracker

SPECTROGRAM_BINS = 64
SPECTROGRAM_INTERVAL_SECONDS = 0.25


def _draw_line_graph(image: np.ndarray, bounds: tuple[int, int, int, int], title: str, series: dict, limit: float, colors: tuple) -> None:
    left, top, right, bottom = bounds
    cv2.rectangle(image, (left, top), (right, bottom), (80, 80, 80), 1)
    cv2.putText(image, title, (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (230, 230, 230), 1)
    for color, values in zip(colors, series.values()):
        points = []
        for index, value in enumerate(values):
            if value is None:
                continue
            x = left + int((right - left) * index / max(1, len(values) - 1))
            y = bottom - int((bottom - top) * min(value, limit) / limit)
            points.append((x, y))
        if len(points) > 1:
            cv2.polylines(image, [np.asarray(points, dtype=np.int32)], False, color, 1)
    cv2.putText(image, f"0-{limit:.1f}", (left, bottom + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (190, 190, 190), 1)


def _draw_spectrogram(image: np.ndarray, bounds: tuple[int, int, int, int], title: str, values: deque, min_frequency_hz: float, max_frequency_hz: float, intensity_limit: float) -> None:
    left, top, right, bottom = bounds
    cv2.rectangle(image, (left, top), (right, bottom), (80, 80, 80), 1)
    cv2.putText(image, title, (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (230, 230, 230), 1)
    if values and intensity_limit > 0:
        matrix = np.asarray(values, dtype=np.float32).T
        pixels = np.uint8(np.clip(matrix / intensity_limit, 0.0, 1.0) * 255)
        colored = cv2.applyColorMap(np.flipud(pixels), cv2.COLORMAP_TURBO)
        image[top:bottom, left:right] = cv2.resize(colored, (right - left, bottom - top), interpolation=cv2.INTER_NEAREST)
    cv2.putText(image, f"{max_frequency_hz:.1f}Hz", (left + 2, top + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (230, 230, 230), 1)
    cv2.putText(image, f"{min_frequency_hz:.1f}Hz", (left + 2, bottom - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (230, 230, 230), 1)
    cv2.putText(image, "past -> now", (right - 58, bottom - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (230, 230, 230), 1)


def _draw_dashboard(width: int, history: dict, delta_history: deque, spectrograms: dict, min_frequency_hz: float, max_frequency_hz: float) -> np.ndarray:
    panel_height = 330
    image = np.full((panel_height, width, 3), 20, dtype=np.uint8)
    colors = ((0, 255, 255), (255, 255, 0), (255, 0, 255))
    raw_magnitudes = [raw for values in history.values() for raw, _, _ in values]
    smoothed_magnitudes = [smoothed for values in history.values() for _, smoothed, _ in values]
    raw_limit = max(1.0, max(raw_magnitudes, default=0.0))
    smoothed_limit = max(1.0, max(smoothed_magnitudes, default=0.0))
    half_width = width // 2
    top_inset = lambda column, row: (column * half_width + 6, row * 80 + 18, (column + 1) * half_width - 6, (row + 1) * 80 - 17)
    _draw_line_graph(image, top_inset(0, 0), "|displacement| raw px", {name: [raw for raw, _, _ in values] for name, values in history.items()}, raw_limit, colors)
    _draw_line_graph(image, top_inset(1, 0), "|displacement| LP px", {name: [smoothed for _, smoothed, _ in values] for name, values in history.items()}, smoothed_limit, colors)
    _draw_line_graph(image, top_inset(0, 1), "frequency Hz", {name: [frequency for _, _, frequency in values] for name, values in history.items()}, max_frequency_hz, colors)
    delta_limit = max(1.0, max(delta_history, default=0.0))
    _draw_line_graph(image, top_inset(1, 1), "roof-mid delta LP px", {"delta": delta_history}, delta_limit, ((0, 165, 255),))
    intensity_limit = max((float(np.max(values)) for values in spectrograms.values() if values), default=0.0)
    spectrogram_bounds = lambda column: (column * half_width + 6, 180, (column + 1) * half_width - 6, panel_height - 18)
    for index, (name, values) in enumerate(spectrograms.items()):
        if index == 2:
            break
        _draw_spectrogram(image, spectrogram_bounds(index), f"{name} spectrogram", values, min_frequency_hz, max_frequency_hz, intensity_limit)
    return image


def _annotate(frame: np.ndarray, config: AppConfig, reports: dict, reference_motion: np.ndarray, reference_points: np.ndarray) -> np.ndarray:
    output = frame.copy()
    x, y, w, h = config.reference_roi
    cv2.rectangle(output, (x, y), (x + w, y + h), (255, 180, 0), 2)
    cv2.putText(output, "STATIC REFERENCE", (x, max(18, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 180, 0), 1)
    for point in reference_points:
        cv2.circle(output, tuple(point.astype(int)), 3, (255, 180, 0), cv2.FILLED)
    for target in config.targets:
        report = reports[target.name]
        state = report["decision"].state
        color = (0, 180, 0) if state == "NORMAL_RELATIVE_TO_BASELINE" else (0, 180, 255) if state == "NO_RESULT" else (0, 0, 255)
        tx, ty, tw, th = target.roi
        cv2.rectangle(output, (tx, ty), (tx + tw, ty + th), color, 2)
        for point in report["motion"].points:
            cv2.circle(output, tuple(point.astype(int)), 3, color, cv2.FILLED)
        metric = report["metrics"]
        hz = "--" if metric.dominant_hz is None else f"{metric.dominant_hz:.2f}Hz"
        label = f"{target.name}: {state} {hz}"
        cv2.putText(output, label, (tx, max(18, ty - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2)
        center = (tx + tw // 2, ty + th // 2)
        vector = report["position"]
        visible_vector = np.clip(vector * 20.0, -45.0, 45.0).astype(int)
        endpoint = (int(center[0] + visible_vector[0]), int(center[1] + visible_vector[1]))
        cv2.circle(output, center, 4, color, cv2.FILLED)
        cv2.arrowedLine(output, center, endpoint, color, 2, tipLength=0.25)
        cv2.putText(output, f"x {vector[0]:+.2f}, y {vector[1]:+.2f}px", (tx + 4, ty + th // 2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        cv2.putText(output, f"|d|LP {report['smoothed_magnitude']:.2f}px  rate {report['smoothed_rate']:+.2f}px/s", (tx + 4, ty + th // 2 + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
    cv2.putText(output, "SCREENING ONLY - NOT A SAFETY CERTIFICATION", (12, output.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(output, f"reference motion: {np.linalg.norm(reference_motion):.2f}px", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(output, "R: recalibrate tracking", (12, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
    return output


def run(config: AppConfig, mode: str, video: Optional[str]) -> None:
    source = make_source(config.camera, mode, video)
    writer: Optional[EvidenceWriter] = None
    tracker = MotionTracker(config.reference_roi, config.targets, config.analysis.minimum_track_points)
    horizontal_analyzers = {target.name: RollingVibrationAnalyzer(config.analysis) for target in config.targets}
    vertical_analyzers = {target.name: RollingVibrationAnalyzer(config.analysis) for target in config.targets}
    magnitude_filters = {target.name: LowPassVector() for target in config.targets}
    # RegionFlow reports inter-frame movement. Integrate it so analysis sees displacement,
    # not frame-to-frame velocity. Re-zero this on process restart; baseline recordings
    # should use the same camera geometry as monitoring.
    cumulative_displacement = {target.name: np.zeros(2, dtype=float) for target in config.targets}
    history = {target.name: deque(maxlen=config.camera.fps * 10) for target in config.targets}
    delta_history = deque(maxlen=config.camera.fps * 10)
    spectrograms = {target.name: deque(maxlen=120) for target in config.targets}
    last_spectrogram_timestamp: Optional[float] = None
    magnifier: Optional[RoiMotionMagnifier] = None
    camera_was_moved = False
    settled_frames = 0

    def recalibrate() -> None:
        nonlocal last_spectrogram_timestamp
        tracker.reset()
        for analyzer in (*horizontal_analyzers.values(), *vertical_analyzers.values()):
            analyzer.samples.clear()
        for magnitude_filter in magnitude_filters.values():
            magnitude_filter.reset()
        for position in cumulative_displacement.values():
            position.fill(0.0)
        for values in history.values():
            values.clear()
        delta_history.clear()
        for values in spectrograms.values():
            values.clear()
        last_spectrogram_timestamp = None
        if magnifier is not None:
            magnifier.reset()
    try:
        while True:
            ok, frame, timestamp = source.read()
            if not ok or frame is None:
                break
            if writer is None:
                tracker.validate(frame)
                frame_size = (frame.shape[1], frame.shape[0])
                writer = EvidenceWriter(config.output.directory, config.to_dict(), frame_size, config.camera.fps, config.output.save_video)
                if config.magnification.enabled:
                    selected = next(target for target in config.targets if target.name == config.magnification.target)
                    magnifier = RoiMotionMagnifier(selected.roi, config.magnification, config.camera.fps)
            reference_motion, targets = tracker.update(frame)
            reports = {}
            spectra = {}
            camera_motion = float(np.linalg.norm(reference_motion))
            if camera_motion > config.analysis.max_camera_motion_px:
                camera_was_moved, settled_frames = True, 0
            elif camera_was_moved:
                settled_frames += 1
                if settled_frames >= config.analysis.camera_settle_frames:
                    recalibrate()
                    camera_was_moved, settled_frames = False, 0
                    continue
            for target in config.targets:
                movement = targets[target.name]
                # Frequency is calculated on both image axes and reported from the stronger one.
                cumulative_displacement[target.name] += movement.relative_displacement
                position = cumulative_displacement[target.name]
                horizontal_analyzers[target.name].add(timestamp, float(position[0]), movement.confidence)
                vertical_analyzers[target.name].add(timestamp, float(position[1]), movement.confidence)
                axis, metrics = choose_stronger_axis(horizontal_analyzers[target.name].metrics(), vertical_analyzers[target.name].metrics())
                selected_analyzer = horizontal_analyzers[target.name] if axis == "horizontal" else vertical_analyzers[target.name]
                spectra[target.name] = selected_analyzer.spectrum()
                smoothed = magnitude_filters[target.name].update(timestamp, position)
                history[target.name].append((float(np.linalg.norm(position)), smoothed.magnitude_px, metrics.dominant_hz))
                reports[target.name] = {"metrics": metrics, "axis": axis, "decision": evaluate(target, metrics, camera_motion, config.analysis), "motion": movement, "position": position.copy(), "smoothed_magnitude": smoothed.magnitude_px, "smoothed_rate": smoothed.magnitude_rate_px_s, "smoothed_vector": smoothed.vector}
            roof = next((report for name, report in reports.items() if "roof" in name.lower()), None)
            mid_facade = reports.get("mid_facade")
            if roof is not None and mid_facade is not None:
                delta_history.append(float(np.linalg.norm(roof["smoothed_vector"] - mid_facade["smoothed_vector"])))
            if last_spectrogram_timestamp is None or timestamp - last_spectrogram_timestamp >= SPECTROGRAM_INTERVAL_SECONDS:
                frequency_grid = np.linspace(config.analysis.min_frequency_hz, config.analysis.max_frequency_hz, SPECTROGRAM_BINS)
                for name, (frequencies, powers) in spectra.items():
                    intensity = np.interp(frequency_grid, frequencies, powers, left=0.0, right=0.0) if len(frequencies) else np.zeros(SPECTROGRAM_BINS)
                    spectrograms[name].append(np.log1p(intensity))
                last_spectrogram_timestamp = timestamp
            preview = _annotate(frame, config, reports, reference_motion, tracker.reference_points)
            magnified = magnifier.apply(preview, timestamp) if magnifier else preview
            record = {
                "timestamp_monotonic": timestamp,
                "camera_reference_motion_px": camera_motion,
                "targets": {
                    name: {
                        "relative_displacement_px": reports[name]["motion"].relative_displacement.tolist(),
                        "displacement_vector_px": reports[name]["position"].tolist(),
                        "displacement_magnitude_px": float(np.linalg.norm(reports[name]["position"])),
                        "smoothed_displacement_magnitude_px": reports[name]["smoothed_magnitude"],
                        "smoothed_magnitude_rate_px_s": reports[name]["smoothed_rate"],
                        "tracking_confidence": reports[name]["motion"].confidence,
                        "tracked_points": reports[name]["motion"].point_count,
                        "sample_count": reports[name]["metrics"].sample_count,
                        "dominant_hz": reports[name]["metrics"].dominant_hz,
                        "dominant_frequency_axis": reports[name]["axis"],
                        "peak_displacement_px": reports[name]["metrics"].peak_displacement_px,
                        "decision": reports[name]["decision"].state,
                        "reasons": reports[name]["decision"].reasons,
                    }
                    for name in reports
                },
            }
            writer.write(record, frame, magnified)
            if config.output.show_preview:
                cv2.imshow("SeismoVision", cv2.vconcat((magnified, _draw_dashboard(frame.shape[1], history, delta_history, spectrograms, config.analysis.min_frequency_hz, config.analysis.max_frequency_hz))))
                key = cv2.waitKey(1) & 0xFF
                if key == ord("r"):
                    recalibrate()
                elif key == ord("q"):
                    break
    finally:
        source.close()
        if writer is not None:
            writer.close()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Conservative structural-response screening camera prototype")
    parser.add_argument("--config", required=True, help="JSON configuration")
    parser.add_argument("--camera", choices=("picamera2", "usb"), default="picamera2")
    parser.add_argument("--video", help="Use a recorded video instead of a camera")
    args = parser.parse_args()
    run(load_config(args.config), args.camera, args.video)


if __name__ == "__main__":
    main()
