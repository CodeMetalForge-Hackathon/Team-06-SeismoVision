from __future__ import annotations

import argparse
from collections import deque
from typing import Optional

import cv2
import numpy as np

from .analysis import RollingVibrationAnalyzer, choose_stronger_axis, evaluate
from .capture import make_source
from .config import AppConfig, load_config
from .evidence import EvidenceWriter
from .magnification import RoiMotionMagnifier
from .tracking import MotionTracker


def _draw_history_graph(image: np.ndarray, history: dict, max_frequency_hz: float) -> None:
    panel_width, panel_height = min(360, image.shape[1] - 16), 138
    left, top = image.shape[1] - panel_width - 8, image.shape[0] - panel_height - 30
    cv2.rectangle(image, (left, top), (left + panel_width, top + panel_height), (20, 20, 20), cv2.FILLED)
    colors = ((0, 255, 255), (255, 255, 0), (255, 0, 255))
    magnitudes = [value for values in history.values() for value, _ in values]
    amplitude_limit = max(1.0, max(magnitudes, default=0.0))
    graphs = (("|displacement| px", 0, amplitude_limit), ("frequency Hz", panel_width // 2, max_frequency_hz))
    for title, offset, limit in graphs:
        graph_left, graph_right = left + offset + 6, left + offset + panel_width // 2 - 6
        graph_top, graph_bottom = top + 20, top + panel_height - 18
        cv2.rectangle(image, (graph_left, graph_top), (graph_right, graph_bottom), (80, 80, 80), 1)
        cv2.putText(image, title, (graph_left, top + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (230, 230, 230), 1)
        for color, values in zip(colors, history.values()):
            series = [magnitude if offset == 0 else frequency for magnitude, frequency in values]
            points = []
            for index, value in enumerate(series):
                if value is None:
                    continue
                x = graph_left + int((graph_right - graph_left) * index / max(1, len(series) - 1))
                y = graph_bottom - int((graph_bottom - graph_top) * min(value, limit) / limit)
                points.append((x, y))
            if len(points) > 1:
                cv2.polylines(image, [np.asarray(points, dtype=np.int32)], False, color, 1)
        cv2.putText(image, f"0-{limit:.1f}", (graph_left, graph_bottom + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (190, 190, 190), 1)


def _annotate(frame: np.ndarray, config: AppConfig, reports: dict, reference_motion: np.ndarray, history: dict) -> np.ndarray:
    output = frame.copy()
    x, y, w, h = config.reference_roi
    cv2.rectangle(output, (x, y), (x + w, y + h), (255, 180, 0), 2)
    cv2.putText(output, "STATIC REFERENCE", (x, max(18, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 180, 0), 1)
    for target in config.targets:
        report = reports[target.name]
        state = report["decision"].state
        color = (0, 180, 0) if state == "NORMAL_RELATIVE_TO_BASELINE" else (0, 180, 255) if state == "NO_RESULT" else (0, 0, 255)
        tx, ty, tw, th = target.roi
        cv2.rectangle(output, (tx, ty), (tx + tw, ty + th), color, 2)
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
    _draw_history_graph(output, history, config.analysis.max_frequency_hz)
    cv2.putText(output, "SCREENING ONLY - NOT A SAFETY CERTIFICATION", (12, output.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(output, f"reference motion: {np.linalg.norm(reference_motion):.2f}px", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return output


def run(config: AppConfig, mode: str, video: Optional[str]) -> None:
    source = make_source(config.camera, mode, video)
    writer: Optional[EvidenceWriter] = None
    tracker = MotionTracker(config.reference_roi, config.targets, config.analysis.minimum_track_points)
    horizontal_analyzers = {target.name: RollingVibrationAnalyzer(config.analysis) for target in config.targets}
    vertical_analyzers = {target.name: RollingVibrationAnalyzer(config.analysis) for target in config.targets}
    # RegionFlow reports inter-frame movement. Integrate it so analysis sees displacement,
    # not frame-to-frame velocity. Re-zero this on process restart; baseline recordings
    # should use the same camera geometry as monitoring.
    cumulative_displacement = {target.name: np.zeros(2, dtype=float) for target in config.targets}
    history = {target.name: deque(maxlen=config.camera.fps * 10) for target in config.targets}
    magnifier: Optional[RoiMotionMagnifier] = None
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
            camera_motion = float(np.linalg.norm(reference_motion))
            for target in config.targets:
                movement = targets[target.name]
                # Frequency is calculated on both image axes and reported from the stronger one.
                cumulative_displacement[target.name] += movement.relative_displacement
                position = cumulative_displacement[target.name]
                horizontal_analyzers[target.name].add(timestamp, float(position[0]), movement.confidence)
                vertical_analyzers[target.name].add(timestamp, float(position[1]), movement.confidence)
                axis, metrics = choose_stronger_axis(horizontal_analyzers[target.name].metrics(), vertical_analyzers[target.name].metrics())
                history[target.name].append((float(np.linalg.norm(position)), metrics.dominant_hz))
                reports[target.name] = {"metrics": metrics, "axis": axis, "decision": evaluate(target, metrics, camera_motion, config.analysis), "motion": movement, "position": position.copy()}
            preview = _annotate(frame, config, reports, reference_motion, history)
            magnified = magnifier.apply(preview, timestamp) if magnifier else preview
            record = {
                "timestamp_monotonic": timestamp,
                "camera_reference_motion_px": camera_motion,
                "targets": {
                    name: {
                        "relative_displacement_px": reports[name]["motion"].relative_displacement.tolist(),
                        "displacement_vector_px": reports[name]["position"].tolist(),
                        "displacement_magnitude_px": float(np.linalg.norm(reports[name]["position"])),
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
                cv2.imshow("SeismoVision", magnified)
                if cv2.waitKey(1) & 0xFF == ord("q"):
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
