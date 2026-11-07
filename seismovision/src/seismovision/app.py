from __future__ import annotations

import argparse
import json
import time
from typing import Optional

import cv2
import numpy as np

from .analysis import RollingVibrationAnalyzer, evaluate
from .capture import make_source
from .config import AppConfig, load_config
from .evidence import EvidenceWriter
from .magnification import RoiMotionMagnifier
from .tracking import MotionTracker


def _annotate(frame: np.ndarray, config: AppConfig, reports: dict, reference_motion: np.ndarray) -> np.ndarray:
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
    cv2.putText(output, "SCREENING ONLY - NOT A SAFETY CERTIFICATION", (12, output.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(output, f"reference motion: {np.linalg.norm(reference_motion):.2f}px", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return output


def run(config: AppConfig, mode: str, video: Optional[str]) -> None:
    source = make_source(config.camera, mode, video)
    writer: Optional[EvidenceWriter] = None
    tracker = MotionTracker(config.reference_roi, config.targets, config.analysis.minimum_track_points)
    analyzers = {target.name: RollingVibrationAnalyzer(config.analysis) for target in config.targets}
    # RegionFlow reports inter-frame movement. Integrate it so analysis sees displacement,
    # not frame-to-frame velocity. Re-zero this on process restart; baseline recordings
    # should use the same camera geometry as monitoring.
    cumulative_displacement = {target.name: np.zeros(2, dtype=float) for target in config.targets}
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
                # Horizontal is the default observed axis; deployers may rotate/calibrate per camera.
                cumulative_displacement[target.name] += movement.relative_displacement
                analyzers[target.name].add(timestamp, float(cumulative_displacement[target.name][0]), movement.confidence)
                metrics = analyzers[target.name].metrics()
                reports[target.name] = {"metrics": metrics, "decision": evaluate(target, metrics, camera_motion, config.analysis), "motion": movement}
            preview = _annotate(frame, config, reports, reference_motion)
            magnified = magnifier.apply(preview, timestamp) if magnifier else preview
            record = {
                "timestamp_monotonic": timestamp,
                "camera_reference_motion_px": camera_motion,
                "targets": {
                    name: {
                        "relative_displacement_px": reports[name]["motion"].relative_displacement.tolist(),
                        "tracking_confidence": reports[name]["motion"].confidence,
                        "tracked_points": reports[name]["motion"].point_count,
                        "sample_count": reports[name]["metrics"].sample_count,
                        "dominant_hz": reports[name]["metrics"].dominant_hz,
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
