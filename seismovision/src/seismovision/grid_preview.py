"""Live automatically located cardboard motion map."""

import argparse
from dataclasses import replace
from typing import Dict

import cv2
import numpy as np

from .capture import make_source
from .auto_regions import detect_structure, select_background_reference
from .config import AnalysisConfig, CameraConfig, MagnificationConfig, load_grid_config
from .grid_mapping import BuildingGridMapper, GridCellState
from .magnification import RoiMotionMagnifier


def _spectrogram(panel: np.ndarray, bounds: tuple[int, int, int, int], index: int, history: object, max_frequency: float) -> None:
    left, top, right, bottom = bounds
    cv2.putText(panel, f"cell {index}: FFT amplitude", (left + 4, top + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (235, 235, 235), 1)
    graph_top = top + 20
    cv2.rectangle(panel, (left, graph_top), (right, bottom), (100, 100, 100), 1)
    columns = list(history)
    if columns:
        data = np.asarray(columns, dtype=np.float32).T
        limit = float(np.max(np.log1p(data)))
        if limit > 0:
            image = np.uint8(255 * np.log1p(data) / limit)
            image = cv2.applyColorMap(np.flipud(image), cv2.COLORMAP_TURBO)
            image = cv2.resize(image, (right - left - 2, bottom - graph_top - 2), interpolation=cv2.INTER_NEAREST)
            panel[graph_top + 1:bottom - 1, left + 1:right - 1] = image
    cv2.putText(panel, f"{max_frequency:.1f}Hz", (left + 3, graph_top + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (230, 230, 230), 1)
    cv2.putText(panel, "time ->", (right - 48, bottom - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (230, 230, 230), 1)


def _draw_grid(frame: np.ndarray, mapper: BuildingGridMapper, states: Dict[str, GridCellState]) -> np.ndarray:
    output = frame.copy()
    for box in (mapper.bbox, mapper.upper_bbox):
        if box is not None:
            x, y, width, height = box
            cv2.rectangle(output, (x, y), (x + width, y + height), (0, 255, 255), 2)
    for index, state in enumerate(states.values()):
        x, y, width, height = state.roi
        anomaly = state.anomaly_score is not None and state.anomaly_score >= 1.0
        color = (0, 180, 255) if state.motion.confidence <= 0 else (0, 0, 255) if anomaly else (0, 255, 0)
        cv2.rectangle(output, (x, y), (x + width, y + height), color, 2)
        for point in state.motion.points:
            cv2.circle(output, tuple(point.astype(int)), 2, color, cv2.FILLED)
        center = (x + width // 2, y + height // 2)
        vector = np.clip(
            state.position * 20.0,
            (-width // 2 + 5, -height // 2 + 18),
            (width // 2 - 5, height // 2 - 5),
        ).astype(int)
        cv2.arrowedLine(output, center, (center[0] + vector[0], center[1] + vector[1]), (0, 255, 255), 2, tipLength=0.25)
        hz = "--" if state.frequency_hz is None else f"{state.frequency_hz:.2f}Hz"
        label = f"{state.name} {hz}" + (" ANOMALY" if anomaly else "")
        cv2.putText(output, label, (x + 3, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)
        cv2.putText(output, f"{state.position[0]:+.1f}, {state.position[1]:+.1f}px", (x + 3, y + height - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 255, 255), 1)
    score = mapper.response_anomaly_score
    message = "response anomaly: collecting"
    color = (0, 180, 255)
    if score is not None:
        message = f"response anomaly: {score:.2f}x sensitivity"
        color = (0, 0, 255) if score >= 1.0 else (0, 220, 0)
    cv2.putText(output, message, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2)
    cv2.putText(output, "screening only - not a stability prediction", (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (210, 210, 210), 1)
    return output


def _draw_status(width: int, mapper: BuildingGridMapper, states: Dict[str, GridCellState]) -> np.ndarray:
    """Shows a screening result and data quality without claiming structural safety."""
    panel = np.full((86, width, 3), 18, dtype=np.uint8)
    confidences = [state.motion.confidence for state in states.values()]
    confidence = float(np.mean(confidences)) if confidences else 0.0
    reliable = sum(value >= mapper.analysis.minimum_confidence for value in confidences)
    score = mapper.response_anomaly_score
    upper_reliable = any(states[name].motion.confidence >= mapper.analysis.minimum_confidence for name in ("upper_0", "upper_1") if name in states)
    if reliable < 3 or not upper_reliable:
        headline, color = "STABILITY SCREEN: MEASUREMENT UNRELIABLE", (0, 180, 255)
    elif score is None:
        headline, color = "STABILITY SCREEN: COLLECTING RESPONSE DATA", (0, 180, 255)
    elif score >= 1.0:
        headline, color = "STABILITY SCREEN: RESPONSE ANOMALY OBSERVED", (0, 0, 255)
    else:
        headline, color = "STABILITY SCREEN: NO RESPONSE ANOMALY CURRENTLY", (0, 220, 0)
    cv2.putText(panel, headline, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2)
    anomaly = "--" if score is None else f"{score:.2f}x configured sensitivity"
    cv2.putText(panel, f"Measurement confidence: {confidence * 100:.0f}% ({reliable}/{len(states)} reliable cells) | response anomaly: {anomaly}", (12, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 235, 235), 1)
    cv2.putText(panel, "Tracking confidence is not a probability that the building is stable or unstable.", (12, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (190, 190, 190), 1)
    return panel


def main() -> None:
    parser = argparse.ArgumentParser(description="Automatic cardboard-building motion map")
    parser.add_argument("--camera", choices=("picamera2", "usb"), default="picamera2")
    parser.add_argument("--config", help="JSON settings for camera, analysis, and magnification")
    parser.add_argument("--video")
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--fps", type=int)
    args = parser.parse_args()

    settings = load_grid_config(args.config) if args.config else None
    camera = settings.camera if settings else CameraConfig(width=640, height=480, fps=30)
    camera = replace(camera, width=args.width or camera.width, height=args.height or camera.height, fps=args.fps or camera.fps)
    analysis = settings.analysis if settings else replace(AnalysisConfig(), max_camera_motion_px=8.0)
    magnification = settings.magnification if settings else replace(MagnificationConfig(), enabled=False)
    source = make_source(camera, args.camera, args.video)
    mapper = None
    magnifiers = ()
    lost_frames = 0
    try:
        while True:
            ok, frame, timestamp = source.read()
            if not ok or frame is None:
                break
            if mapper is None:
                boxes = detect_structure(frame)
                reference = select_background_reference(frame, boxes.full) if boxes else None
                if boxes is not None and reference is not None:
                    mapper = BuildingGridMapper(boxes.lower, reference, analysis, boxes.upper)
                    mapper.tracker.validate(frame)
                    magnifiers = tuple(RoiMotionMagnifier(roi, magnification, camera.fps) for roi in (*mapper.cells, *mapper.upper_cells)) if magnification.enabled else ()
                    lost_frames = 0
                preview = frame.copy()
                cv2.putText(preview, "Acquiring cardboard and fixed background" if mapper is None else "Acquiring tracking points", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 180, 255), 2)
                cv2.imshow("SeismoVision grid map", preview)
            else:
                camera_motion, states = mapper.update(frame, timestamp)
                upper_ok = any(states[name].motion.confidence > 0 for name in ("upper_0", "upper_1"))
                lower_ok = sum(states[f"cell_{index}"].motion.confidence > 0 for index in range(8)) >= 2
                lost_frames = 0 if upper_ok and lower_ok else lost_frames + 1
                if lost_frames >= 4:
                    mapper, magnifiers = None, ()
                    preview = frame.copy()
                    cv2.putText(preview, "MEASUREMENT UNRELIABLE - reacquiring structure", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 180, 255), 2)
                    cv2.imshow("SeismoVision grid map", preview)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    continue
                magnified = frame
                for magnifier in magnifiers:
                    magnified = magnifier.apply(magnified, timestamp)
                preview = _draw_grid(magnified, mapper, states)
                dashboard_height = max(frame.shape[0], 750)
                left = np.full((dashboard_height, frame.shape[1], 3), 10, dtype=np.uint8)
                left[:frame.shape[0]] = preview
                right = np.full((dashboard_height, 520, 3), 25, dtype=np.uint8)
                graph_width, graph_height = right.shape[1] // 2, right.shape[0] // 5
                for index, history in enumerate(mapper.spectrogram_history.values()):
                    row, column = divmod(index, 2)
                    _spectrogram(right, (column * graph_width, row * graph_height, (column + 1) * graph_width - 2, (row + 1) * graph_height - 2), index, history, mapper.analysis.max_frequency_hz)
                dashboard = cv2.hconcat((left, right))
                cv2.imshow("SeismoVision grid map", cv2.vconcat((dashboard, _draw_status(dashboard.shape[1], mapper, states))))
            key = cv2.waitKey(1) & 0xFF
            if key == ord("r"):
                mapper, lost_frames = None, 0
                magnifiers = ()
            elif key == ord("q"):
                break
    finally:
        source.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
