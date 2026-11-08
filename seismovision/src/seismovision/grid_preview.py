"""Live automatically located cardboard motion map."""

import argparse
from dataclasses import replace
from typing import Dict

import cv2
import numpy as np

from .capture import make_source
from .config import AnalysisConfig, CameraConfig, MagnificationConfig, load_grid_config
from .auto_regions import StructureBoxes, detect_structure, select_background_reference
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


def _draw_grid(frame: np.ndarray, boxes: StructureBoxes, states: Dict[str, GridCellState]) -> np.ndarray:
    output = frame.copy()
    x, y, width, height = boxes.full
    cv2.rectangle(output, (x, y), (x + width, y + height), (0, 255, 255), 2)
    for box in (boxes.upper, boxes.lower):
        x, y, width, height = box
        cv2.rectangle(output, (x, y), (x + width, y + height), (0, 255, 255), 1)
    for index, state in enumerate(states.values()):
        x, y, width, height = state.roi
        anomaly = state.anomaly_score is not None and state.anomaly_score > 3.0
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
    return output


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
    invalid_frames = 0
    try:
        while True:
            ok, frame, timestamp = source.read()
            if not ok or frame is None:
                break
            boxes = detect_structure(frame)
            if boxes is None:
                mapper, magnifiers = None, ()
                invalid_frames = 0
                preview = frame.copy()
                cv2.putText(preview, "MEASUREMENT UNRELIABLE - find full cardboard stack", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 180, 255), 2)
                cv2.imshow("SeismoVision grid map", preview)
            else:
                old = mapper.bbox if mapper is not None else None
                if old is not None and (max(abs(boxes.lower[i] - old[i]) for i in (0, 1)) > 12 or max(abs(boxes.lower[i] - old[i]) for i in (2, 3)) > 20):
                    mapper, magnifiers = None, ()
                if mapper is None:
                    reference_box = select_background_reference(frame, boxes.full)
                    if reference_box is not None:
                        mapper = BuildingGridMapper(boxes.lower, reference_box, analysis, boxes.upper)
                        mapper.tracker.validate(frame)
                        magnifiers = tuple(RoiMotionMagnifier(roi, magnification, camera.fps) for roi in (*mapper.cells, *mapper.upper_cells)) if magnification.enabled else ()
                        invalid_frames = 0
                if mapper is None:
                    preview = frame.copy()
                    cv2.putText(preview, "MEASUREMENT UNRELIABLE - no fixed background reference", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 180, 255), 2)
                    cv2.imshow("SeismoVision grid map", preview)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    continue
                camera_motion, states = mapper.update(frame, timestamp)
                if not any(state.motion.confidence > 0 for state in states.values()):
                    reason = "camera motion" if camera_motion > mapper.analysis.max_camera_motion_px else "reacquiring points"
                    invalid_frames += 1
                    if invalid_frames >= 4:
                        mapper, magnifiers = None, ()
                    preview = frame.copy()
                    cv2.putText(preview, f"MEASUREMENT UNRELIABLE - {reason}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 180, 255), 2)
                    cv2.imshow("SeismoVision grid map", preview)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    continue
                invalid_frames = 0
                magnified = frame
                for magnifier in magnifiers:
                    magnified = magnifier.apply(magnified, timestamp)
                preview = _draw_grid(magnified, boxes, states)
                dashboard_height = max(frame.shape[0], 750)
                left = np.full((dashboard_height, frame.shape[1], 3), 10, dtype=np.uint8)
                left[:frame.shape[0]] = preview
                right = np.full((dashboard_height, 520, 3), 25, dtype=np.uint8)
                graph_width, graph_height = right.shape[1] // 2, right.shape[0] // 5
                for index, history in enumerate(mapper.spectrogram_history.values()):
                    row, column = divmod(index, 2)
                    _spectrogram(right, (column * graph_width, row * graph_height, (column + 1) * graph_width - 2, (row + 1) * graph_height - 2), index, history, mapper.analysis.max_frequency_hz)
                cv2.imshow("SeismoVision grid map", cv2.hconcat((left, right)))
            key = cv2.waitKey(1) & 0xFF
            if key == ord("r"):
                mapper, invalid_frames = None, 0
                magnifiers = ()
            elif key == ord("q"):
                break
    finally:
        source.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
