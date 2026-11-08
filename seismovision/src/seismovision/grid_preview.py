"""Live YOLO-bounded building grid preview."""

import argparse
from typing import Dict

import cv2
import numpy as np

from .capture import make_source
from .config import AnalysisConfig, CameraConfig, Rect
from .grid_mapping import BuildingGridMapper, GridCellState, YoloBuildingDetector


def _rect(value: str) -> Rect:
    parts = tuple(int(part.strip()) for part in value.split(","))
    if len(parts) != 4 or parts[2] <= 0 or parts[3] <= 0:
        raise argparse.ArgumentTypeError("ROI must be x,y,width,height")
    return parts  # type: ignore[return-value]


def _heatmap(panel: np.ndarray, top: int, title: str, states: Dict[str, GridCellState], value: str, maximum: float) -> None:
    cv2.putText(panel, title, (8, top + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 235, 235), 1)
    cell_width, cell_height = (panel.shape[1] - 16) // 2, 42
    for index, state in enumerate(states.values()):
        row, column = divmod(index, 2)
        left, y = 8 + column * cell_width, top + 22 + row * cell_height
        raw = getattr(state, value)
        intensity = 0 if raw is None else int(255 * min(float(raw) / maximum, 1.0))
        color = tuple(int(channel) for channel in cv2.applyColorMap(np.uint8([[intensity]]), cv2.COLORMAP_TURBO)[0, 0])
        cv2.rectangle(panel, (left, y), (left + cell_width - 4, y + cell_height - 4), color, cv2.FILLED)
        label = "--" if raw is None else f"{float(raw):.2f}"
        cv2.putText(panel, f"{index}: {label}", (left + 4, y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (10, 10, 10), 1)


def _history(panel: np.ndarray, top: int, histories: dict, max_frequency: float) -> None:
    cv2.putText(panel, "cell frequency history", (8, top + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 235, 235), 1)
    left, right, bottom = 8, panel.shape[1] - 8, panel.shape[0] - 10
    cv2.rectangle(panel, (left, top + 22), (right, bottom), (100, 100, 100), 1)
    colors = ((0, 255, 255), (255, 255, 0), (255, 0, 255), (0, 165, 255), (255, 255, 255), (0, 255, 0), (255, 0, 0), (180, 180, 255))
    for color, values in zip(colors, histories.values()):
        points = []
        for index, value in enumerate(values):
            if value is None:
                continue
            x = left + int((right - left) * index / max(1, len(values) - 1))
            y = bottom - int((bottom - (top + 22)) * min(value, max_frequency) / max_frequency)
            points.append((x, y))
        if len(points) > 1:
            cv2.polylines(panel, [np.asarray(points, dtype=np.int32)], False, color, 1)


def _draw_grid(frame: np.ndarray, mapper: BuildingGridMapper, states: Dict[str, GridCellState]) -> np.ndarray:
    output = frame.copy()
    x, y, width, height = mapper.bbox
    cv2.rectangle(output, (x, y), (x + width, y + height), (0, 255, 255), 2)
    for index, state in enumerate(states.values()):
        x, y, width, height = state.roi
        anomaly = state.anomaly_score is not None and state.anomaly_score > 3.0
        color = (0, 0, 255) if anomaly else (0, 255, 0)
        cv2.rectangle(output, (x, y), (x + width, y + height), color, 2)
        for point in state.motion.points:
            cv2.circle(output, tuple(point.astype(int)), 2, color, cv2.FILLED)
        hz = "--" if state.frequency_hz is None else f"{state.frequency_hz:.2f}Hz"
        label = f"{index} {hz}" + (" ANOMALY" if anomaly else "")
        cv2.putText(output, label, (x + 3, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO-bounded 4x2 cardboard-building motion map")
    parser.add_argument("--model", required=True, help="Custom YOLO model with a cardboard-building class")
    parser.add_argument("--class-name", default="cardboard_building")
    parser.add_argument("--reference-roi", required=True, type=_rect, help="Stationary x,y,width,height reference")
    parser.add_argument("--camera", choices=("picamera2", "usb"), default="picamera2")
    parser.add_argument("--video")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--confidence", type=float, default=0.7)
    args = parser.parse_args()

    camera = CameraConfig(width=args.width, height=args.height, fps=args.fps)
    source = make_source(camera, args.camera, args.video)
    detector = YoloBuildingDetector(args.model, args.class_name, args.confidence)
    mapper = None
    moved, settled = False, 0
    try:
        while True:
            ok, frame, timestamp = source.read()
            if not ok or frame is None:
                break
            if mapper is None:
                bbox = detector.detect(frame)
                preview = frame.copy()
                cv2.putText(preview, "Searching for cardboard_building YOLO box...", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                if bbox is not None:
                    mapper = BuildingGridMapper(bbox, args.reference_roi, AnalysisConfig())
                    mapper.tracker.validate(frame)
                cv2.imshow("SeismoVision grid map", preview)
            else:
                camera_motion, states = mapper.update(frame, timestamp)
                if camera_motion > mapper.analysis.max_camera_motion_px:
                    moved, settled = True, 0
                elif moved:
                    settled += 1
                    if settled >= mapper.analysis.camera_settle_frames:
                        mapper.reset_measurement()
                        moved, settled = False, 0
                        continue
                preview = _draw_grid(frame, mapper, states)
                dashboard_height = max(frame.shape[0], 720)
                left = np.full((dashboard_height, frame.shape[1], 3), 10, dtype=np.uint8)
                left[:frame.shape[0]] = preview
                right = np.full((dashboard_height, 360, 3), 25, dtype=np.uint8)
                frequencies = [state.frequency_hz for state in states.values() if state.frequency_hz is not None]
                amplitudes = [state.smoothed_magnitude for state in states.values()]
                scores = [state.anomaly_score for state in states.values() if state.anomaly_score is not None]
                _heatmap(right, 0, "frequency Hz", states, "frequency_hz", max(1.0, max(frequencies, default=0.0)))
                _heatmap(right, 195, "LP magnitude px", states, "smoothed_magnitude", max(1.0, max(amplitudes, default=0.0)))
                _heatmap(right, 390, "frequency outlier score", states, "anomaly_score", max(3.0, max(scores, default=0.0)))
                _history(right, 580, mapper.frequency_history, mapper.analysis.max_frequency_hz)
                cv2.imshow("SeismoVision grid map", cv2.hconcat((left, right)))
            key = cv2.waitKey(1) & 0xFF
            if key == ord("r"):
                mapper, moved, settled = None, False, 0
            elif key == ord("q"):
                break
    finally:
        source.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
