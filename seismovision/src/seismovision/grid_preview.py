"""Live user-bounded building grid preview."""

import argparse
from typing import Dict

import cv2
import numpy as np

from .capture import make_source
from .config import AnalysisConfig, CameraConfig, Rect
from .grid_mapping import BuildingGridMapper, GridCellState, refine_building_box


def _select_regions(frame: np.ndarray) -> tuple[Rect, Rect]:
    seed = tuple(int(value) for value in cv2.selectROI("Draw rough building box, then Enter", frame, showCrosshair=True, fromCenter=False))
    if seed[2] <= 0 or seed[3] <= 0:
        raise RuntimeError("A building box is required")
    reference = tuple(int(value) for value in cv2.selectROI("Draw stable reference box, then Enter", frame, showCrosshair=True, fromCenter=False))
    if reference[2] <= 0 or reference[3] <= 0:
        raise RuntimeError("A stable reference box is required")
    return refine_building_box(frame, seed), reference  # type: ignore[return-value]


COLORS = ((0, 255, 255), (255, 255, 0), (255, 0, 255), (0, 165, 255), (255, 255, 255), (0, 255, 0), (255, 0, 0), (180, 180, 255))


def _history(panel: np.ndarray, top: int, bottom: int, title: str, histories: dict, maximum: float) -> None:
    cv2.putText(panel, title, (8, top + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 235, 235), 1)
    left, right, graph_top = 8, panel.shape[1] - 8, top + 22
    cv2.rectangle(panel, (left, graph_top), (right, bottom), (100, 100, 100), 1)
    for color, values in zip(COLORS, histories.values()):
        points = []
        for index, value in enumerate(values):
            if value is None:
                continue
            x = left + int((right - left) * index / max(1, len(values) - 1))
            y = bottom - int((bottom - graph_top) * min(value, maximum) / maximum)
            points.append((x, y))
        if len(points) > 1:
            cv2.polylines(panel, [np.asarray(points, dtype=np.int32)], False, color, 1)
    cv2.putText(panel, f"0-{maximum:.2f}", (left, bottom - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (210, 210, 210), 1)


def _spectrum(panel: np.ndarray, top: int, bottom: int, spectra: dict, max_frequency: float) -> None:
    cv2.putText(panel, "FFT amplitude vs frequency", (8, top + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 235, 235), 1)
    left, right, graph_top = 8, panel.shape[1] - 8, top + 22
    cv2.rectangle(panel, (left, graph_top), (right, bottom), (100, 100, 100), 1)
    limit = max((float(np.max(np.log1p(power))) for _, power in spectra.values() if len(power)), default=0.0)
    if limit > 0:
        for color, (frequencies, powers) in zip(COLORS, spectra.values()):
            if len(frequencies) < 2:
                continue
            points = [(left + int((right - left) * frequency / max_frequency), bottom - int((bottom - graph_top) * np.log1p(power) / limit)) for frequency, power in zip(frequencies, powers)]
            cv2.polylines(panel, [np.asarray(points, dtype=np.int32)], False, color, 1)
    cv2.putText(panel, f"0-{max_frequency:.1f}Hz", (left, bottom - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (210, 210, 210), 1)


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
        center = (x + width // 2, y + height // 2)
        vector = np.clip(
            state.position * 20.0,
            (-width // 2 + 5, -height // 2 + 18),
            (width // 2 - 5, height // 2 - 5),
        ).astype(int)
        cv2.arrowedLine(output, center, (center[0] + vector[0], center[1] + vector[1]), (0, 255, 255), 2, tipLength=0.25)
        hz = "--" if state.frequency_hz is None else f"{state.frequency_hz:.2f}Hz"
        label = f"{index} {hz}" + (" ANOMALY" if anomaly else "")
        cv2.putText(output, label, (x + 3, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)
        cv2.putText(output, f"{state.position[0]:+.1f}, {state.position[1]:+.1f}px", (x + 3, y + height - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 255, 255), 1)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="User-bounded 4x2 cardboard-building motion map")
    parser.add_argument("--camera", choices=("picamera2", "usb"), default="picamera2")
    parser.add_argument("--video")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    camera = CameraConfig(width=args.width, height=args.height, fps=args.fps)
    source = make_source(camera, args.camera, args.video)
    mapper = None
    moved, settled = False, 0
    try:
        while True:
            ok, frame, timestamp = source.read()
            if not ok or frame is None:
                break
            if mapper is None:
                preview = frame.copy()
                cv2.putText(preview, "Draw building and stable-reference boxes in the selection windows", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 2)
                cv2.imshow("SeismoVision grid map", preview)
                building_box, reference_box = _select_regions(frame)
                mapper = BuildingGridMapper(building_box, reference_box, AnalysisConfig())
                mapper.tracker.validate(frame)
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
                amplitudes = [value for history in mapper.amplitude_history.values() for value in history]
                _history(right, 0, 210, "motion amplitude vs time", mapper.amplitude_history, max(1.0, max(amplitudes, default=0.0)))
                _history(right, 240, 450, "dominant frequency vs time", mapper.frequency_history, mapper.analysis.max_frequency_hz)
                _spectrum(right, 480, 700, mapper.spectra, mapper.analysis.max_frequency_hz)
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
