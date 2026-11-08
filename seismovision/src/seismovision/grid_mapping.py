"""YOLO-box-initialized 4-by-2 optical-flow map for the cardboard demo."""

from collections import deque
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from .analysis import LowPassVector, RollingVibrationAnalyzer, choose_stronger_axis
from .config import AnalysisConfig, Rect, TargetConfig
from .tracking import MotionTracker, TargetMotion


def grid_cells(bbox: Rect, rows: int = 4, columns: int = 2, inset_fraction: float = 0.14) -> list[Rect]:
    """Creates smaller motion ROIs inside each cell of a detector bounding box."""
    x, y, width, height = bbox
    cell_width, cell_height = width / columns, height / rows
    cells = []
    for row in range(rows):
        for column in range(columns):
            inset_x, inset_y = int(cell_width * inset_fraction), int(cell_height * inset_fraction)
            left = int(x + column * cell_width + inset_x)
            top = int(y + row * cell_height + inset_y)
            cells.append((left, top, max(1, int(cell_width) - 2 * inset_x), max(1, int(cell_height) - 2 * inset_y)))
    return cells


class YoloBuildingDetector:
    """Optional Ultralytics adapter. The model must have a dedicated cardboard-building class."""

    def __init__(self, model_path: str, class_name: str, confidence: float):
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Install the model runtime with: pip install ultralytics") from exc
        self.model = YOLO(model_path)
        self.class_name = class_name
        self.confidence = confidence

    def detect(self, frame: np.ndarray) -> Optional[Rect]:
        result = self.model(frame, conf=self.confidence, verbose=False)[0]
        names = result.names
        candidates = []
        for box in result.boxes:
            label = names[int(box.cls[0])]
            if label != self.class_name:
                continue
            x1, y1, x2, y2 = (int(value) for value in box.xyxy[0].tolist())
            candidates.append((x1, y1, x2 - x1, y2 - y1))
        return max(candidates, key=lambda rect: rect[2] * rect[3]) if candidates else None


@dataclass
class GridCellState:
    name: str
    roi: Rect
    motion: TargetMotion
    position: np.ndarray
    smoothed_magnitude: float
    frequency_hz: Optional[float]
    anomaly_score: Optional[float]


class BuildingGridMapper:
    """Tracks eight fixed inner regions after a YOLO detector establishes the building box."""

    def __init__(self, bbox: Rect, reference_roi: Rect, analysis: AnalysisConfig):
        self.bbox = bbox
        self.analysis = analysis
        self.cells = grid_cells(bbox)
        self.targets = tuple(TargetConfig(f"cell_{index}", roi) for index, roi in enumerate(self.cells))
        self.tracker = MotionTracker(reference_roi, self.targets, analysis.minimum_track_points)
        self.horizontal = {target.name: RollingVibrationAnalyzer(analysis) for target in self.targets}
        self.vertical = {target.name: RollingVibrationAnalyzer(analysis) for target in self.targets}
        self.smoothers = {target.name: LowPassVector() for target in self.targets}
        self.positions = {target.name: np.zeros(2, dtype=float) for target in self.targets}
        self.frequency_history = {target.name: deque(maxlen=90) for target in self.targets}

    def reset_measurement(self) -> None:
        self.tracker.reset()
        for analyzer in (*self.horizontal.values(), *self.vertical.values()):
            analyzer.samples.clear()
        for smoother in self.smoothers.values():
            smoother.reset()
        for position in self.positions.values():
            position.fill(0.0)
        for history in self.frequency_history.values():
            history.clear()

    def update(self, frame: np.ndarray, timestamp: float) -> tuple[float, Dict[str, GridCellState]]:
        reference_motion, motions = self.tracker.update(frame)
        states: Dict[str, GridCellState] = {}
        frequencies = []
        for target in self.targets:
            motion = motions[target.name]
            position = self.positions[target.name]
            position += motion.relative_displacement
            self.horizontal[target.name].add(timestamp, float(position[0]), motion.confidence)
            self.vertical[target.name].add(timestamp, float(position[1]), motion.confidence)
            _, metrics = choose_stronger_axis(self.horizontal[target.name].metrics(), self.vertical[target.name].metrics())
            smoothed = self.smoothers[target.name].update(timestamp, position)
            self.frequency_history[target.name].append(metrics.dominant_hz)
            if metrics.dominant_hz is not None:
                frequencies.append(metrics.dominant_hz)
            states[target.name] = GridCellState(target.name, target.roi, motion, position.copy(), smoothed.magnitude_px, metrics.dominant_hz, None)

        if len(frequencies) >= 3:
            median = float(np.median(frequencies))
            mad = float(np.median(np.abs(np.asarray(frequencies) - median)))
            scale = max(0.02, 1.4826 * mad)
            for state in states.values():
                if state.frequency_hz is not None:
                    state.anomaly_score = abs(state.frequency_hz - median) / scale
        return float(np.linalg.norm(reference_motion)), states
