"""User-box-initialized 4-by-2 optical-flow map for the cardboard demo."""

from collections import deque
from dataclasses import dataclass
from typing import Dict, Optional

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


def refine_building_box(frame: np.ndarray, seed: Rect) -> Rect:
    """Refines a user-drawn building box to its strongest enclosing edge contour."""
    x, y, width, height = seed
    crop = frame[y:y + height, x:x + width]
    if crop.size == 0:
        return seed
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    median = float(np.median(gray))
    edges = cv2.Canny(gray, max(0, int(0.66 * median)), min(255, int(1.33 * median)))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return seed
    contour = max(contours, key=cv2.contourArea)
    left, top, refined_width, refined_height = cv2.boundingRect(contour)
    if refined_width * refined_height < width * height * 0.2:
        return seed
    return x + left, y + top, refined_width, refined_height


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
    """Tracks the lower grid and upper response relative to the lower base."""

    def __init__(self, bbox: Rect, reference_roi: Rect, analysis: AnalysisConfig, upper_bbox: Optional[Rect] = None):
        self.bbox = bbox
        self.upper_bbox = upper_bbox
        self.analysis = analysis
        self.cells = grid_cells(bbox)
        self.upper_cells = grid_cells(upper_bbox, rows=1) if upper_bbox else []
        self.targets = tuple(TargetConfig(f"cell_{index}", roi) for index, roi in enumerate(self.cells)) + tuple(
            TargetConfig(f"upper_{index}", roi) for index, roi in enumerate(self.upper_cells)
        )
        self.tracker = MotionTracker(reference_roi, self.targets, analysis.minimum_track_points)
        self.horizontal = {target.name: RollingVibrationAnalyzer(analysis) for target in self.targets}
        self.vertical = {target.name: RollingVibrationAnalyzer(analysis) for target in self.targets}
        self.smoothers = {target.name: LowPassVector() for target in self.targets}
        self.positions = {target.name: np.zeros(2, dtype=float) for target in self.targets}
        self.amplitude_history = {target.name: deque(maxlen=90) for target in self.targets}
        self.frequency_history = {target.name: deque(maxlen=90) for target in self.targets}
        self.spectra = {target.name: (np.empty(0), np.empty(0)) for target in self.targets}
        self.spectral_bins = np.linspace(analysis.min_frequency_hz, analysis.max_frequency_hz, 32)
        self.spectrogram_history = {target.name: deque(maxlen=90) for target in self.targets}
        self.response_anomaly_score: Optional[float] = None

    def reset_measurement(self) -> None:
        self.tracker.reset()
        for analyzer in (*self.horizontal.values(), *self.vertical.values()):
            analyzer.samples.clear()
        for smoother in self.smoothers.values():
            smoother.reset()
        for position in self.positions.values():
            position.fill(0.0)
        for histories in (self.amplitude_history, self.frequency_history):
            for history in histories.values():
                history.clear()
        self.spectra = {target.name: (np.empty(0), np.empty(0)) for target in self.targets}
        for history in self.spectrogram_history.values():
            history.clear()
        self.response_anomaly_score = None

    def update(self, frame: np.ndarray, timestamp: float) -> tuple[float, Dict[str, GridCellState]]:
        reference_motion, motions = self.tracker.update(frame)
        camera_motion = float(np.linalg.norm(reference_motion))
        if camera_motion > self.analysis.max_camera_motion_px:
            for motion in motions.values():
                motion.confidence = 0.0
        base = [motions[f"cell_{index}"].relative_displacement for index in (6, 7) if motions[f"cell_{index}"].confidence > 0]
        for target in self.targets:
            if target.name.startswith("upper_"):
                if base:
                    motions[target.name].relative_displacement -= np.median(base, axis=0)
                else:
                    motions[target.name].confidence = 0.0
        states: Dict[str, GridCellState] = {}
        for target in self.targets:
            motion = motions[target.name]
            position = self.positions[target.name]
            if motion.confidence > 0.0:
                position += motion.relative_displacement
                self.horizontal[target.name].add(timestamp, float(position[0]), motion.confidence)
                self.vertical[target.name].add(timestamp, float(position[1]), motion.confidence)
            else:
                self.horizontal[target.name].samples.clear()
                self.vertical[target.name].samples.clear()
                self.smoothers[target.name].reset()
                position.fill(0.0)
                self.amplitude_history[target.name].clear()
                self.frequency_history[target.name].clear()
                self.spectrogram_history[target.name].clear()
                self.spectra[target.name] = (np.empty(0), np.empty(0))
                states[target.name] = GridCellState(target.name, target.roi, motion, position.copy(), 0.0, None, None)
                continue
            axis, metrics = choose_stronger_axis(self.horizontal[target.name].metrics(), self.vertical[target.name].metrics())
            selected = self.horizontal[target.name] if axis == "horizontal" else self.vertical[target.name]
            self.spectra[target.name] = selected.spectrum()
            spectrum_frequencies, spectrum_powers = self.spectra[target.name]
            history = self.spectrogram_history[target.name]
            if len(spectrum_frequencies) and np.any(spectrum_powers > 0):
                spectrum_column = np.interp(self.spectral_bins, spectrum_frequencies, spectrum_powers, left=0.0, right=0.0)
            elif history:
                spectrum_column = history[-1].copy()
            else:
                spectrum_column = np.zeros_like(self.spectral_bins)
            history.append(spectrum_column)
            smoothed = self.smoothers[target.name].update(timestamp, position)
            self.amplitude_history[target.name].append(smoothed.magnitude_px)
            self.frequency_history[target.name].append(metrics.dominant_hz)
            states[target.name] = GridCellState(target.name, target.roi, motion, position.copy(), smoothed.magnitude_px, metrics.dominant_hz, None)

        reliable = [state for state in states.values() if state.motion.confidence >= self.analysis.minimum_confidence]

        def outlier_scores(values: list[GridCellState], value: str) -> Dict[str, float]:
            numbers = np.asarray([getattr(state, value) for state in values], dtype=float)
            if len(numbers) < 3:
                return {}
            median = float(np.median(numbers))
            scale = max(0.02, 1.4826 * float(np.median(np.abs(numbers - median))))
            return {state.name: abs(float(getattr(state, value)) - median) / scale for state in values}

        frequency_scores = outlier_scores([state for state in reliable if state.frequency_hz is not None], "frequency_hz")
        amplitude_scores = outlier_scores(reliable, "smoothed_magnitude")
        combined_scores = []
        for state in states.values():
            components = []
            if state.name in frequency_scores:
                components.append(frequency_scores[state.name] / self.analysis.grid_frequency_outlier_limit)
            if state.name in amplitude_scores:
                components.append(amplitude_scores[state.name] / self.analysis.grid_amplitude_outlier_limit)
            state.anomaly_score = max(components) if components else None
            if state.anomaly_score is not None:
                combined_scores.append(state.anomaly_score)
        self.response_anomaly_score = max(combined_scores, default=None)
        return camera_motion, states
