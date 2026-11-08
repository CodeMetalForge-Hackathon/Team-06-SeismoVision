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
    """Tracks eight fixed inner regions after edge refinement establishes the building box."""

    def __init__(self, bbox: Rect, reference_roi: Rect, analysis: AnalysisConfig, cells: Optional[list[Rect]] = None):
        self.bbox = bbox
        self.analysis = analysis
        self.cells = list(cells) if cells is not None else grid_cells(bbox)
        if len(self.cells) != 8:
            raise ValueError("the grid requires exactly eight tracking boxes")
        self.targets = tuple(TargetConfig(f"cell_{index}", roi) for index, roi in enumerate(self.cells))
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

    def update(self, frame: np.ndarray, timestamp: float) -> tuple[float, Dict[str, GridCellState]]:
        reference_motion, motions = self.tracker.update(frame)
        states: Dict[str, GridCellState] = {}
        frequencies = []
        for target in self.targets:
            motion = motions[target.name]
            position = self.positions[target.name]
            if motion.confidence > 0.0:
                position += motion.relative_displacement
                self.horizontal[target.name].add(timestamp, float(position[0]), motion.confidence)
                self.vertical[target.name].add(timestamp, float(position[1]), motion.confidence)
            axis, metrics = choose_stronger_axis(self.horizontal[target.name].metrics(), self.vertical[target.name].metrics())
            selected = self.horizontal[target.name] if axis == "horizontal" else self.vertical[target.name]
            self.spectra[target.name] = selected.spectrum()
            spectrum_frequencies, spectrum_powers = self.spectra[target.name]
            if len(spectrum_frequencies):
                spectrum_column = np.interp(self.spectral_bins, spectrum_frequencies, spectrum_powers, left=0.0, right=0.0)
            else:
                spectrum_column = np.zeros_like(self.spectral_bins)
            self.spectrogram_history[target.name].append(spectrum_column)
            smoothed = self.smoothers[target.name].update(timestamp, position)
            self.amplitude_history[target.name].append(smoothed.magnitude_px)
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
