from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import cv2
import numpy as np

from .config import Rect, TargetConfig


def _crop(gray: np.ndarray, roi: Rect) -> np.ndarray:
    x, y, w, h = roi
    return gray[y : y + h, x : x + w]


def validate_roi(frame: np.ndarray, roi: Rect, name: str) -> None:
    x, y, w, h = roi
    height, width = frame.shape[:2]
    if x < 0 or y < 0 or x + w > width or y + h > height:
        raise ValueError(f"{name} ROI {roi} lies outside {width}x{height} frame")


@dataclass
class FlowResult:
    displacement: np.ndarray
    point_count: int
    confidence: float


class RegionFlow:
    """Sparse local feature tracker with automatic reacquisition in a fixed ROI."""

    def __init__(self, roi: Rect, min_points: int):
        self.roi = roi
        self.min_points = min_points
        self._points: Optional[np.ndarray] = None
        self._previous: Optional[np.ndarray] = None

    def _detect(self, gray: np.ndarray) -> Optional[np.ndarray]:
        x, y, _, _ = self.roi
        points = cv2.goodFeaturesToTrack(
            _crop(gray, self.roi), maxCorners=80, qualityLevel=0.01,
            minDistance=5, blockSize=7,
        )
        if points is not None:
            points[:, 0, 0] += x
            points[:, 0, 1] += y
        return points

    def update(self, gray: np.ndarray) -> FlowResult:
        if self._previous is None or self._points is None or len(self._points) < self.min_points:
            self._previous = gray
            self._points = self._detect(gray)
            count = 0 if self._points is None else len(self._points)
            return FlowResult(np.zeros(2), count, 0.0)

        new_points, status, error = cv2.calcOpticalFlowPyrLK(
            self._previous, gray, self._points, None,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        )
        valid = status.reshape(-1).astype(bool) if status is not None else np.zeros(len(self._points), dtype=bool)
        if error is not None:
            valid &= error.reshape(-1) < 20.0
        old = self._points[valid]
        new = new_points[valid] if new_points is not None else np.empty((0, 1, 2), np.float32)
        self._previous = gray
        self._points = new.reshape(-1, 1, 2) if len(new) else self._detect(gray)
        count = len(new)
        if count == 0:
            return FlowResult(np.zeros(2), 0, 0.0)
        vectors = (new - old).reshape(-1, 2)
        displacement = np.median(vectors, axis=0)
        # A robust spread measure makes independently moving foliage reduce confidence.
        median_abs_deviation = np.median(np.linalg.norm(vectors - displacement, axis=1))
        count_confidence = min(1.0, count / float(self.min_points * 3))
        spread_confidence = float(np.exp(-median_abs_deviation / 1.5))
        return FlowResult(displacement, count, count_confidence * spread_confidence)


@dataclass
class TargetMotion:
    name: str
    relative_displacement: np.ndarray
    confidence: float
    point_count: int


class MotionTracker:
    """Tracks building regions after subtracting the fixed reference's apparent motion."""

    def __init__(self, reference_roi: Rect, targets: Iterable[TargetConfig], min_points: int):
        self.reference = RegionFlow(reference_roi, min_points)
        self.targets: Dict[str, RegionFlow] = {
            target.name: RegionFlow(target.roi, min_points) for target in targets
        }
        self._target_configs = {target.name: target for target in targets}

    def validate(self, frame: np.ndarray) -> None:
        validate_roi(frame, self.reference.roi, "reference")
        for name, tracker in self.targets.items():
            validate_roi(frame, tracker.roi, name)

    def update(self, frame: np.ndarray) -> tuple[np.ndarray, Dict[str, TargetMotion]]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ref = self.reference.update(gray)
        output: Dict[str, TargetMotion] = {}
        for name, tracker in self.targets.items():
            target = tracker.update(gray)
            output[name] = TargetMotion(
                name=name,
                relative_displacement=target.displacement - ref.displacement,
                confidence=min(target.confidence, ref.confidence),
                point_count=min(target.point_count, ref.point_count),
            )
        return ref.displacement, output
