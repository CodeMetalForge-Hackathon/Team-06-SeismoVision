from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np

from .config import MagnificationConfig, Rect


class RoiMotionMagnifier:
    """Causal, ROI-only EVM-style visualization. It is not used for measurements."""

    def __init__(self, roi: Rect, config: MagnificationConfig, fps: float):
        self.roi = roi
        self.config = config
        self.fps = fps
        self._low: list[np.ndarray] = []
        self._high: list[np.ndarray] = []
        self._last_timestamp: Optional[float] = None

    @staticmethod
    def _pyramid(image: np.ndarray, levels: int) -> list[np.ndarray]:
        gaussian = [image]
        for _ in range(levels - 1):
            gaussian.append(cv2.pyrDown(gaussian[-1]))
        laplacian: list[np.ndarray] = []
        for level in range(len(gaussian) - 1):
            expanded = cv2.pyrUp(gaussian[level + 1], dstsize=(gaussian[level].shape[1], gaussian[level].shape[0]))
            laplacian.append(gaussian[level] - expanded)
        laplacian.append(gaussian[-1])
        return laplacian

    @staticmethod
    def _collapse(levels: list[np.ndarray]) -> np.ndarray:
        result = levels[-1]
        for level in reversed(levels[:-1]):
            result = cv2.pyrUp(result, dstsize=(level.shape[1], level.shape[0])) + level
        return result

    def apply(self, frame: np.ndarray, timestamp: float) -> np.ndarray:
        x, y, w, h = self.roi
        output = frame.copy()
        roi = frame[y : y + h, x : x + w]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        levels = self._pyramid(gray, self.config.pyramid_levels)
        if not self._low:
            self._low = [level.copy() for level in levels]
            self._high = [level.copy() for level in levels]
            self._last_timestamp = timestamp
            return output
        dt = max(1.0 / (self.fps * 2), min(0.25, timestamp - (self._last_timestamp or timestamp)))
        self._last_timestamp = timestamp
        # Difference of two causal exponential low-pass filters = temporal band-pass.
        low_alpha = 1.0 - math.exp(-2 * math.pi * self.config.low_hz * dt)
        high_alpha = 1.0 - math.exp(-2 * math.pi * self.config.high_hz * dt)
        amplified: list[np.ndarray] = []
        for index, level in enumerate(levels):
            self._low[index] += low_alpha * (level - self._low[index])
            self._high[index] += high_alpha * (level - self._high[index])
            band = self._high[index] - self._low[index]
            # Coarser levels receive lower gain to limit reconstruction artifacts.
            gain = self.config.alpha / (1 + index)
            amplified.append(level + gain * band)
        magnified = np.clip(self._collapse(amplified) * 255.0, 0, 255).astype(np.uint8)
        output[y : y + h, x : x + w] = cv2.cvtColor(magnified, cv2.COLOR_GRAY2BGR)
        return output
