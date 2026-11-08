from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

import numpy as np

from .config import AnalysisConfig, TargetConfig


@dataclass(frozen=True)
class VibrationMetrics:
    sample_count: int
    dominant_hz: Optional[float]
    peak_displacement_px: Optional[float]
    confidence: float


@dataclass(frozen=True)
class SmoothedVector:
    vector: np.ndarray
    magnitude_px: float
    magnitude_rate_px_s: float


class LowPassVector:
    """Timestamp-aware low-pass filter for the measured two-axis displacement."""

    def __init__(self, cutoff_hz: float = 0.5):
        self.cutoff_hz = cutoff_hz
        self._timestamp: Optional[float] = None
        self._vector = np.zeros(2, dtype=float)
        self._magnitude = 0.0

    def reset(self) -> None:
        self._timestamp = None
        self._vector.fill(0.0)
        self._magnitude = 0.0

    def update(self, timestamp: float, vector: np.ndarray) -> SmoothedVector:
        if self._timestamp is None:
            self._timestamp = timestamp
            self._vector = vector.astype(float).copy()
            self._magnitude = float(np.linalg.norm(self._vector))
            return SmoothedVector(self._vector.copy(), self._magnitude, 0.0)
        dt = max(1e-3, timestamp - self._timestamp)
        alpha = 1.0 - np.exp(-2.0 * np.pi * self.cutoff_hz * dt)
        self._vector += alpha * (vector - self._vector)
        magnitude = float(np.linalg.norm(self._vector))
        rate = (magnitude - self._magnitude) / dt
        self._timestamp, self._magnitude = timestamp, magnitude
        return SmoothedVector(self._vector.copy(), magnitude, rate)


class RollingVibrationAnalyzer:
    """Timestamp-aware frequency estimator for one unamplified displacement axis."""

    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.samples: Deque[tuple[float, float, float]] = deque()

    def add(self, timestamp: float, displacement_px: float, confidence: float) -> None:
        self.samples.append((timestamp, displacement_px, confidence))
        while self.samples and timestamp - self.samples[0][0] > self.config.window_seconds:
            self.samples.popleft()

    def metrics(self) -> VibrationMetrics:
        if len(self.samples) < self.config.minimum_samples:
            return VibrationMetrics(len(self.samples), None, None, 0.0)
        data = np.asarray(self.samples, dtype=float)
        timestamps, values, confidences = data[:, 0], data[:, 1], data[:, 2]
        dt = float(np.median(np.diff(timestamps)))
        if not np.isfinite(dt) or dt <= 0:
            return VibrationMetrics(len(data), None, None, 0.0)
        fps = 1.0 / dt
        # Resample because capture cadence is not perfectly uniform.
        uniform_time = np.arange(timestamps[0], timestamps[-1], dt)
        if len(uniform_time) < self.config.minimum_samples:
            return VibrationMetrics(len(data), None, None, float(np.mean(confidences)))
        uniform_values = np.interp(uniform_time, timestamps, values)
        centered = uniform_values - np.mean(uniform_values)
        spectrum = np.abs(np.fft.rfft(centered * np.hanning(len(centered)))) ** 2
        frequencies = np.fft.rfftfreq(len(centered), d=dt)
        valid = (frequencies >= self.config.min_frequency_hz) & (frequencies <= self.config.max_frequency_hz)
        if not np.any(valid) or np.max(spectrum[valid]) <= 0:
            dominant = None
        else:
            dominant = float(frequencies[valid][np.argmax(spectrum[valid])])
        peak = float(np.max(np.abs(uniform_values - np.median(uniform_values))))
        return VibrationMetrics(len(data), dominant, peak, float(np.mean(confidences)))

    def spectrum(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns the current one-axis power spectrum inside the configured frequency band."""
        if len(self.samples) < self.config.minimum_samples:
            return np.empty(0), np.empty(0)
        data = np.asarray(self.samples, dtype=float)
        timestamps, values = data[:, 0], data[:, 1]
        dt = float(np.median(np.diff(timestamps)))
        if not np.isfinite(dt) or dt <= 0:
            return np.empty(0), np.empty(0)
        uniform_time = np.arange(timestamps[0], timestamps[-1], dt)
        if len(uniform_time) < self.config.minimum_samples:
            return np.empty(0), np.empty(0)
        uniform_values = np.interp(uniform_time, timestamps, values)
        frequencies = np.fft.rfftfreq(len(uniform_values), d=dt)
        powers = np.abs(np.fft.rfft((uniform_values - np.mean(uniform_values)) * np.hanning(len(uniform_values)))) ** 2
        valid = (frequencies >= self.config.min_frequency_hz) & (frequencies <= self.config.max_frequency_hz)
        return frequencies[valid], powers[valid]


@dataclass(frozen=True)
class Decision:
    state: str
    reasons: tuple[str, ...]


def choose_stronger_axis(horizontal: VibrationMetrics, vertical: VibrationMetrics) -> tuple[str, VibrationMetrics]:
    """Uses the axis with the larger observed peak for a single frequency report."""
    horizontal_peak = horizontal.peak_displacement_px or 0.0
    vertical_peak = vertical.peak_displacement_px or 0.0
    return ("vertical", vertical) if vertical_peak > horizontal_peak else ("horizontal", horizontal)


def evaluate(target: TargetConfig, metrics: VibrationMetrics, camera_motion_px: float, config: AnalysisConfig) -> Decision:
    """Conservative policy: unreliable data never becomes a normal result."""
    if camera_motion_px > config.max_camera_motion_px:
        return Decision("NO_RESULT", ("camera_motion_exceeds_limit",))
    if metrics.sample_count < config.minimum_samples:
        return Decision("NO_RESULT", ("insufficient_samples",))
    if metrics.confidence < config.minimum_confidence:
        return Decision("NO_RESULT", ("tracking_confidence_too_low",))
    if metrics.dominant_hz is None:
        return Decision("NO_RESULT", ("no_dominant_frequency",))
    reasons: list[str] = []
    if target.baseline_hz is not None:
        shift = abs(metrics.dominant_hz - target.baseline_hz) / target.baseline_hz
        if shift > target.frequency_shift_fraction_limit:
            reasons.append("frequency_shift_exceeds_limit")
    if target.relative_displacement_px_limit is not None and metrics.peak_displacement_px is not None:
        if metrics.peak_displacement_px > target.relative_displacement_px_limit:
            reasons.append("relative_displacement_exceeds_limit")
    if reasons:
        return Decision("INSPECTION_RECOMMENDED", tuple(reasons))
    return Decision("NORMAL_RELATIVE_TO_BASELINE", ("no_configured_anomaly_observed",))
