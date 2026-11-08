import math
import unittest

import numpy as np

from seismovision.analysis import LowPassVector, RollingVibrationAnalyzer, VibrationMetrics, choose_stronger_axis, evaluate
from seismovision.config import AnalysisConfig, TargetConfig


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.config = AnalysisConfig(
            window_seconds=20.0,
            minimum_samples=120,
            min_frequency_hz=0.2,
            max_frequency_hz=5.0,
            minimum_track_points=6,
            max_camera_motion_px=1.5,
            minimum_confidence=0.65,
        )

    def test_recovers_dominant_frequency_from_timestamped_sine(self):
        analyzer = RollingVibrationAnalyzer(self.config)
        expected_hz = 1.25
        for index in range(600):
            # Deliberately vary timing a little to exercise resampling.
            t = index / 30.0 + (0.001 if index % 2 else 0.0)
            analyzer.add(t, 2.0 * math.sin(2 * math.pi * expected_hz * t), 0.95)
        metrics = analyzer.metrics()
        self.assertIsNotNone(metrics.dominant_hz)
        self.assertAlmostEqual(metrics.dominant_hz, expected_hz, delta=0.08)
        self.assertGreater(metrics.peak_displacement_px, 1.8)

    def test_camera_motion_produces_no_result(self):
        target = TargetConfig("roof", (0, 0, 10, 10), baseline_hz=1.0)
        metrics = VibrationMetrics(300, 1.0, 0.4, 0.9)
        decision = evaluate(target, metrics, 1.6, self.config)
        self.assertEqual(decision.state, "NO_RESULT")
        self.assertIn("camera_motion_exceeds_limit", decision.reasons)

    def test_frequency_shift_requests_inspection(self):
        target = TargetConfig("roof", (0, 0, 10, 10), baseline_hz=1.0, frequency_shift_fraction_limit=0.1)
        metrics = VibrationMetrics(300, 1.3, 0.4, 0.9)
        decision = evaluate(target, metrics, 0.1, self.config)
        self.assertEqual(decision.state, "INSPECTION_RECOMMENDED")

    def test_frequency_uses_the_stronger_motion_axis(self):
        horizontal = VibrationMetrics(300, 1.0, 0.2, 0.9)
        vertical = VibrationMetrics(300, 1.4, 0.8, 0.9)
        axis, metrics = choose_stronger_axis(horizontal, vertical)
        self.assertEqual(axis, "vertical")
        self.assertEqual(metrics.dominant_hz, 1.4)

    def test_low_pass_vector_reports_smoothed_magnitude_rate(self):
        low_pass = LowPassVector()
        low_pass.update(0.0, np.array([0.0, 0.0]))
        result = low_pass.update(0.1, np.array([10.0, 0.0]))
        self.assertGreater(result.magnitude_px, 0.0)
        self.assertLess(result.magnitude_px, 10.0)
        self.assertGreater(result.magnitude_rate_px_s, 0.0)


if __name__ == "__main__":
    unittest.main()
