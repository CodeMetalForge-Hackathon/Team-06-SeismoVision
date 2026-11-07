import json
import math
import unittest

import cv2
import numpy as np

from motion import MotionMap


BUILDING = (210, 60, 180, 180)
REFERENCE = (20, 30, 150, 240)


def scene(dx=0, dy=0, camera=(0, 0), angle=0, roof_only=False):
    """Deterministic textured target and independent reference; known pixel motion."""
    rng = np.random.default_rng(42)
    background = np.full((300, 440), 35, np.uint8)
    texture = rng.integers(30, 225, (240, 150), dtype=np.uint8)
    background[30:270, 20:170] = cv2.GaussianBlur(texture, (3,3), 0)
    target = np.zeros_like(background)
    texture = rng.integers(30, 225, (180, 180), dtype=np.uint8)
    target[60:240, 210:390] = cv2.GaussianBlur(texture, (3,3), 0)
    if roof_only:
        top = target.copy()
        top[120:] = 0
        target[:120] = 0
        target = cv2.add(target, cv2.warpAffine(top, np.float32([[1,0,dx],[0,1,dy]]), (440,300)))
    else:
        target = cv2.warpAffine(target, np.float32([[1,0,dx],[0,1,dy]]), (440,300))
    background[target > 0] = target[target > 0]
    matrix = cv2.getRotationMatrix2D((220,150), angle, 1)
    matrix[:, 2] += camera
    image = cv2.warpAffine(background, matrix, (440,300))
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


class MotionTests(unittest.TestCase):
    def setUp(self):
        self.tracker = MotionMap(BUILDING, REFERENCE)
        self.tracker.update(scene(), 0)

    def test_idle(self):
        report = self.tracker.update(scene(), 1/30)
        self.assertEqual(report['state'], 'NO_MOTION_ABOVE_THRESHOLD')
        self.assertLess(max(c['magnitude_px'] for c in report['cells']), 0.05)

    def test_known_target_translation(self):
        report = self.tracker.update(scene(dx=3, dy=2), 1/30)
        self.assertEqual(report['state'], 'MOTION_ABOVE_THRESHOLD')
        for cell in report['cells']:
            np.testing.assert_allclose(cell['displacement_px'], [3,2], atol=0.15)

    def test_camera_translation_and_rotation(self):
        report = self.tracker.update(scene(camera=(4,-3), angle=1.5), 1/30)
        self.assertEqual(report['state'], 'NO_MOTION_ABOVE_THRESHOLD')
        self.assertLess(max(c['magnitude_px'] for c in report['cells']), 0.25)

    def test_simultaneous_camera_and_target_motion(self):
        report = self.tracker.update(scene(dx=3, camera=(4,-3), angle=1.5), 1/30)
        self.assertEqual(report['state'], 'MOTION_ABOVE_THRESHOLD')
        for cell in report['cells']:
            self.assertAlmostEqual(cell['magnitude_px'], 3, delta=0.25)

    def test_coupled_camera_target_with_independent_reference(self):
        report = self.tracker.update(scene(dx=3, camera=(-3,0)), 1/30)
        self.assertEqual(report['state'], 'MOTION_ABOVE_THRESHOLD')
        for cell in report['cells']:
            self.assertAlmostEqual(cell['displacement_px'][0], 3, delta=0.15)

    def test_localizes_moving_roof(self):
        report = self.tracker.update(scene(dx=3, roof_only=True), 1/30)
        self.assertTrue(all(c['state'] == 'MOTION_ABOVE_THRESHOLD' for c in report['cells'][:3]))
        self.assertTrue(all(c['state'] == 'NO_MOTION_ABOVE_THRESHOLD' for c in report['cells'][3:]))

    def test_reference_loss_has_no_numbers_and_can_recover(self):
        frame = scene()
        frame[30:270,20:170] = 35
        report = self.tracker.update(frame, 1/30)
        self.assertEqual(report['state'], 'UNRELIABLE')
        self.assertTrue(all(c['magnitude_px'] is None for c in report['cells']))
        self.assertEqual(self.tracker.update(scene(), 2/30)['state'], 'NO_MOTION_ABOVE_THRESHOLD')

    def test_no_texture_never_reports_quiet(self):
        tracker = MotionMap(BUILDING, REFERENCE)
        frame = np.full((300,440,3), 127, np.uint8)
        tracker.update(frame, 0)
        self.assertEqual(tracker.update(frame, 1/30)['state'], 'UNRELIABLE')

    def test_target_occlusion(self):
        frame = scene()
        frame[60:240,210:390] = 35
        report = self.tracker.update(frame, 1/30)
        self.assertEqual(report['state'], 'UNRELIABLE')

    def test_recovers_frequency_and_returns_to_anchor(self):
        for i in range(1,181):
            report = self.tracker.update(scene(dx=2 * math.sin(2*math.pi*1.25*i/30)), i/30)
        for cell in report['cells']:
            self.assertIsNotNone(cell['dominant_hz'])
            self.assertAlmostEqual(cell['dominant_hz'], 1.25, delta=0.12)
            self.assertLess(cell['magnitude_px'], 0.1)
        json.dumps(report, allow_nan=False)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            self.tracker.update(scene(), 0)
        with self.assertRaises(ValueError):
            self.tracker.update(scene(), float('nan'))
        with self.assertRaises(ValueError):
            MotionMap(BUILDING, BUILDING).update(scene(), 0)
        with self.assertRaises(ValueError):
            MotionMap((-1,0,100,100), REFERENCE).update(scene(), 0)


if __name__ == '__main__':
    unittest.main()
