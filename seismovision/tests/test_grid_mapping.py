import importlib.util
import unittest
from unittest.mock import patch

import numpy as np

HAS_OPENCV = importlib.util.find_spec("cv2") is not None


@unittest.skipUnless(HAS_OPENCV, "OpenCV is required for grid mapper tests")
class GridGeometryTests(unittest.TestCase):
    def test_creates_eight_inner_cells(self):
        from seismovision.grid_mapping import grid_cells

        cells = grid_cells((100, 50, 200, 400))
        self.assertEqual(len(cells), 8)
        self.assertTrue(all(100 <= x < 300 and 50 <= y < 450 for x, y, _, _ in cells))
        self.assertTrue(all(width < 100 and height < 100 for _, _, width, height in cells))

    def test_reacquires_lost_points_without_changing_roi(self):
        from seismovision.tracking import RegionFlow

        tracker = RegionFlow((10, 20, 30, 40), min_points=3)
        tracker._previous = np.zeros((80, 80), dtype=np.uint8)
        tracker._points = np.array([[[12.0 + index, 22.0 + index]] for index in range(10)], dtype=np.float32)
        tracker._expected_points = 10
        reacquired = np.array([[[13.0, 23.0]], [[16.0, 26.0]], [[19.0, 29.0]]], dtype=np.float32)
        with patch("seismovision.tracking.cv2.calcOpticalFlowPyrLK", return_value=(tracker._points + 1, np.array([[1], [1], [1], [1], [0], [0], [0], [0], [0], [0]], dtype=np.uint8), np.zeros((10, 1), dtype=np.float32))), patch.object(tracker, "_detect", return_value=reacquired) as detect:
            result = tracker.update(np.ones((80, 80), dtype=np.uint8))
        detect.assert_called_once()
        self.assertEqual(tracker.roi, (10, 20, 30, 40))
        self.assertEqual(result.point_count, 3)
        self.assertEqual(result.confidence, 0.0)
        np.testing.assert_array_equal(result.displacement, np.zeros(2))
