import importlib.util
import unittest
from unittest.mock import patch

import numpy as np

HAS_OPENCV = importlib.util.find_spec("cv2") is not None


@unittest.skipUnless(HAS_OPENCV, "OpenCV is required for grid mapper tests")
class GridGeometryTests(unittest.TestCase):
    def test_upper_cells_keep_existing_lower_grid(self):
        from seismovision.config import AnalysisConfig
        from seismovision.grid_mapping import BuildingGridMapper

        mapper = BuildingGridMapper((260, 145, 120, 260), (10, 40, 180, 360), AnalysisConfig(), (235, 60, 170, 85))
        self.assertEqual(len(mapper.cells), 8)
        self.assertEqual(len(mapper.upper_cells), 2)
        self.assertEqual(len(mapper.targets), 10)
        self.assertEqual(mapper.targets[8].name, "upper_0")

    def test_upper_motion_is_relative_to_base_and_lost_cell_clears_history(self):
        from seismovision.config import AnalysisConfig
        from seismovision.grid_mapping import BuildingGridMapper
        from seismovision.tracking import TargetMotion

        mapper = BuildingGridMapper((260, 145, 120, 260), (10, 40, 180, 360), AnalysisConfig(), (235, 60, 170, 85))
        motions = {
            target.name: TargetMotion(target.name, np.array([3.0 if target.name.startswith("upper_") else 1.0, 0.0]), 1.0, 20, np.empty((0, 2)))
            for target in mapper.targets
        }
        frame = np.zeros((480, 640, 3), np.uint8)
        with patch.object(mapper.tracker, "update", return_value=(np.zeros(2), motions)):
            mapper.update(frame, 1.0)
        self.assertAlmostEqual(mapper.positions["upper_0"][0], 2.0)
        self.assertEqual(len(mapper.horizontal["cell_0"].samples), 1)
        motions["cell_0"].confidence = 0.0
        with patch.object(mapper.tracker, "update", return_value=(np.zeros(2), motions)):
            _, states = mapper.update(frame, 2.0)
        self.assertEqual(len(mapper.horizontal["cell_0"].samples), 0)
        self.assertIsNone(states["cell_0"].frequency_hz)

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
