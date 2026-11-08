import importlib.util
import unittest

HAS_OPENCV = importlib.util.find_spec("cv2") is not None


@unittest.skipUnless(HAS_OPENCV, "OpenCV is required for grid mapper tests")
class GridGeometryTests(unittest.TestCase):
    def test_creates_eight_inner_cells(self):
        from seismovision.grid_mapping import grid_cells

        cells = grid_cells((100, 50, 200, 400))
        self.assertEqual(len(cells), 8)
        self.assertTrue(all(100 <= x < 300 and 50 <= y < 450 for x, y, _, _ in cells))
        self.assertTrue(all(width < 100 and height < 100 for _, _, width, height in cells))
