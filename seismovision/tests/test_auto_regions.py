import unittest

import cv2
import numpy as np

from seismovision.auto_regions import detect_structure, select_background_reference


class AutoRegionTests(unittest.TestCase):
    def test_detects_two_boxes_with_dark_joint(self):
        frame = np.full((480, 640, 3), 205, np.uint8)
        cv2.rectangle(frame, (235, 60), (405, 145), (45, 130, 185), -1)
        cv2.rectangle(frame, (260, 145), (380, 405), (45, 130, 185), -1)
        cv2.rectangle(frame, (258, 145), (382, 195), (25, 25, 25), -1)
        for x in range(20, 190, 20):
            for y in range(40, 420, 40):
                cv2.rectangle(frame, (x, y), (x + 6, y + 6), (80, 80, 80), -1)
        boxes = detect_structure(frame)
        self.assertIsNotNone(boxes)
        self.assertLess(abs(boxes.upper[0] - 235), 8)
        self.assertLess(abs(boxes.lower[1] - 145), 15)
        self.assertGreater(boxes.lower[3], 200)
        reference = select_background_reference(frame, boxes.full)
        self.assertIsNotNone(reference)
        self.assertLess(reference[0] + reference[2], boxes.full[0])

    def test_blocked_frame_is_not_a_structure(self):
        self.assertIsNone(detect_structure(np.full((480, 640, 3), 20, np.uint8)))


if __name__ == "__main__":
    unittest.main()
