import importlib.util
import unittest


HAS_OPENCV = importlib.util.find_spec("cv2") is not None


@unittest.skipUnless(HAS_OPENCV, "OpenCV is required for cardboard detector tests")
class CardboardDetectorTests(unittest.TestCase):
    def test_detects_central_cardboard_shape_and_returns_outline(self):
        import cv2
        import numpy as np

        from seismovision.cardboard import CardboardDetector

        image = np.full((480, 640, 3), 245, dtype=np.uint8)
        cardboard = cv2.cvtColor(np.uint8([[[18, 180, 175]]]), cv2.COLOR_HSV2BGR)[0, 0].tolist()
        cv2.rectangle(image, (220, 80), (430, 420), cardboard, thickness=cv2.FILLED)
        result = CardboardDetector().detect(image)

        self.assertIsNotNone(result.outline)
        self.assertGreater(result.area_fraction, 0.15)
        self.assertGreaterEqual(len(result.outline), 4)
