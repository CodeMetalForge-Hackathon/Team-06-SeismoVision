"""Standalone live cardboard-outline preview for the Pi demo."""

import argparse

import cv2

from .cardboard import CardboardDetector, draw_cardboard_outline
from .capture import make_source
from .config import CameraConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Live cardboard-building outline preview")
    parser.add_argument("--camera", choices=("picamera2", "usb"), default="picamera2")
    parser.add_argument("--video", help="Use a recorded video instead of a camera")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--mask", action="store_true", help="Show the selected cardboard mask beside the camera image")
    args = parser.parse_args()

    camera = CameraConfig(width=args.width, height=args.height, fps=args.fps)
    source = make_source(camera, args.camera, args.video)
    detector = CardboardDetector()
    try:
        while True:
            ok, frame, _ = source.read()
            if not ok or frame is None:
                break
            result = detector.detect(frame)
            preview = draw_cardboard_outline(frame, result)
            if args.mask:
                mask = cv2.cvtColor(result.mask, cv2.COLOR_GRAY2BGR)
                preview = cv2.hconcat((preview, mask))
            cv2.imshow("SeismoVision cardboard preview", preview)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        source.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
