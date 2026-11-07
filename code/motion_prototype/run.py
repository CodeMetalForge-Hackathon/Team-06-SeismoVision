"""Headless replay/USB runner. Pi capture can call MotionMap.update directly."""
import argparse
import json
import math
import time
from pathlib import Path

import cv2

from motion import MotionMap, annotate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video")
    source.add_argument("--camera", type=int)
    parser.add_argument("--building", required=True, type=int, nargs=4, metavar=("X", "Y", "W", "H"))
    parser.add_argument("--reference", required=True, type=int, nargs=4, metavar=("X", "Y", "W", "H"))
    parser.add_argument("--threshold-px", type=float, default=0.75)
    parser.add_argument("--output", required=True, help="New output directory (never overwrites)")
    args = parser.parse_args()
    cap = cv2.VideoCapture(args.video if args.video is not None else args.camera)
    writer = None
    try:
        if not cap.isOpened():
            raise RuntimeError("Cannot open capture source")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise RuntimeError("Source FPS is missing; remux recording with a known frame rate")
        folder = Path(args.output)
        folder.mkdir(parents=True, exist_ok=False)
        tracker = MotionMap(args.building, args.reference, threshold_px=args.threshold_px)
        count = 0
        with (folder / "measurements.jsonl").open("w") as log:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                # Constant-frame-rate recordings use media time, not processing speed.
                timestamp = count / fps if args.video is not None else time.monotonic()
                report = tracker.update(frame, timestamp)
                log.write(json.dumps(report, allow_nan=False) + "\n")
                if writer is None:
                    writer = cv2.VideoWriter(str(folder / "motion-map.avi"),
                                             cv2.VideoWriter_fourcc(*"MJPG"), fps,
                                             (frame.shape[1], frame.shape[0]))
                    if not writer.isOpened():
                        raise RuntimeError("Cannot create motion-map video")
                    (folder / "config.json").write_text(json.dumps(vars(args), indent=2))
                writer.write(annotate(frame, report))
                count += 1
        if not count:
            raise RuntimeError("Source produced no readable frames")
        print(f"Processed {count} frames; evidence: {folder}")
    finally:
        cap.release()
        if writer is not None:
            writer.release()


if __name__ == "__main__":
    main()
