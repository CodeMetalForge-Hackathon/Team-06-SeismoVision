"""Generate explicitly synthetic input, measurements and a preview; no real footage."""
import argparse
import json
import math
import time
from pathlib import Path

import cv2

from motion import MotionMap, annotate
from tests.test_motion import BUILDING, REFERENCE, scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    folder = Path(args.output)
    folder.mkdir(parents=True, exist_ok=False)
    raw = cv2.VideoWriter(str(folder/'synthetic-input.avi'), cv2.VideoWriter_fourcc(*'MJPG'), 30, (440,300))
    overlay = cv2.VideoWriter(str(folder/'synthetic-motion-map.avi'), cv2.VideoWriter_fourcc(*'MJPG'), 30, (440,300))
    if not raw.isOpened() or not overlay.isOpened():
        raise RuntimeError('Cannot open demo video writers')
    tracker = MotionMap(BUILDING, REFERENCE)
    timings = []
    try:
        with (folder/'measurements.jsonl').open('w') as log:
            for i in range(360):
                phase = i // 90
                wave = math.sin(2*math.pi*1.25*i/30)
                dx = 0 if phase < 2 else 3*wave
                camera = (0,0) if phase == 0 else (4*math.sin(i/12), 2*math.cos(i/12))
                angle = 0 if phase == 0 else math.sin(i/15)
                frame = scene(dx=dx, camera=camera, angle=angle, roof_only=phase==3)
                started = time.perf_counter()
                report = tracker.update(frame, i/30)
                timings.append(time.perf_counter()-started)
                report['synthetic'] = True
                report['scenario'] = ['idle','camera_only','target_and_camera','roof_and_camera'][phase]
                report['ground_truth_target_dx_px'] = dx
                log.write(json.dumps(report, allow_nan=False)+'\n')
                preview = annotate(frame, report)
                cv2.putText(preview, 'SYNTHETIC: '+report['scenario'], (10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
                raw.write(frame)
                overlay.write(preview)
                if i == 294:
                    if not cv2.imwrite(str(folder/'preview.png'), preview):
                        raise RuntimeError('Cannot save preview')
        summary = dict(frames=len(timings), mean_processing_ms=1000*sum(timings)/len(timings),
                       note='Local host only; not a Raspberry Pi benchmark. Synthetic textured planar scene.')
        (folder/'summary.json').write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary))
    finally:
        raw.release()
        overlay.release()


if __name__ == '__main__':
    main()
