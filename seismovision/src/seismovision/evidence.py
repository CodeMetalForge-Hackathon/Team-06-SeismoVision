from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


class EvidenceWriter:
    def __init__(self, root: str, config: dict, frame_size: tuple[int, int], fps: float, save_video: bool):
        run_name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.path = Path(root) / run_name
        self.path.mkdir(parents=True, exist_ok=False)
        (self.path / "effective-config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        self._metrics = (self.path / "measurements.jsonl").open("a", encoding="utf-8", buffering=1)
        self._raw: Optional[cv2.VideoWriter] = None
        self._magnified: Optional[cv2.VideoWriter] = None
        if save_video:
            codec = cv2.VideoWriter_fourcc(*"mp4v")
            self._raw = cv2.VideoWriter(str(self.path / "raw.mp4"), codec, fps, frame_size)
            self._magnified = cv2.VideoWriter(str(self.path / "magnified.mp4"), codec, fps, frame_size)

    def write(self, record: dict, raw: np.ndarray, magnified: np.ndarray) -> None:
        self._metrics.write(json.dumps(record, separators=(",", ":")) + "\n")
        if self._raw is not None:
            self._raw.write(raw)
        if self._magnified is not None:
            self._magnified.write(magnified)

    def close(self) -> None:
        self._metrics.close()
        if self._raw is not None:
            self._raw.release()
        if self._magnified is not None:
            self._magnified.release()
