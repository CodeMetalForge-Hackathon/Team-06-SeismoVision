from __future__ import annotations

import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

from .config import CameraConfig


class FrameSource(ABC):
    """Produces BGR frames and monotonic capture timestamps."""

    @abstractmethod
    def read(self) -> tuple[bool, Optional[np.ndarray], float]:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError


class OpenCVSource(FrameSource):
    def __init__(self, source: Union[int, str], camera: CameraConfig):
        self._capture = cv2.VideoCapture(source)
        if not self._capture.isOpened():
            raise RuntimeError(f"Unable to open video source: {source}")
        if isinstance(source, int):
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, camera.width)
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, camera.height)
            self._capture.set(cv2.CAP_PROP_FPS, camera.fps)

    def read(self) -> tuple[bool, Optional[np.ndarray], float]:
        ok, frame = self._capture.read()
        return ok, frame if ok else None, time.monotonic()

    def close(self) -> None:
        self._capture.release()


class PiCamera2Source(FrameSource):
    """Optional Picamera2 source. Import is delayed to preserve Jetson portability."""

    def __init__(self, camera: CameraConfig):
        try:
            from picamera2 import Picamera2  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Picamera2 is unavailable. Install python3-picamera2 or use --video.") from exc
        self._camera = Picamera2()
        config = self._camera.create_video_configuration(
            main={"size": (camera.width, camera.height), "format": "RGB888"},
            controls={"FrameDurationLimits": (int(1_000_000 / camera.fps),) * 2},
        )
        self._camera.configure(config)
        self._camera.start()
        time.sleep(camera.warmup_seconds)

    def read(self) -> tuple[bool, Optional[np.ndarray], float]:
        # Picamera2 RGB888 frames are converted to OpenCV's BGR convention.
        rgb = self._camera.capture_array("main")
        return True, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), time.monotonic()

    def close(self) -> None:
        self._camera.stop()
        self._camera.close()


def make_source(camera: CameraConfig, mode: str, video: Optional[str]) -> FrameSource:
    if video:
        return OpenCVSource(str(Path(video)), camera)
    if mode == "picamera2":
        return PiCamera2Source(camera)
    if mode == "usb":
        return OpenCVSource(0, camera)
    raise ValueError(f"Unknown camera mode: {mode}")
