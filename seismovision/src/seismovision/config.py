from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

Rect = Tuple[int, int, int, int]


@dataclass(frozen=True)
class TargetConfig:
    name: str
    roi: Rect
    baseline_hz: Optional[float] = None
    frequency_shift_fraction_limit: float = 0.15
    relative_displacement_px_limit: Optional[float] = None


@dataclass(frozen=True)
class CameraConfig:
    width: int = 1280
    height: int = 720
    fps: int = 30
    warmup_seconds: float = 2.0


@dataclass(frozen=True)
class AnalysisConfig:
    window_seconds: float = 30.0
    minimum_samples: int = 180
    min_frequency_hz: float = 0.2
    max_frequency_hz: float = 5.0
    minimum_track_points: int = 6
    max_camera_motion_px: float = 1.5
    camera_settle_frames: int = 15
    inter_node_frequency_difference_limit_hz: Optional[float] = None
    minimum_confidence: float = 0.65


@dataclass(frozen=True)
class MagnificationConfig:
    enabled: bool = True
    target: str = ""
    alpha: float = 8.0
    pyramid_levels: int = 3
    low_hz: float = 0.4
    high_hz: float = 3.0


@dataclass(frozen=True)
class OutputConfig:
    directory: str = "evidence"
    save_video: bool = True
    show_preview: bool = True


@dataclass(frozen=True)
class AppConfig:
    camera: CameraConfig
    reference_roi: Rect
    targets: Tuple[TargetConfig, ...]
    analysis: AnalysisConfig
    magnification: MagnificationConfig
    output: OutputConfig

    def to_dict(self) -> dict:
        return asdict(self)


def _rect(value: object, field: str) -> Rect:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"{field} must be [x, y, width, height]")
    result = tuple(int(v) for v in value)
    if result[2] <= 0 or result[3] <= 0:
        raise ValueError(f"{field} width and height must be positive")
    return result  # type: ignore[return-value]


def load_config(path: Union[str, Path]) -> AppConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    camera = CameraConfig(**raw.get("camera", {}))
    targets = tuple(
        TargetConfig(
            name=item["name"], roi=_rect(item["roi"], f"targets[{item['name']}].roi"),
            baseline_hz=item.get("baseline_hz"),
            frequency_shift_fraction_limit=item.get("frequency_shift_fraction_limit", 0.15),
            relative_displacement_px_limit=item.get("relative_displacement_px_limit"),
        )
        for item in raw["targets"]
    )
    if not targets:
        raise ValueError("at least one target is required")
    analysis = AnalysisConfig(**raw.get("analysis", {}))
    if analysis.min_frequency_hz <= 0 or analysis.max_frequency_hz <= analysis.min_frequency_hz:
        raise ValueError("analysis frequency limits are invalid")
    if analysis.camera_settle_frames < 1:
        raise ValueError("analysis.camera_settle_frames must be positive")
    if analysis.inter_node_frequency_difference_limit_hz is not None and analysis.inter_node_frequency_difference_limit_hz <= 0:
        raise ValueError("analysis.inter_node_frequency_difference_limit_hz must be positive")
    magnification = MagnificationConfig(**raw.get("magnification", {}))
    if magnification.enabled and magnification.target not in {t.name for t in targets}:
        raise ValueError("magnification.target must match a configured target")
    return AppConfig(
        camera=camera,
        reference_roi=_rect(raw["reference_roi"], "reference_roi"),
        targets=targets,
        analysis=analysis,
        magnification=magnification,
        output=OutputConfig(**raw.get("output", {})),
    )
