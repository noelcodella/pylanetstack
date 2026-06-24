from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CropRecord:
    """Metadata for one extracted crop.

    The crop manifest is deliberately verbose: it lets users audit which source
    frame produced each TIFF and gives future tools enough information to relate
    a compact crop back to the original full-resolution capture.
    """

    source_path: str
    crop_path: str
    output_mode: str
    crop_size: int
    top_left_x: int
    top_left_y: int
    detected_center_x: float
    detected_center_y: float
    crop_center_x: float
    crop_center_y: float
    source_width: int
    source_height: int
    dtype: str
    color_desc: str | None = None
    white_balance: str | None = None
    white_balance_multipliers: list[float] | None = None
    white_level: int | None = None
    black_level_per_channel: list[int] | None = None
    raw_crop_path: str | None = None
    source_type: str = "image"
    source_frame_index: int | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FrameRecord:
    """Metadata for one frame contribution to a stack."""

    path: str
    weight: float
    quality_score: float
    shift_y: float
    shift_x: float
    registration_error: float | None
    method: str
    flow_mean_magnitude: float | None = None
    flow_max_magnitude: float | None = None
    alignment_points: int | None = None
    channel_shifts: dict[str, list[float]] | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def path_text(path: Path) -> str:
    """Return a stable string path for metadata files."""

    return str(path.expanduser().resolve())
