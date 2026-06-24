"""File discovery and image I/O helpers.

The rest of the package intentionally works with NumPy arrays and compact TIFF
intermediates instead of keeping camera-specific readers spread through the
stacking code. This module is the small boundary layer that knows how to find
supported inputs, normalize common image/video frames, and write metadata.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import imageio.v3 as iio
import numpy as np
import tifffile


RAW_SUFFIXES = (".arw", ".ARW")
TIFF_SUFFIXES = (".tif", ".tiff", ".TIF", ".TIFF")
PNG_SUFFIXES = (".png", ".PNG")
JPEG_SUFFIXES = (".jpg", ".jpeg", ".JPG", ".JPEG")
BMP_SUFFIXES = (".bmp", ".BMP")
VIDEO_SUFFIXES = (
    ".mov",
    ".MOV",
    ".mp4",
    ".MP4",
    ".m4v",
    ".M4V",
    ".avi",
    ".AVI",
    ".mkv",
    ".MKV",
    ".webm",
    ".WEBM",
)
STACK_IMAGE_SUFFIXES = TIFF_SUFFIXES + PNG_SUFFIXES + JPEG_SUFFIXES + BMP_SUFFIXES
RAW_SUFFIX_SET = frozenset(suffix.lower() for suffix in RAW_SUFFIXES)
TIFF_SUFFIX_SET = frozenset(suffix.lower() for suffix in TIFF_SUFFIXES)
VIDEO_SUFFIX_SET = frozenset(suffix.lower() for suffix in VIDEO_SUFFIXES)


def natural_key(path: Path) -> list[object]:
    """Sort file paths in human order while preserving deterministic behavior."""

    parts = re.split(r"(\d+)", str(path))
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def list_files(
    input_path: Path | str,
    suffixes: Sequence[str],
    *,
    recursive: bool = False,
) -> list[Path]:
    """Return sorted files under a path matching the requested suffixes."""

    root = Path(input_path).expanduser()
    suffix_set = frozenset(suffix.lower() for suffix in suffixes)
    if root.is_file():
        return [root] if root.suffix.lower() in suffix_set else []

    pattern = "**/*" if recursive else "*"
    files = [path for path in root.glob(pattern) if path.is_file() and path.suffix.lower() in suffix_set]
    return sorted(files, key=natural_key)


def list_raw_files(input_path: Path | str, *, recursive: bool = False) -> list[Path]:
    return list_files(input_path, RAW_SUFFIXES, recursive=recursive)


def list_tiff_files(input_path: Path | str, *, recursive: bool = False) -> list[Path]:
    return list_files(input_path, TIFF_SUFFIXES, recursive=recursive)


def list_image_files(input_path: Path | str, *, recursive: bool = False) -> list[Path]:
    return list_files(input_path, STACK_IMAGE_SUFFIXES, recursive=recursive)


def list_video_files(input_path: Path | str, *, recursive: bool = False) -> list[Path]:
    return list_files(input_path, VIDEO_SUFFIXES, recursive=recursive)


def ensure_dir(path: Path | str) -> Path:
    directory = Path(path).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def read_tiff(path: Path | str) -> np.ndarray:
    return tifffile.imread(Path(path).expanduser())


def read_image(path: Path | str) -> np.ndarray:
    image_path = Path(path).expanduser()
    if image_path.suffix.lower() in TIFF_SUFFIX_SET:
        image = tifffile.imread(image_path)
    else:
        image = iio.imread(image_path)
    return normalize_stack_image(np.asarray(image))


def iter_video_frames(path: Path | str, *, frame_step: int = 1) -> Iterator[tuple[int, np.ndarray]]:
    """Yield normalized video frames without loading the full movie at once.

    ``frame_step`` is applied to the decoded frame index, so ``frame_step=5``
    keeps frames 0, 5, 10, and so on. The returned index is the original video
    frame number, which keeps output filenames traceable back to the capture.
    """

    frame_step = int(frame_step)
    if frame_step <= 0:
        raise ValueError("video_frame_step must be positive")
    video_path = Path(path).expanduser()
    for frame_index, frame in enumerate(iio.imiter(video_path)):
        if frame_index % frame_step != 0:
            continue
        yield frame_index, normalize_stack_image(np.asarray(frame))


def normalize_stack_image(image: np.ndarray) -> np.ndarray:
    """Return a grayscale/RGB image with stack-friendly numeric scaling."""

    # Drop alpha and extra channels up front. The stacker treats every frame as
    # grayscale or RGB signal; alpha masks from PNG/video decoders are display
    # metadata and would otherwise confuse shape comparisons.
    if image.ndim == 3 and image.shape[-1] >= 4:
        image = image[..., :3]
    if image.ndim == 3 and image.shape[-1] > 3:
        image = image[..., :3]
    # Most video codecs and many exported PNG/JPEG examples are 8-bit. Expanding
    # 0..255 to 0..65535 does not recover lost precision, but it keeps later
    # preview/postprocessing defaults calibrated to the same scale as 16-bit
    # TIFF crops.
    if image.dtype == np.uint8:
        return image.astype(np.uint16) * np.uint16(257)
    if image.dtype == np.bool_:
        return image.astype(np.uint16) * np.uint16(65535)
    return image


def write_tiff(
    path: Path | str,
    image: np.ndarray,
    *,
    compression: str | None = None,
    overwrite: bool = True,
) -> Path:
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {output_path}")
    kwargs: dict[str, object] = {}
    if image.ndim == 3 and image.shape[-1] in {3, 4}:
        kwargs["photometric"] = "rgb"
    tifffile.imwrite(output_path, image, compression=compression, **kwargs)
    return output_path


def write_json(path: Path | str, payload: object) -> Path:
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def write_jsonl(path: Path | str, records: Iterable[dict[str, object]]) -> Path:
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True))
            handle.write("\n")
    return output_path
