"""Crop full-frame captures into compact stack-ready TIFF frames.

The crop stage is intentionally conservative with memory. RAW files are located
and cropped before demosaicing, rendered image files are cropped directly, and
video files are decoded one frame at a time. All paths produce the same compact
TIFF crop format so the stacker can stay independent of the original capture
container.
"""

from __future__ import annotations

import os
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage as ndi

from .image_math import luminance, planet_mask
from .io import (
    RAW_SUFFIX_SET,
    VIDEO_SUFFIX_SET,
    ensure_dir,
    iter_video_frames,
    list_image_files,
    list_raw_files,
    list_video_files,
    natural_key,
    read_image,
    write_jsonl,
    write_tiff,
)
from .types import CropRecord, path_text


AUTO_CROP_SIZE = "auto"
CropSize = int | str


def _resolve_worker_count(workers: int | None) -> int:
    if workers is None:
        return 1
    workers = int(workers)
    if workers < 0:
        raise ValueError("workers must be >= 0")
    if workers == 0:
        return max(1, os.cpu_count() or 1)
    return workers


def _import_rawpy():
    try:
        import rawpy  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "rawpy is required for ARW input. Install with `python -m pip install rawpy` "
            "or install this package with `python -m pip install -e .`."
        ) from exc
    return rawpy


def _is_raw_read_error(error: Exception) -> bool:
    """Return whether an exception came from LibRaw/rawpy while opening RAW data."""

    error_type = type(error)
    module = error_type.__module__.lower()
    name = error_type.__name__.lower()
    return "rawpy" in module or name.startswith("libraw")


def _remove_stale_crop_outputs(output_dir: Path, source: Path) -> None:
    """Remove stale crop outputs for a skipped source after an overwrite run."""

    for suffix in ("_crop.tif", "_raw_mosaic.tif"):
        path = output_dir / f"{source.stem}{suffix}"
        if path.exists():
            path.unlink()


def _is_raw_path(path: Path) -> bool:
    return path.suffix.lower() in RAW_SUFFIX_SET


def _is_video_path(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_SUFFIX_SET


def list_crop_input_files(input_path: Path | str, *, recursive: bool = False) -> list[Path]:
    """Return RAW, common image, and common video files that can be cropped."""

    files = list_raw_files(input_path, recursive=recursive)
    files.extend(list_image_files(input_path, recursive=recursive))
    files.extend(list_video_files(input_path, recursive=recursive))
    return sorted(set(files), key=natural_key)


def downsample_mean(image: np.ndarray, factor: int) -> np.ndarray:
    """Downsample a 2-D image with block means."""

    if factor <= 1:
        return image.astype(np.float32, copy=False)
    height, width = image.shape
    h2 = height // factor
    w2 = width // factor
    if h2 == 0 or w2 == 0:
        return image.astype(np.float32, copy=False)
    trimmed = image[: h2 * factor, : w2 * factor].astype(np.float32, copy=False)
    return trimmed.reshape(h2, factor, w2, factor).mean(axis=(1, 3))


def _robust_noise(image: np.ndarray) -> float:
    median = float(np.median(image))
    mad = float(np.median(np.abs(image - median)))
    return max(1.4826 * mad, 1e-6)


def _validate_fixed_crop_size(crop_size: int) -> int:
    size = int(crop_size)
    if size <= 0:
        raise ValueError("crop_size must be positive")
    return size


def _is_auto_crop_size(crop_size: CropSize) -> bool:
    return isinstance(crop_size, str) and crop_size.lower() == AUTO_CROP_SIZE


def _validate_auto_crop_options(
    *,
    margin: float,
    min_size: int,
    max_size: int | None,
) -> tuple[float, int, int | None]:
    margin = float(margin)
    min_size = int(min_size)
    if not np.isfinite(margin) or margin < 1.0:
        raise ValueError("auto_crop_margin must be a finite value >= 1")
    if min_size <= 0:
        raise ValueError("auto_crop_min_size must be positive")
    if max_size is not None:
        max_size = int(max_size)
        if max_size <= 0:
            raise ValueError("auto_crop_max_size must be positive")
        if max_size < min_size:
            raise ValueError("auto_crop_max_size must be >= auto_crop_min_size")
    return margin, min_size, max_size


def _otsu_threshold(image: np.ndarray, *, bins: int = 256) -> float:
    """Compute an Otsu threshold without requiring a full-size mask copy."""

    values = np.asarray(image, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    lo, hi = np.percentile(finite, [0.1, 99.9])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return float(np.max(finite))
    hist, edges = np.histogram(finite, bins=bins, range=(float(lo), float(hi)))
    total = float(hist.sum())
    if total <= 0:
        return float(np.median(finite))

    centers = (edges[:-1] + edges[1:]) * 0.5
    weight_background = np.cumsum(hist).astype(np.float64)
    weight_foreground = total - weight_background
    sum_background = np.cumsum(hist * centers).astype(np.float64)
    sum_total = float(sum_background[-1])
    mean_background = np.divide(
        sum_background,
        weight_background,
        out=np.zeros_like(sum_background),
        where=weight_background > 0,
    )
    mean_foreground = np.divide(
        sum_total - sum_background,
        weight_foreground,
        out=np.zeros_like(sum_background),
        where=weight_foreground > 0,
    )
    variance = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
    return float(centers[int(np.argmax(variance))])


def _largest_true_run(values: np.ndarray) -> tuple[int, int]:
    """Return inclusive start/end indices for the largest contiguous true run."""

    best_start = 0
    best_end = len(values) - 1
    best_length = 0
    current_start: int | None = None
    for index, value in enumerate(values):
        if bool(value):
            if current_start is None:
                current_start = index
            continue
        if current_start is not None:
            length = index - current_start
            if length > best_length:
                best_start = current_start
                best_end = index - 1
                best_length = length
            current_start = None
    if current_start is not None:
        length = len(values) - current_start
        if length > best_length:
            best_start = current_start
            best_end = len(values) - 1
            best_length = length
    return best_start, best_end


def auto_crop_geometry(
    image: np.ndarray,
    *,
    margin: float = 1.35,
    min_size: int = 250,
    max_size: int | None = None,
    downsample: int = 8,
) -> tuple[float, float, int]:
    """Find a comfortable square crop around the dominant bright object."""

    margin, min_size, max_size = _validate_auto_crop_options(
        margin=margin,
        min_size=min_size,
        max_size=max_size,
    )
    signal = luminance(image) if image.ndim == 3 else np.asarray(image)
    height, width = signal.shape[:2]
    small = downsample_mean(signal, downsample)
    smooth = ndi.gaussian_filter(small, sigma=1.2)
    median = float(np.median(smooth))
    noise = _robust_noise(smooth)
    robust_mask = smooth > (median + 6.0 * noise)
    robust_fraction = float(np.mean(robust_mask))

    if 0.0 < robust_fraction < 0.5:
        mask = robust_mask
    else:
        # Otsu is useful for large lunar/solar disks where much of the frame is
        # bright target signal and a fixed "background + N sigma" threshold can
        # under-estimate the object.
        threshold = _otsu_threshold(smooth)
        mask = smooth > threshold

    mask_fraction = float(np.mean(mask))
    if mask_fraction > 0.85:
        # A near-full mask usually means the frame is mostly target signal,
        # as can happen for close lunar/solar views.
        mask = np.ones_like(mask, dtype=bool)

    labels, count = ndi.label(mask)
    if count == 0 or not np.any(mask):
        center_y, center_x = find_planet_center(signal, crop_size=min_size, downsample=downsample)
        return center_y, center_x, min(min_size, max_size or min_size)

    areas = ndi.sum(mask, labels, index=np.arange(1, count + 1))
    label = int(np.argmax(areas)) + 1
    component = labels == label
    component = ndi.binary_fill_holes(component)
    rows = np.any(component, axis=1)
    cols = np.any(component, axis=0)
    y0_small, y1_small = _largest_true_run(rows)
    x0_small, x1_small = _largest_true_run(cols)

    y0 = max(0, y0_small * downsample)
    y1 = min(height, (y1_small + 1) * downsample)
    x0 = max(0, x0_small * downsample)
    x1 = min(width, (x1_small + 1) * downsample)
    box_height = max(1, y1 - y0)
    box_width = max(1, x1 - x0)
    source_limit = max(height, width)
    max_allowed = source_limit if max_size is None else min(max_size, source_limit)
    crop_size = int(np.ceil(max(box_height, box_width) * margin))
    crop_size = max(min_size, min(crop_size, max_allowed))

    if crop_size <= int(np.ceil(min_size * 1.5)):
        # For compact planets, a centroid of the bright disk is usually more
        # precise than the bounding-box midpoint after downsampling.
        center_y, center_x = find_planet_center(signal, crop_size=crop_size, downsample=downsample)
    else:
        center_y = (float(y0) + float(y1) - 1.0) * 0.5
        center_x = (float(x0) + float(x1) - 1.0) * 0.5
    return center_y, center_x, crop_size


def resolve_crop_geometry(
    image: np.ndarray,
    crop_size: CropSize,
    *,
    auto_crop_margin: float = 1.35,
    auto_crop_min_size: int = 250,
    auto_crop_max_size: int | None = None,
) -> tuple[float, float, int]:
    """Resolve fixed or automatic crop settings into center and square size."""

    if _is_auto_crop_size(crop_size):
        return auto_crop_geometry(
            image,
            margin=auto_crop_margin,
            min_size=auto_crop_min_size,
            max_size=auto_crop_max_size,
        )
    if isinstance(crop_size, str):
        raise ValueError("crop_size must be a positive integer or 'auto'")
    fixed_size = _validate_fixed_crop_size(crop_size)
    center_y, center_x = find_planet_center(image, crop_size=fixed_size)
    return center_y, center_x, fixed_size


def _sample_evenly(files: list[Path], count: int) -> list[Path]:
    """Return up to ``count`` files spread across the input sequence."""

    count = int(count)
    if count <= 0:
        raise ValueError("auto_crop_sample_frames must be positive")
    if len(files) <= count:
        return list(files)

    sampled: list[Path] = []
    seen: set[int] = set()
    for value in np.linspace(0, len(files) - 1, num=count):
        index = max(0, min(len(files) - 1, int(round(float(value)))))
        if index in seen:
            continue
        seen.add(index)
        sampled.append(files[index])
    return sampled


def estimate_auto_crop_size_for_file(
    source_path: Path | str,
    *,
    auto_crop_margin: float = 1.35,
    auto_crop_min_size: int = 250,
    auto_crop_max_size: int | None = None,
    video_frame_step: int = 1,
) -> int:
    """Estimate the automatic crop size needed for one source file."""

    source = Path(source_path).expanduser()
    if _is_raw_path(source):
        rawpy = _import_rawpy()
        with rawpy.imread(str(source)) as raw:
            _, _, crop_size = auto_crop_geometry(
                raw.raw_image_visible,
                margin=auto_crop_margin,
                min_size=auto_crop_min_size,
                max_size=auto_crop_max_size,
            )
        return crop_size

    if _is_video_path(source):
        for _frame_index, frame in iter_video_frames(source, frame_step=video_frame_step):
            _, _, crop_size = auto_crop_geometry(
                frame,
                margin=auto_crop_margin,
                min_size=auto_crop_min_size,
                max_size=auto_crop_max_size,
            )
            return crop_size
        raise RuntimeError(f"No readable frames found in video: {source}")

    image = read_image(source)
    _, _, crop_size = auto_crop_geometry(
        image,
        margin=auto_crop_margin,
        min_size=auto_crop_min_size,
        max_size=auto_crop_max_size,
    )
    return crop_size


def resolve_dataset_crop_size(
    files: list[Path],
    crop_size: CropSize,
    *,
    auto_crop_margin: float = 1.35,
    auto_crop_min_size: int = 250,
    auto_crop_max_size: int | None = None,
    auto_crop_sample_frames: int = 12,
    skip_bad_raw: bool = True,
    video_frame_step: int = 1,
) -> int:
    """Resolve crop size for a full batch, keeping automatic crops uniform."""

    if not _is_auto_crop_size(crop_size):
        if isinstance(crop_size, str):
            raise ValueError("crop_size must be a positive integer or 'auto'")
        return _validate_fixed_crop_size(crop_size)

    target_count = min(len(files), int(auto_crop_sample_frames))
    sample_files = _sample_evenly(files, target_count)
    sample_set = set(sample_files)
    fallback_files = [path for path in files if path not in sample_set]
    sizes: list[int] = []
    print(f"[crop:auto] sampling up to {target_count}/{len(files)} files", flush=True)

    for source in [*sample_files, *fallback_files]:
        if len(sizes) >= target_count:
            break
        try:
            estimated_size = estimate_auto_crop_size_for_file(
                source,
                auto_crop_margin=auto_crop_margin,
                auto_crop_min_size=auto_crop_min_size,
                auto_crop_max_size=auto_crop_max_size,
                video_frame_step=video_frame_step,
            )
        except Exception as error:
            if not _is_raw_path(source) or not skip_bad_raw or not _is_raw_read_error(error):
                raise
            print(
                f"[crop:auto:skip] {source.name}: {type(error).__name__}: {error}",
                flush=True,
            )
            continue
        sizes.append(estimated_size)
        print(f"[crop:auto] {source.name}: {estimated_size}px", flush=True)

    if not sizes:
        raise RuntimeError("No readable sample files were available for auto crop sizing")

    resolved_size = max(sizes)
    print(f"[crop:auto] using dataset crop size {resolved_size}px", flush=True)
    return resolved_size


def find_planet_center(
    raw_mosaic: np.ndarray,
    *,
    crop_size: int = 250,
    downsample: int = 8,
    threshold_sigma: float = 6.0,
) -> tuple[float, float]:
    """Find a planet centroid as ``(y, x)`` in full-resolution mosaic pixels."""

    small = downsample_mean(raw_mosaic, downsample)
    smooth = ndi.gaussian_filter(small, sigma=1.2)
    background = float(np.median(smooth))
    noise = _robust_noise(smooth)
    threshold = background + threshold_sigma * noise
    mask = smooth > threshold

    labels, count = ndi.label(mask)
    if count:
        values = np.clip(smooth - background, 0, None)
        scores = ndi.sum(values, labels, index=np.arange(1, count + 1))
        label = int(np.argmax(scores)) + 1
        center_y_small, center_x_small = ndi.center_of_mass(values, labels, label)
    else:
        center_y_small, center_x_small = np.unravel_index(int(np.argmax(smooth)), smooth.shape)

    center_y = (float(center_y_small) + 0.5) * downsample - 0.5
    center_x = (float(center_x_small) + 0.5) * downsample - 0.5
    return refine_center(raw_mosaic, center_y, center_x, crop_size=crop_size)


def refine_center(
    raw_mosaic: np.ndarray,
    center_y: float,
    center_x: float,
    *,
    crop_size: int,
) -> tuple[float, float]:
    """Refine a centroid on the original mosaic in a local window."""

    height, width = raw_mosaic.shape
    radius = max(crop_size, 64)
    y0 = max(0, int(round(center_y)) - radius)
    y1 = min(height, int(round(center_y)) + radius + 1)
    x0 = max(0, int(round(center_x)) - radius)
    x1 = min(width, int(round(center_x)) + radius + 1)
    patch = raw_mosaic[y0:y1, x0:x1].astype(np.float32, copy=False)
    if patch.size == 0:
        return center_y, center_x

    smooth = ndi.gaussian_filter(patch, sigma=2.0)
    background = float(np.median(smooth))
    noise = _robust_noise(smooth)
    weights = np.clip(smooth - (background + 2.5 * noise), 0, None)
    if float(weights.sum()) <= 0.0:
        py, px = np.unravel_index(int(np.argmax(smooth)), smooth.shape)
    else:
        py, px = ndi.center_of_mass(weights)
    return float(y0 + py), float(x0 + px)


def crop_bounds(center_y: float, center_x: float, crop_size: int) -> tuple[int, int, int, int]:
    """Return integer crop bounds as ``(y0, y1, x0, x1)``."""

    y0 = int(round(center_y)) - crop_size // 2
    x0 = int(round(center_x)) - crop_size // 2
    return y0, y0 + crop_size, x0, x0 + crop_size


def crop_with_padding(
    image: np.ndarray,
    *,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
    fill_value: int | float = 0,
) -> np.ndarray:
    """Crop an image, padding when the requested square crosses an edge."""

    height, width = image.shape[:2]
    src_y0 = max(0, y0)
    src_y1 = min(height, y1)
    src_x0 = max(0, x0)
    src_x1 = min(width, x1)
    crop = image[src_y0:src_y1, src_x0:src_x1]

    pad_before_y = src_y0 - y0
    pad_after_y = y1 - src_y1
    pad_before_x = src_x0 - x0
    pad_after_x = x1 - src_x1
    if not any((pad_before_y, pad_after_y, pad_before_x, pad_after_x)):
        return crop.copy()

    pad_width: list[tuple[int, int]] = [
        (pad_before_y, pad_after_y),
        (pad_before_x, pad_after_x),
    ]
    pad_width.extend((0, 0) for _ in image.shape[2:])
    return np.pad(crop, pad_width, mode="constant", constant_values=fill_value)


def demosaic_bilinear(
    mosaic: np.ndarray,
    color_ids: np.ndarray,
    color_desc: str,
) -> np.ndarray:
    """Bilinear-demosaic a small Bayer crop without touching the full frame."""

    mosaic_float = mosaic.astype(np.float32, copy=False)
    output = np.zeros((*mosaic.shape, 3), dtype=np.float32)
    kernel = np.array(
        [
            [1.0, 2.0, 1.0],
            [2.0, 4.0, 2.0],
            [1.0, 2.0, 1.0],
        ],
        dtype=np.float32,
    )
    channel_map = {"R": 0, "G": 1, "B": 2}

    for color_id in np.unique(color_ids):
        if color_id < 0 or color_id >= len(color_desc):
            continue
        channel = channel_map.get(color_desc[int(color_id)].upper())
        if channel is None:
            continue
        mask = color_ids == color_id
        samples = np.where(mask, mosaic_float, 0.0)
        sample_sum = ndi.convolve(samples, kernel, mode="mirror")
        sample_weight = ndi.convolve(mask.astype(np.float32), kernel, mode="mirror")
        interpolated = np.divide(
            sample_sum,
            sample_weight,
            out=np.zeros_like(sample_sum),
            where=sample_weight > 0,
        )
        if color_desc[int(color_id)].upper() == "G" and np.any(output[..., channel]):
            # Bayer mosaics have two green sites. Average the two interpolated
            # green planes so the final RGB crop does not inherit a green bias
            # simply from sample density.
            existing = output[..., channel]
            output[..., channel] = np.where(existing > 0, 0.5 * (existing + interpolated), interpolated)
        else:
            output[..., channel] = interpolated

    return np.clip(np.rint(output), 0, np.iinfo(mosaic.dtype).max).astype(mosaic.dtype)


def _camera_white_balance_multipliers(raw: Any) -> list[float] | None:
    """Return RGB multipliers normalized to green from RAW metadata."""

    raw_values = getattr(raw, "camera_whitebalance", None) or getattr(raw, "daylight_whitebalance", None)
    if not raw_values or len(raw_values) < 3:
        return None
    red = float(raw_values[0])
    green_values = [float(value) for value in raw_values[1::2] if float(value) > 0]
    green = float(np.mean(green_values)) if green_values else float(raw_values[1])
    blue = float(raw_values[2])
    if green <= 0 or red <= 0 or blue <= 0:
        return None
    return [red / green, 1.0, blue / green]


def _auto_white_balance_multipliers(image: np.ndarray, black_level: float) -> list[float]:
    """Estimate neutral RGB multipliers from the bright planet signal."""

    signal = np.clip(image.astype(np.float32) - black_level, 0.0, None)
    mask = planet_mask(luminance(signal))
    channel_levels = []
    for channel in range(3):
        values = signal[..., channel][mask]
        if values.size == 0 or float(np.max(values)) <= 0.0:
            values = signal[..., channel].ravel()
        level = float(np.percentile(values, 95.0))
        channel_levels.append(max(level, 1e-6))
    target = float(np.mean(channel_levels))
    multipliers = np.clip(target / np.asarray(channel_levels, dtype=np.float32), 0.25, 6.0)
    return [float(value) for value in multipliers]


def apply_white_balance(
    image: np.ndarray,
    *,
    multipliers: list[float],
    black_level: float,
) -> np.ndarray:
    """Apply RGB white balance to signal above the black level."""

    if image.ndim != 3 or image.shape[-1] < 3:
        return image
    balanced = image.astype(np.float32, copy=False)
    signal = np.clip(balanced[..., :3] - black_level, 0.0, None)
    multiplier_array = np.asarray(multipliers[:3], dtype=np.float32)
    balanced_rgb = signal * multiplier_array + black_level
    output = balanced.copy()
    output[..., :3] = balanced_rgb
    max_value = np.iinfo(image.dtype).max if np.issubdtype(image.dtype, np.integer) else np.inf
    return np.clip(np.rint(output), 0, max_value).astype(image.dtype, copy=False)


def crop_one_arw(
    source_path: Path | str,
    output_dir: Path | str,
    *,
    crop_size: CropSize = 250,
    auto_crop_margin: float = 1.35,
    auto_crop_min_size: int = 250,
    auto_crop_max_size: int | None = None,
    output_mode: str = "linear-rgb",
    white_balance: str = "camera",
    save_raw_crop: bool = False,
    compression: str | None = None,
    overwrite: bool = False,
) -> CropRecord:
    """Find the planet in one ARW file and write a compact crop."""

    if output_mode not in {"linear-rgb", "raw-mosaic"}:
        raise ValueError("output_mode must be 'linear-rgb' or 'raw-mosaic'")
    if white_balance not in {"camera", "auto", "none"}:
        raise ValueError("white_balance must be 'camera', 'auto', or 'none'")

    rawpy = _import_rawpy()
    source = Path(source_path).expanduser()
    out_dir = ensure_dir(output_dir)

    with rawpy.imread(str(source)) as raw:
        raw_mosaic = raw.raw_image_visible
        source_height, source_width = raw_mosaic.shape
        center_y, center_x, resolved_crop_size = resolve_crop_geometry(
            raw_mosaic,
            crop_size,
            auto_crop_margin=auto_crop_margin,
            auto_crop_min_size=auto_crop_min_size,
            auto_crop_max_size=auto_crop_max_size,
        )
        y0, y1, x0, x1 = crop_bounds(center_y, center_x, resolved_crop_size)
        mosaic_crop = crop_with_padding(raw_mosaic, y0=y0, y1=y1, x0=x0, x1=x1)

        color_desc_raw = getattr(raw, "color_desc", None)
        if isinstance(color_desc_raw, bytes):
            color_desc = color_desc_raw.decode("ascii", errors="ignore")
        else:
            color_desc = str(color_desc_raw) if color_desc_raw is not None else None

        crop_stem = source.stem
        raw_crop_path: Path | None = None
        applied_wb: list[float] | None = None
        if output_mode == "raw-mosaic":
            output_image = mosaic_crop
            suffix = "_raw_mosaic.tif"
        else:
            color_ids_full = raw.raw_colors_visible
            color_crop = crop_with_padding(color_ids_full, y0=y0, y1=y1, x0=x0, x1=x1, fill_value=-1)
            output_image = demosaic_bilinear(mosaic_crop, color_crop, color_desc or "RGBG")
            black_levels = getattr(raw, "black_level_per_channel", None)
            black_level = float(np.mean(black_levels)) if black_levels is not None else 0.0
            if white_balance == "camera":
                applied_wb = _camera_white_balance_multipliers(raw)
            elif white_balance == "auto":
                applied_wb = _auto_white_balance_multipliers(output_image, black_level)
            if applied_wb is not None:
                output_image = apply_white_balance(
                    output_image,
                    multipliers=applied_wb,
                    black_level=black_level,
                )
            suffix = "_crop.tif"
            if save_raw_crop:
                raw_crop_path = out_dir / f"{crop_stem}_raw_mosaic.tif"
                write_tiff(raw_crop_path, mosaic_crop, compression=compression, overwrite=overwrite)

        crop_path = out_dir / f"{crop_stem}{suffix}"
        write_tiff(crop_path, output_image, compression=compression, overwrite=overwrite)

        black_levels = getattr(raw, "black_level_per_channel", None)
        white_level = getattr(raw, "white_level", None)

    return CropRecord(
        source_path=path_text(source),
        crop_path=path_text(crop_path),
        output_mode=output_mode,
        crop_size=resolved_crop_size,
        top_left_x=x0,
        top_left_y=y0,
        detected_center_x=float(center_x),
        detected_center_y=float(center_y),
        crop_center_x=float(center_x - x0),
        crop_center_y=float(center_y - y0),
        source_width=int(source_width),
        source_height=int(source_height),
        dtype=str(mosaic_crop.dtype),
        color_desc=color_desc,
        white_balance=white_balance if output_mode != "raw-mosaic" else None,
        white_balance_multipliers=applied_wb,
        white_level=int(white_level) if white_level is not None else None,
        black_level_per_channel=[int(value) for value in black_levels] if black_levels is not None else None,
        raw_crop_path=path_text(raw_crop_path) if raw_crop_path else None,
        source_type="raw",
    )


def crop_one_image(
    source_path: Path | str,
    output_dir: Path | str,
    *,
    crop_size: CropSize = 250,
    auto_crop_margin: float = 1.35,
    auto_crop_min_size: int = 250,
    auto_crop_max_size: int | None = None,
    output_mode: str = "linear-rgb",
    compression: str | None = None,
    overwrite: bool = False,
) -> CropRecord:
    """Find the planet in one already-rendered image file and write a TIFF crop."""

    if output_mode == "raw-mosaic":
        raise ValueError("output_mode='raw-mosaic' is only available for RAW inputs")
    if output_mode != "linear-rgb":
        raise ValueError("output_mode must be 'linear-rgb' for image inputs")

    source = Path(source_path).expanduser()
    image = read_image(source)
    return crop_rendered_frame(
        image,
        source,
        output_dir,
        crop_stem=source.stem,
        source_type="image",
        crop_size=crop_size,
        auto_crop_margin=auto_crop_margin,
        auto_crop_min_size=auto_crop_min_size,
        auto_crop_max_size=auto_crop_max_size,
        compression=compression,
        overwrite=overwrite,
    )


def crop_one_video_frame(
    source_path: Path | str,
    output_dir: Path | str,
    frame: np.ndarray,
    *,
    frame_index: int,
    crop_size: CropSize = 250,
    auto_crop_margin: float = 1.35,
    auto_crop_min_size: int = 250,
    auto_crop_max_size: int | None = None,
    output_mode: str = "linear-rgb",
    compression: str | None = None,
    overwrite: bool = False,
) -> CropRecord:
    """Crop one decoded video frame and write it as a stack-ready TIFF."""

    if output_mode == "raw-mosaic":
        raise ValueError("output_mode='raw-mosaic' is only available for RAW inputs")
    if output_mode != "linear-rgb":
        raise ValueError("output_mode must be 'linear-rgb' for video inputs")

    source = Path(source_path).expanduser()
    crop_stem = f"{source.stem}_frame{int(frame_index):06d}"
    return crop_rendered_frame(
        frame,
        source,
        output_dir,
        crop_stem=crop_stem,
        source_type="video",
        source_frame_index=int(frame_index),
        crop_size=crop_size,
        auto_crop_margin=auto_crop_margin,
        auto_crop_min_size=auto_crop_min_size,
        auto_crop_max_size=auto_crop_max_size,
        compression=compression,
        overwrite=overwrite,
    )


def crop_rendered_frame(
    image: np.ndarray,
    source: Path,
    output_dir: Path | str,
    *,
    crop_stem: str,
    source_type: str,
    source_frame_index: int | None = None,
    crop_size: CropSize = 250,
    auto_crop_margin: float = 1.35,
    auto_crop_min_size: int = 250,
    auto_crop_max_size: int | None = None,
    compression: str | None = None,
    overwrite: bool = False,
) -> CropRecord:
    """Shared crop implementation for rendered still images and video frames."""

    out_dir = ensure_dir(output_dir)
    source_height, source_width = image.shape[:2]
    center_y, center_x, resolved_crop_size = resolve_crop_geometry(
        luminance(image),
        crop_size,
        auto_crop_margin=auto_crop_margin,
        auto_crop_min_size=auto_crop_min_size,
        auto_crop_max_size=auto_crop_max_size,
    )
    y0, y1, x0, x1 = crop_bounds(center_y, center_x, resolved_crop_size)
    output_image = crop_with_padding(image, y0=y0, y1=y1, x0=x0, x1=x1)

    crop_path = out_dir / f"{crop_stem}_crop.tif"
    write_tiff(crop_path, output_image, compression=compression, overwrite=overwrite)

    return CropRecord(
        source_path=path_text(source),
        crop_path=path_text(crop_path),
        output_mode="image",
        crop_size=resolved_crop_size,
        top_left_x=x0,
        top_left_y=y0,
        detected_center_x=float(center_x),
        detected_center_y=float(center_y),
        crop_center_x=float(center_x - x0),
        crop_center_y=float(center_y - y0),
        source_width=int(source_width),
        source_height=int(source_height),
        dtype=str(output_image.dtype),
        source_type=source_type,
        source_frame_index=source_frame_index,
    )


def crop_arw_directory(
    input_path: Path | str,
    output_dir: Path | str,
    *,
    crop_size: CropSize = 250,
    auto_crop_margin: float = 1.35,
    auto_crop_min_size: int = 250,
    auto_crop_max_size: int | None = None,
    auto_crop_sample_frames: int = 12,
    output_mode: str = "linear-rgb",
    white_balance: str = "camera",
    recursive: bool = False,
    save_raw_crop: bool = False,
    compression: str | None = None,
    overwrite: bool = False,
    max_frames: int | None = None,
    manifest_name: str = "crops.jsonl",
    error_manifest_name: str = "crop_errors.jsonl",
    skip_bad_raw: bool = True,
    workers: int = 8,
    video_frame_step: int = 1,
) -> list[CropRecord]:
    """Crop every supported still/video source in a folder and write manifests."""

    files = list_crop_input_files(input_path, recursive=recursive)
    if not files:
        raise FileNotFoundError(f"No RAW, video, or stackable image files found in {input_path}")

    out_dir = ensure_dir(output_dir)
    has_video_inputs = any(_is_video_path(source) for source in files)
    if not has_video_inputs and max_frames is not None:
        files = files[:max_frames]
    if not files:
        raise RuntimeError(f"No crops were requested from {input_path}")

    video_frame_step = int(video_frame_step)
    if video_frame_step <= 0:
        raise ValueError("video_frame_step must be positive")
    resolved_crop_size = resolve_dataset_crop_size(
        files,
        crop_size,
        auto_crop_margin=auto_crop_margin,
        auto_crop_min_size=auto_crop_min_size,
        auto_crop_max_size=auto_crop_max_size,
        auto_crop_sample_frames=auto_crop_sample_frames,
        skip_bad_raw=skip_bad_raw,
        video_frame_step=video_frame_step,
    )
    worker_count = _resolve_worker_count(workers)
    if worker_count > 1:
        print(f"[crop] using {worker_count} worker threads", flush=True)

    if has_video_inputs:
        records: list[CropRecord] = []
        errors: list[dict[str, object]] = []
        frame_limit = int(max_frames) if max_frames is not None else None
        limit_text = f"/{frame_limit}" if frame_limit is not None else ""

        def maybe_stop() -> bool:
            return frame_limit is not None and len(records) >= frame_limit

        for source_index, source in enumerate(files, start=1):
            if maybe_stop():
                break
            if _is_video_path(source):
                # Video files are processed serially so decoded frames can be
                # cropped and released immediately. This avoids accidentally
                # materializing thousands of frames in RAM.
                print(
                    f"[crop:video] source {source_index}/{len(files)} {source.name} "
                    f"frame_step={video_frame_step}",
                    flush=True,
                )
                for frame_index, frame in iter_video_frames(source, frame_step=video_frame_step):
                    if maybe_stop():
                        break
                    print(
                        f"[crop] {len(records) + 1}{limit_text} "
                        f"{source.name}:frame{frame_index:06d}",
                        flush=True,
                    )
                    record = crop_one_video_frame(
                        source,
                        out_dir,
                        frame,
                        frame_index=frame_index,
                        crop_size=resolved_crop_size,
                        output_mode=output_mode,
                        compression=compression,
                        overwrite=overwrite,
                    )
                    records.append(record)
                continue

            print(f"[crop] {len(records) + 1}{limit_text} {source.name}", flush=True)
            try:
                if _is_raw_path(source):
                    record = crop_one_arw(
                        source,
                        out_dir,
                        crop_size=resolved_crop_size,
                        output_mode=output_mode,
                        white_balance=white_balance,
                        save_raw_crop=save_raw_crop,
                        compression=compression,
                        overwrite=overwrite,
                    )
                else:
                    record = crop_one_image(
                        source,
                        out_dir,
                        crop_size=resolved_crop_size,
                        output_mode=output_mode,
                        compression=compression,
                        overwrite=overwrite,
                    )
            except Exception as error:
                if not _is_raw_path(source) or not skip_bad_raw or not _is_raw_read_error(error):
                    raise
                if overwrite:
                    _remove_stale_crop_outputs(out_dir, source)
                error_record = {
                    "source_path": path_text(source),
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
                errors.append(error_record)
                print(
                    f"[crop:skip] {source.name}: {type(error).__name__}: {error}",
                    flush=True,
                )
                continue
            records.append(record)

        if not records:
            detail = f"; {len(errors)} RAW files failed" if errors else ""
            raise RuntimeError(f"No crops were written from {input_path}{detail}")

        manifest_path = out_dir / manifest_name
        write_jsonl(manifest_path, (record.to_json() for record in records))
        if errors:
            write_jsonl(out_dir / error_manifest_name, errors)
        return records

    records_by_index: list[CropRecord | None] = [None for _file in files]
    errors: list[dict[str, object]] = []

    def process_source(index: int, source: Path) -> tuple[int, CropRecord | None, dict[str, object] | None]:
        print(f"[crop] {index}/{len(files)} {source.name}", flush=True)
        try:
            if _is_raw_path(source):
                record = crop_one_arw(
                    source,
                    out_dir,
                    crop_size=resolved_crop_size,
                    output_mode=output_mode,
                    white_balance=white_balance,
                    save_raw_crop=save_raw_crop,
                    compression=compression,
                    overwrite=overwrite,
                )
            else:
                record = crop_one_image(
                    source,
                    out_dir,
                    crop_size=resolved_crop_size,
                    output_mode=output_mode,
                    compression=compression,
                    overwrite=overwrite,
                )
        except Exception as error:
            if not _is_raw_path(source) or not skip_bad_raw or not _is_raw_read_error(error):
                raise
            if overwrite:
                _remove_stale_crop_outputs(out_dir, source)
            error_record = {
                "source_path": path_text(source),
                "error_type": type(error).__name__,
                "error": str(error),
            }
            print(
                f"[crop:skip] {source.name}: {type(error).__name__}: {error}",
                flush=True,
            )
            return index - 1, None, error_record
        return index - 1, record, None

    def consume_result(result: tuple[int, CropRecord | None, dict[str, object] | None]) -> None:
        record_index, record, error_record = result
        if error_record is not None:
            errors.append(error_record)
        if record is not None:
            records_by_index[record_index] = record

    if worker_count == 1 or len(files) <= 1:
        for index, source in enumerate(files, start=1):
            consume_result(process_source(index, source))
    else:
        next_index = 0
        pending: dict[Any, None] = {}
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            while next_index < len(files) or pending:
                while next_index < len(files) and len(pending) < worker_count:
                    future = executor.submit(process_source, next_index + 1, files[next_index])
                    pending[future] = None
                    next_index += 1
                done, _not_done = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    pending.pop(future)
                    consume_result(future.result())

    records = [record for record in records_by_index if record is not None]

    if not records:
        detail = f"; {len(errors)} RAW files failed" if errors else ""
        raise RuntimeError(f"No crops were written from {input_path}{detail}")

    manifest_path = out_dir / manifest_name
    write_jsonl(manifest_path, (record.to_json() for record in records))
    if errors:
        write_jsonl(out_dir / error_manifest_name, errors)
    return records
