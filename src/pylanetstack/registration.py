"""Registration primitives used by the stacker.

Registration starts with a global phase-correlation shift, then optionally adds
local correction with either alignment points (`ap`) or dense TV-L1 optical flow
(`flow`). The returned `RegistrationInfo` stores both the global shift and any
local flow fields so stack-time upsampling can resample directly onto a larger
grid when requested.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from .image_math import luminance, planet_mask, robust_normalize


@dataclass(frozen=True)
class RegistrationInfo:
    method: str
    shift_y: float
    shift_x: float
    error: float | None
    flow_mean_magnitude: float | None = None
    flow_max_magnitude: float | None = None
    alignment_points: int | None = None
    flow_y: np.ndarray | None = None
    flow_x: np.ndarray | None = None


def _ensure_skimage_cache() -> None:
    """Keep scikit-image imports usable in sandboxed environments."""

    if "SKIMAGE_DATADIR" not in os.environ:
        cache_dir = Path(tempfile.gettempdir()) / "pylanetstack-skimage-cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["SKIMAGE_DATADIR"] = str(cache_dir)


def _phase_cross_correlation(reference: np.ndarray, moving: np.ndarray, upsample_factor: int):
    _ensure_skimage_cache()
    from skimage.registration import phase_cross_correlation

    return phase_cross_correlation(
        reference,
        moving,
        upsample_factor=upsample_factor,
        normalization="phase",
    )


def _optical_flow_tvl1(
    reference: np.ndarray,
    moving: np.ndarray,
    *,
    num_warp: int,
    num_iter: int,
    attachment: float,
    tightness: float,
) -> tuple[np.ndarray, np.ndarray]:
    _ensure_skimage_cache()
    from skimage.registration import optical_flow_tvl1

    flow_y, flow_x = optical_flow_tvl1(
        reference,
        moving,
        num_warp=num_warp,
        num_iter=num_iter,
        attachment=attachment,
        tightness=tightness,
        prefilter=True,
    )
    return flow_y.astype(np.float32), flow_x.astype(np.float32)


def shift_image(image: np.ndarray, shift_y: float, shift_x: float) -> np.ndarray:
    """Apply a sub-pixel translation to a grayscale or channel-last image."""

    array = image.astype(np.float32, copy=False)
    if array.ndim == 2:
        shift = (shift_y, shift_x)
    elif array.ndim == 3:
        shift = (shift_y, shift_x, 0.0)
    else:
        raise ValueError(f"Unsupported image shape for shift: {array.shape}")
    return ndi.shift(array, shift=shift, order=3, mode="nearest", prefilter=True)


def warp_with_flow(image: np.ndarray, flow_y: np.ndarray, flow_x: np.ndarray) -> np.ndarray:
    """Warp an image according to row and column flow fields."""

    array = image.astype(np.float32, copy=False)
    height, width = flow_y.shape
    rows, cols = np.meshgrid(
        np.arange(height, dtype=np.float32),
        np.arange(width, dtype=np.float32),
        indexing="ij",
    )
    coords = np.array([rows + flow_y, cols + flow_x], dtype=np.float32)

    if array.ndim == 2:
        return ndi.map_coordinates(array, coords, order=3, mode="nearest", prefilter=True).astype(np.float32)
    if array.ndim == 3:
        channels = [
            ndi.map_coordinates(array[..., channel], coords, order=3, mode="nearest", prefilter=True)
            for channel in range(array.shape[2])
        ]
        return np.stack(channels, axis=-1).astype(np.float32)
    raise ValueError(f"Unsupported image shape for warp: {array.shape}")


def registered_frame_to_grid(
    image: np.ndarray,
    info: RegistrationInfo,
    *,
    scale: float = 1.0,
) -> np.ndarray:
    """Resample a registered frame directly onto an enlarged output grid."""

    scale = float(scale)
    if not np.isfinite(scale) or scale < 1.0:
        raise ValueError("stack upsample must be a finite value >= 1")
    if scale == 1.0:
        shifted = shift_image(image, info.shift_y, info.shift_x)
        if info.flow_y is not None and info.flow_x is not None:
            return warp_with_flow(shifted, info.flow_y, info.flow_x)
        return shifted

    array = image.astype(np.float32, copy=False)
    height, width = array.shape[:2]
    output_height = int(round(height * scale))
    output_width = int(round(width * scale))
    rows, cols = np.meshgrid(
        np.arange(output_height, dtype=np.float32),
        np.arange(output_width, dtype=np.float32),
        indexing="ij",
    )
    rows = (rows + 0.5) / scale - 0.5
    cols = (cols + 0.5) / scale - 0.5

    # When stack-time upsampling is enabled, resample the local flow field onto
    # the larger output grid first. This preserves sub-pixel offsets instead of
    # warping at native size and then enlarging the result afterward.
    if info.flow_y is not None and info.flow_x is not None:
        flow_coords = np.array([rows, cols], dtype=np.float32)
        flow_y = ndi.map_coordinates(info.flow_y, flow_coords, order=3, mode="nearest", prefilter=True)
        flow_x = ndi.map_coordinates(info.flow_x, flow_coords, order=3, mode="nearest", prefilter=True)
    else:
        flow_y = 0.0
        flow_x = 0.0

    coords = np.array(
        [
            rows + flow_y - float(info.shift_y),
            cols + flow_x - float(info.shift_x),
        ],
        dtype=np.float32,
    )
    if array.ndim == 2:
        return ndi.map_coordinates(array, coords, order=3, mode="nearest", prefilter=True).astype(np.float32)
    if array.ndim == 3:
        channels = [
            ndi.map_coordinates(array[..., channel], coords, order=3, mode="nearest", prefilter=True)
            for channel in range(array.shape[2])
        ]
        return np.stack(channels, axis=-1).astype(np.float32)
    raise ValueError(f"Unsupported image shape for registered resampling: {array.shape}")


def regularize_flow(
    flow_y: np.ndarray,
    flow_x: np.ndarray,
    *,
    smoothing_sigma: float = 0.7,
    max_pixels: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Smooth and magnitude-limit an optical-flow field."""

    y = flow_y.astype(np.float32, copy=True)
    x = flow_x.astype(np.float32, copy=True)
    if smoothing_sigma > 0:
        # Atmospheric distortion varies smoothly across neighboring pixels.
        # Smoothing keeps the local warp from chasing single-pixel noise.
        y = ndi.gaussian_filter(y, sigma=smoothing_sigma)
        x = ndi.gaussian_filter(x, sigma=smoothing_sigma)

    if max_pixels > 0:
        # Very large local shifts are usually registration failures for compact
        # planets. Limiting the vector magnitude keeps one poor AP/flow estimate
        # from dragging the stack into a ghost image.
        magnitude = np.hypot(y, x)
        scale = np.minimum(1.0, max_pixels / np.maximum(magnitude, 1e-6))
        y *= scale
        x *= scale
    return y.astype(np.float32), x.astype(np.float32)


def flow_stats(flow_y: np.ndarray, flow_x: np.ndarray) -> tuple[float, float]:
    """Return mean and maximum flow-vector magnitudes."""

    magnitude = np.hypot(flow_y, flow_x)
    return float(np.mean(magnitude)), float(np.max(magnitude))


def alignment_point_centers(
    reference_gray: np.ndarray,
    *,
    patch_size: int = 48,
    spacing: int = 24,
    max_points: int = 2500,
    min_mask_fraction: float = 0.25,
    min_contrast: float = 0.015,
) -> list[tuple[int, int]]:
    """Choose local alignment point centers."""

    if patch_size < 8:
        raise ValueError("ap_patch_size must be at least 8")
    if spacing < 4:
        raise ValueError("ap_spacing must be at least 4")
    if not (0.0 <= min_mask_fraction <= 1.0):
        raise ValueError("ap_min_mask_fraction must be between 0 and 1")

    image = robust_normalize(reference_gray)
    mask = planet_mask(image)
    height, width = image.shape
    half = patch_size // 2
    centers: list[tuple[int, int]] = []

    max_points = int(max_points)
    if max_points < 0:
        raise ValueError("ap_max_points must be >= 0")
    effective_spacing = int(spacing)
    y_count = len(range(half, max(half + 1, height - half + 1), effective_spacing))
    x_count = len(range(half, max(half + 1, width - half + 1), effective_spacing))
    candidate_count = y_count * x_count
    if max_points > 0 and candidate_count > max_points:
        # Lunar and solar crops can generate enormous AP grids. Increasing the
        # spacing keeps the runtime bounded while still covering the full disk.
        scale = float(np.sqrt(candidate_count / max_points))
        effective_spacing = max(effective_spacing, int(np.ceil(effective_spacing * scale)))
        print(
            f"[ap] candidate grid {candidate_count:,} exceeds cap {max_points:,}; "
            f"using spacing {effective_spacing}px",
            flush=True,
        )

    y_values = range(half, max(half + 1, height - half + 1), effective_spacing)
    x_values = range(half, max(half + 1, width - half + 1), effective_spacing)
    for y in y_values:
        for x in x_values:
            y0 = y - half
            y1 = y0 + patch_size
            x0 = x - half
            x1 = x0 + patch_size
            if y0 < 0 or x0 < 0 or y1 > height or x1 > width:
                continue
            patch_mask = mask[y0:y1, x0:x1]
            if float(np.mean(patch_mask)) < min_mask_fraction:
                continue
            patch = image[y0:y1, x0:x1]
            patch_values = patch[patch_mask] if np.any(patch_mask) else patch.ravel()
            if float(np.std(patch_values)) < min_contrast:
                continue
            centers.append((y, x))

    if max_points > 0 and len(centers) > max_points:
        indices = np.linspace(0, len(centers) - 1, num=max_points)
        centers = [centers[int(round(index))] for index in indices]
        print(f"[ap] selected {len(centers):,} alignment points after cap", flush=True)

    if centers:
        return centers

    center = ndi.center_of_mass(mask.astype(np.float32))
    if not np.all(np.isfinite(center)):
        return [(height // 2, width // 2)]
    return [(int(round(center[0])), int(round(center[1])))]


def estimate_alignment_point_shifts(
    reference_gray: np.ndarray,
    moving_gray: np.ndarray,
    centers: list[tuple[int, int]],
    *,
    patch_size: int = 48,
    upsample_factor: int = 20,
    max_shift_pixels: float = 2.0,
    min_contrast: float = 0.015,
) -> list[tuple[int, int, float, float]]:
    """Estimate local residual shifts for alignment points."""

    height, width = reference_gray.shape
    half = patch_size // 2
    window_1d = np.hanning(patch_size).astype(np.float32)
    window = np.outer(window_1d, window_1d)
    shifts: list[tuple[int, int, float, float]] = []

    for y, x in centers:
        y0 = y - half
        y1 = y0 + patch_size
        x0 = x - half
        x1 = x0 + patch_size
        if y0 < 0 or x0 < 0 or y1 > height or x1 > width:
            continue

        reference_patch = reference_gray[y0:y1, x0:x1].astype(np.float32, copy=False)
        moving_patch = moving_gray[y0:y1, x0:x1].astype(np.float32, copy=False)
        if float(np.std(reference_patch)) < min_contrast or float(np.std(moving_patch)) < min_contrast:
            continue

        ref = (reference_patch - float(np.mean(reference_patch))) * window
        mov = (moving_patch - float(np.mean(moving_patch))) * window
        # Windowing makes phase correlation less sensitive to the hard edge of
        # each AP patch, improving sub-pixel shift estimates on small regions.
        shift, _error, _phase = _phase_cross_correlation(ref, mov, upsample_factor)
        shift_y = float(shift[0])
        shift_x = float(shift[1])
        if not np.isfinite(shift_y) or not np.isfinite(shift_x):
            continue
        magnitude = float(np.hypot(shift_y, shift_x))
        if max_shift_pixels > 0 and magnitude > max_shift_pixels:
            scale = max_shift_pixels / magnitude
            shift_y *= scale
            shift_x *= scale
        shifts.append((y, x, shift_y, shift_x))

    return shifts


def alignment_points_to_flow(
    shape: tuple[int, int],
    shifts: list[tuple[int, int, float, float]],
    *,
    blend_sigma: float = 18.0,
    smoothing_sigma: float = 1.0,
    max_pixels: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Blend sparse AP shifts into a smooth dense flow field."""

    height, width = shape
    if not shifts:
        return np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=np.float32)

    sigma = max(float(blend_sigma), 1.0, _estimate_ap_spacing(shifts) * 0.75)
    denominator = np.zeros(shape, dtype=np.float32)
    flow_y = np.zeros(shape, dtype=np.float32)
    flow_x = np.zeros(shape, dtype=np.float32)

    for y, x, shift_y, shift_x in shifts:
        y_index = max(0, min(height - 1, int(round(y))))
        x_index = max(0, min(width - 1, int(round(x))))
        denominator[y_index, x_index] += 1.0
        # map_coordinates samples input at output+flow. A phase-correlation
        # shift is the desired output displacement, so the dense flow is negated.
        flow_y[y_index, x_index] += -shift_y
        flow_x[y_index, x_index] += -shift_x

    print(f"[ap] blending {len(shifts):,} AP shifts with sigma {sigma:.1f}px", flush=True)
    ndi.gaussian_filter(denominator, sigma=sigma, mode="nearest", output=denominator)
    ndi.gaussian_filter(flow_y, sigma=sigma, mode="nearest", output=flow_y)
    ndi.gaussian_filter(flow_x, sigma=sigma, mode="nearest", output=flow_x)

    flow_y = np.divide(flow_y, denominator, out=np.zeros_like(flow_y), where=denominator > 1e-12)
    flow_x = np.divide(flow_x, denominator, out=np.zeros_like(flow_x), where=denominator > 1e-12)
    return regularize_flow(
        flow_y,
        flow_x,
        smoothing_sigma=smoothing_sigma,
        max_pixels=max_pixels,
    )


def _estimate_ap_spacing(shifts: list[tuple[int, int, float, float]]) -> float:
    """Estimate AP grid spacing from sparse shift locations."""

    if len(shifts) < 2:
        return 0.0
    ys = np.array(sorted({int(y) for y, _x, _shift_y, _shift_x in shifts}), dtype=np.float32)
    xs = np.array(sorted({int(x) for _y, x, _shift_y, _shift_x in shifts}), dtype=np.float32)
    spacings: list[float] = []
    for values in (ys, xs):
        diffs = np.diff(values)
        diffs = diffs[diffs > 0]
        if diffs.size:
            spacings.append(float(np.median(diffs)))
    return max(spacings) if spacings else 0.0


CHANNEL_NAME_TO_INDEX = {
    "red": 0,
    "green": 1,
    "blue": 2,
}
CHANNEL_INDEX_TO_NAME = {index: name for name, index in CHANNEL_NAME_TO_INDEX.items()}


def align_color_channels(
    image: np.ndarray,
    *,
    reference_channel: str = "green",
    max_shift_pixels: float = 3.0,
    upsample_factor: int = 20,
) -> tuple[np.ndarray, dict[str, list[float]]]:
    """Align RGB channels to one reference channel to reduce color fringing."""

    if image.ndim != 3 or image.shape[-1] < 3:
        return image.astype(np.float32, copy=False), {}
    if reference_channel not in CHANNEL_NAME_TO_INDEX:
        raise ValueError("reference_channel must be 'red', 'green', or 'blue'")

    array = image.astype(np.float32, copy=False)
    reference_index = CHANNEL_NAME_TO_INDEX[reference_channel]
    reference = robust_normalize(array[..., reference_index])
    aligned = array.copy()
    shifts: dict[str, list[float]] = {}

    for channel in range(3):
        if channel == reference_index:
            continue
        shift, _error, _phase = _phase_cross_correlation(
            reference,
            robust_normalize(array[..., channel]),
            upsample_factor,
        )
        shift_y = float(shift[0])
        shift_x = float(shift[1])
        magnitude = float(np.hypot(shift_y, shift_x))
        if max_shift_pixels > 0 and magnitude > max_shift_pixels:
            scale = max_shift_pixels / magnitude
            shift_y *= scale
            shift_x *= scale
        aligned[..., channel] = shift_image(array[..., channel], shift_y, shift_x)
        shifts[CHANNEL_INDEX_TO_NAME[channel]] = [shift_y, shift_x]

    return aligned.astype(np.float32), shifts


def register_frame(
    moving: np.ndarray,
    reference: np.ndarray,
    *,
    method: str = "ap",
    upsample_factor: int = 20,
    max_global_shift: float = 0.0,
    flow_warps: int = 5,
    flow_iterations: int = 10,
    flow_attachment: float = 15.0,
    flow_tightness: float = 0.3,
    flow_smoothing_sigma: float = 0.1,
    flow_max_pixels: float = 2.0,
    ap_patch_size: int = 12,
    ap_spacing: int = 6,
    ap_blend_sigma: float = 18.0,
    ap_max_points: int = 2500,
    ap_min_mask_fraction: float = 0.25,
    ap_min_contrast: float = 0.015,
) -> tuple[np.ndarray, RegistrationInfo]:
    """Register one frame to a reference and return the warped moving frame."""

    if method not in {"none", "rigid", "flow", "ap"}:
        raise ValueError("method must be 'none', 'rigid', 'flow', or 'ap'")
    max_global_shift = float(max_global_shift)
    if max_global_shift < 0:
        raise ValueError("max_global_shift must be >= 0")

    moving_float = moving.astype(np.float32, copy=False)
    if method == "none":
        return moving_float, RegistrationInfo(method=method, shift_y=0.0, shift_x=0.0, error=None)

    reference_gray = robust_normalize(luminance(reference))
    moving_gray = robust_normalize(luminance(moving_float))
    # Every nonlinear method starts with a global translation. This handles
    # mount drift and crop-center jitter before local AP/flow corrections try to
    # model seeing-induced distortion.
    shift, error, _phase = _phase_cross_correlation(reference_gray, moving_gray, upsample_factor)
    shift_y = float(shift[0])
    shift_x = float(shift[1])
    if max_global_shift > 0 and float(np.hypot(shift_y, shift_x)) > max_global_shift:
        return moving_float, RegistrationInfo(method=method, shift_y=shift_y, shift_x=shift_x, error=float(error))

    shifted = shift_image(moving_float, shift_y, shift_x)

    if method == "rigid":
        return shifted, RegistrationInfo(method=method, shift_y=shift_y, shift_x=shift_x, error=float(error))

    shifted_gray = robust_normalize(luminance(shifted))
    if method == "ap":
        centers = alignment_point_centers(
            reference_gray,
            patch_size=ap_patch_size,
            spacing=ap_spacing,
            max_points=ap_max_points,
            min_mask_fraction=ap_min_mask_fraction,
            min_contrast=ap_min_contrast,
        )
        print(f"[ap] estimating shifts for {len(centers):,} alignment points", flush=True)
        ap_shifts = estimate_alignment_point_shifts(
            reference_gray,
            shifted_gray,
            centers,
            patch_size=ap_patch_size,
            upsample_factor=upsample_factor,
            max_shift_pixels=flow_max_pixels,
            min_contrast=ap_min_contrast,
        )
        print(f"[ap] accepted {len(ap_shifts):,} alignment-point shifts", flush=True)
        flow_y, flow_x = alignment_points_to_flow(
            reference_gray.shape,
            ap_shifts,
            blend_sigma=ap_blend_sigma,
            smoothing_sigma=flow_smoothing_sigma,
            max_pixels=flow_max_pixels,
        )
        flow_mean, flow_max = flow_stats(flow_y, flow_x)
        warped = warp_with_flow(shifted, flow_y, flow_x)
        return warped, RegistrationInfo(
            method=method,
            shift_y=shift_y,
            shift_x=shift_x,
            error=float(error),
            flow_mean_magnitude=flow_mean,
            flow_max_magnitude=flow_max,
            alignment_points=len(ap_shifts),
            flow_y=flow_y,
            flow_x=flow_x,
        )

    flow_y, flow_x = _optical_flow_tvl1(
        reference_gray,
        shifted_gray,
        num_warp=flow_warps,
        num_iter=flow_iterations,
        attachment=flow_attachment,
        tightness=flow_tightness,
    )
    flow_y, flow_x = regularize_flow(
        flow_y,
        flow_x,
        smoothing_sigma=flow_smoothing_sigma,
        max_pixels=flow_max_pixels,
    )
    flow_mean, flow_max = flow_stats(flow_y, flow_x)
    warped = warp_with_flow(shifted, flow_y, flow_x)
    return warped, RegistrationInfo(
        method=method,
        shift_y=shift_y,
        shift_x=shift_x,
        error=float(error),
        flow_mean_magnitude=flow_mean,
        flow_max_magnitude=flow_max,
        flow_y=flow_y,
        flow_x=flow_x,
    )
