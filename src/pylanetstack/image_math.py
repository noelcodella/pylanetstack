"""Small numerical helpers shared by crop, stack, and postprocessing code.

These functions avoid changing the scientific image values unless their name
explicitly says they are producing a preview/display image. Registration and
quality scoring use robust normalized copies so the final stack can still be an
average of the original linear crop intensities.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


RGB_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def luminance(image: np.ndarray) -> np.ndarray:
    """Return a float32 luminance image from grayscale or RGB-like data."""

    array = np.asarray(image)
    if array.ndim == 2:
        return array.astype(np.float32, copy=False)
    if array.ndim == 3 and array.shape[2] >= 3:
        rgb = array[..., :3].astype(np.float32, copy=False)
        return np.tensordot(rgb, RGB_WEIGHTS, axes=([-1], [0])).astype(np.float32)
    raise ValueError(f"Expected a 2-D or RGB image, got shape {array.shape}")


def robust_sigma(values: np.ndarray) -> float:
    """Estimate standard deviation using median absolute deviation."""

    flat = np.asarray(values, dtype=np.float32).ravel()
    median = float(np.median(flat))
    mad = float(np.median(np.abs(flat - median)))
    return max(1.4826 * mad, 1e-6)


def robust_normalize(image: np.ndarray, *, low: float = 1.0, high: float = 99.8) -> np.ndarray:
    """Normalize for registration without changing stack pixel values."""

    gray = np.asarray(image, dtype=np.float32)
    # Percentile normalization is much less sensitive to one hot pixel or a
    # saturated specular highlight than min/max normalization.
    lo, hi = np.percentile(gray, [low, high])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(gray))
        hi = float(np.max(gray))
    if hi <= lo:
        return np.zeros_like(gray, dtype=np.float32)
    normalized = (gray - lo) / (hi - lo)
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


def planet_mask(gray: np.ndarray) -> np.ndarray:
    """Build a conservative mask around the bright planetary disk."""

    normalized = robust_normalize(gray)
    blurred = ndi.gaussian_filter(normalized, sigma=1.0)
    median = float(np.median(blurred))
    sigma = robust_sigma(blurred)
    threshold = max(median + 2.5 * sigma, float(np.percentile(blurred, 90.0)))
    mask = blurred > threshold

    labels, count = ndi.label(mask)
    if count == 0:
        return np.ones_like(normalized, dtype=bool)

    # Use only the largest connected bright component. That keeps hot pixels,
    # moons, and edge artifacts from dominating frame quality estimates.
    areas = ndi.sum(mask, labels, index=np.arange(1, count + 1))
    label = int(np.argmax(areas)) + 1
    mask = labels == label
    mask = ndi.binary_dilation(mask, iterations=2)
    return mask


def high_frequency_score(image: np.ndarray) -> float:
    """Score detail with RMS Laplacian energy inside the planet mask."""

    gray = luminance(image)
    normalized = robust_normalize(gray)
    mask = planet_mask(normalized)
    # A Laplacian responds strongly to crisp edges and fine texture. The score
    # is not a beauty metric; it is a simple lucky-imaging proxy for seeing.
    laplacian = ndi.laplace(normalized)
    selected = laplacian[mask]
    if selected.size == 0:
        selected = laplacian.ravel()
    return float(np.sqrt(np.mean(selected * selected)) + 1e-12)


def cast_output(image: np.ndarray, dtype: str) -> np.ndarray:
    """Cast stack output without applying a contrast stretch."""

    if dtype == "float32":
        return image.astype(np.float32, copy=False)
    if dtype == "uint16":
        clipped = np.clip(np.rint(image), 0, np.iinfo(np.uint16).max)
        return clipped.astype(np.uint16)
    raise ValueError(f"Unsupported output dtype: {dtype}")


def preview_stretch(
    image: np.ndarray,
    *,
    low_percentile: float = 0.5,
    high_percentile: float = 99.95,
) -> np.ndarray:
    """Return a display-friendly 16-bit preview without changing science data."""

    values = np.asarray(image, dtype=np.float32)
    exposure_values = luminance(values) if values.ndim == 3 and values.shape[-1] >= 3 else values
    finite = exposure_values[np.isfinite(exposure_values)]
    if finite.size == 0:
        return np.zeros_like(values, dtype=np.uint16)

    low, high = np.percentile(finite, [low_percentile, high_percentile])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low = float(np.min(finite))
        high = float(np.max(finite))
    if high <= low:
        return np.zeros_like(values, dtype=np.uint16)

    stretched = (values - low) / (high - low)
    stretched = np.clip(stretched, 0.0, 1.0)
    return np.rint(stretched * np.iinfo(np.uint16).max).astype(np.uint16)


def image_stats(image: np.ndarray) -> dict[str, float]:
    """Summarize image intensities for metadata and quick diagnostics."""

    values = np.asarray(image, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "p01": 0.0,
            "p50": 0.0,
            "p99_8": 0.0,
        }
    p01, p50, p998 = np.percentile(finite, [0.1, 50.0, 99.8])
    return {
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "p01": float(p01),
        "p50": float(p50),
        "p99_8": float(p998),
    }
