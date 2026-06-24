"""Display-oriented processing for stack and best-frame previews.

The scientific output remains the linear stack written by `stack.py`. Functions
here intentionally create a separate view/export image: levels, exposure,
contrast, sharpening, color balance, and display upsampling. Keeping this in one
module makes it clearer which operations are aesthetic/display choices rather
than changes to the underlying stacked data.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from .image_math import luminance, planet_mask

LevelValue = float | str


def _is_auto_level(value: LevelValue) -> bool:
    return isinstance(value, str) and value.strip().lower() == "auto"


def _level_float(value: LevelValue, *, name: str) -> float:
    try:
        level = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a number on a 0-255 scale or 'auto'") from error
    if not np.isfinite(level):
        raise ValueError(f"{name} must be finite")
    return level


def _amount(value: float) -> float:
    return float(np.clip(value, 0.0, 100.0) / 100.0)


def _uncapped_amount(value: float) -> float:
    return max(float(value), 0.0) / 100.0


def _as_display_float(image: np.ndarray, input_max: float) -> np.ndarray:
    values = np.asarray(image, dtype=np.float32)
    if input_max <= 0:
        input_max = float(np.max(values)) if float(np.max(values)) > 0 else 1.0
    return np.clip(values / input_max, 0.0, 1.0)


def _apply_luminance(image: np.ndarray, new_luminance: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return np.clip(new_luminance, 0.0, 1.0)
    current = luminance(image)
    scale = np.divide(
        new_luminance,
        current,
        out=np.zeros_like(new_luminance, dtype=np.float32),
        where=current > 1e-6,
    )
    return np.clip(image * scale[..., None], 0.0, 1.0)


def apply_levels(
    image: np.ndarray,
    *,
    black: float = 1.0,
    white: float = 10.0,
    gamma: float = 1.0,
) -> np.ndarray:
    """Apply Photoshop-style input levels on a 0-255 scale."""

    black_norm = black / 255.0
    white_norm = white / 255.0
    if white_norm <= black_norm:
        raise ValueError("post level white must be greater than post level black")
    gamma = max(float(gamma), 1e-6)
    adjusted = np.clip((image - black_norm) / (white_norm - black_norm), 0.0, 1.0)
    if gamma != 1.0:
        adjusted = adjusted ** (1.0 / gamma)
    return adjusted.astype(np.float32)


def resolve_levels(
    image: np.ndarray,
    *,
    input_max: float = 65535.0,
    black: LevelValue = 1.0,
    white: LevelValue = 10.0,
) -> tuple[float, float]:
    """Resolve numeric or automatic Photoshop-style input levels.

    Level values are expressed on a 0-255 scale after normalizing by
    ``input_max``. ``auto`` uses robust luminance percentiles so the same
    planetary and lunar recipes can adapt to different exposure ranges.
    """

    black_value = None if _is_auto_level(black) else _level_float(black, name="level black")
    white_value = None if _is_auto_level(white) else _level_float(white, name="level white")
    if black_value is None or white_value is None:
        source = _as_display_float(image, input_max)
        luma = luminance(source).astype(np.float32, copy=False) * 255.0
        values = luma.ravel()
        values = values[np.isfinite(values)]
        if values.size == 0:
            values = np.array([0.0, 255.0], dtype=np.float32)
        if black_value is None:
            # Keep the auto black point just below the measured dark pedestal
            # instead of exactly on it. RAW-derived linear crops often retain a
            # small black offset; clipping that offset can skew channel balance
            # in dim planetary detail after strong color processing.
            black_value = max(0.0, float(np.percentile(values, 0.1)) - 1.0)
        if white_value is None:
            white_value = _auto_white_level(luma, values, black_value)

    black_value = float(np.clip(black_value, 0.0, 254.999))
    white_value = float(np.clip(white_value, 0.001, 255.0))
    if white_value <= black_value:
        white_value = min(255.0, black_value + 1.0)
        if white_value <= black_value:
            black_value = max(0.0, white_value - 1.0)
    return black_value, white_value


def _auto_white_level(luma: np.ndarray, values: np.ndarray, black_value: float) -> float:
    """Pick a conservative automatic white point with highlight headroom."""

    target_values = values
    try:
        mask = planet_mask(luma)
    except ValueError:
        mask = np.zeros_like(luma, dtype=bool)
    mask_fraction = float(np.mean(mask)) if mask.size else 0.0
    if 0.0005 <= mask_fraction <= 0.98:
        masked = luma[mask]
        masked = masked[np.isfinite(masked)]
        if masked.size >= 64:
            target_values = masked

    target_high = float(np.percentile(target_values, 99.8))
    global_high = float(np.percentile(values, 99.8))
    high = max(target_high, global_high)
    span = max(high - black_value, 1e-3)

    # Dim planets have only a few display-level counts above background, while
    # compact bright planets need extra room for later texture/dehaze/sharpening
    # so their disk detail does not pin to white. Large lunar/solar surfaces
    # already fill much of the frame, so they keep a tighter white point.
    headroom = float(np.clip(1.25 + 6.0 / (span + 1.0), 1.35, 3.25))
    if span >= 20.0 and 0.0005 <= mask_fraction < 0.25:
        compactness = (0.25 - mask_fraction) / 0.25
        compact_headroom = 1.35 + 2.20 * float(np.clip(compactness, 0.0, 1.0))
        headroom = max(headroom, compact_headroom)
    return black_value + span * headroom


def level_preview_image(
    image: np.ndarray,
    *,
    input_max: float = 65535.0,
    level_black: LevelValue = 1.0,
    level_white: LevelValue = 10.0,
    level_gamma: float = 1.0,
) -> np.ndarray:
    """Return a level-adjusted 16-bit view without other postprocessing."""

    black, white = resolve_levels(
        image,
        input_max=input_max,
        black=level_black,
        white=level_white,
    )
    source = _as_display_float(image, input_max)
    leveled = apply_levels(source, black=black, white=white, gamma=level_gamma)
    return np.rint(np.clip(leveled, 0.0, 1.0) * np.iinfo(np.uint16).max).astype(np.uint16)


def apply_exposure(image: np.ndarray, *, stops: float = 1.0) -> np.ndarray:
    """Apply exposure in photographic stops."""

    return (image * (2.0 ** stops)).astype(np.float32)


def compress_highlights(image: np.ndarray, *, amount: float = 0.35) -> np.ndarray:
    """Apply a soft shoulder so aggressive levels/exposure do not hard-clip."""

    amount = max(float(amount), 0.0)
    if amount == 0:
        return np.clip(image, 0.0, 1.0).astype(np.float32)
    compressed = image / (1.0 + amount * np.maximum(image, 0.0))
    return np.clip(compressed, 0.0, 1.0).astype(np.float32)


def apply_texture(image: np.ndarray, *, amount: float = 100.0, radius: float = 2.0) -> np.ndarray:
    """Approximate Camera Raw Texture with mid-scale luminance contrast."""

    a = _amount(amount)
    if a == 0:
        return image.astype(np.float32, copy=False)
    luma = luminance(image)
    base = ndi.gaussian_filter(luma, sigma=max(radius, 0.1))
    detail = luma - base
    new_luma = np.clip(luma + detail * (0.9 * a), 0.0, 1.0)
    return _apply_luminance(image, new_luma).astype(np.float32)


def apply_dehaze(image: np.ndarray, *, amount: float = 100.0, radius: float = 12.0) -> np.ndarray:
    """Approximate Camera Raw Dehaze with broad local contrast and black recovery."""

    a = _amount(amount)
    if a == 0:
        return image.astype(np.float32, copy=False)
    luma = luminance(image)
    base = ndi.gaussian_filter(luma, sigma=max(radius, 0.1))
    broad_detail = luma - base
    new_luma = luma + broad_detail * (1.2 * a)
    new_luma = (new_luma - 0.5) * (1.0 + 0.35 * a) + 0.5
    haze_floor = 0.04 * a
    new_luma = (new_luma - haze_floor) / max(1.0 - haze_floor, 1e-6)
    return _apply_luminance(image, np.clip(new_luma, 0.0, 1.0)).astype(np.float32)


def apply_sharpening(image: np.ndarray, *, amount: float = 500.0, radius: float = 0.8) -> np.ndarray:
    """Apply luminance unsharp masking.

    Values well above Camera Raw's 100-style slider are allowed so users can
    intentionally oversharpen while searching for the best setting.
    """

    a = _uncapped_amount(amount)
    if a == 0:
        return image.astype(np.float32, copy=False)
    luma = luminance(image)
    blurred = ndi.gaussian_filter(luma, sigma=max(radius, 0.1))
    detail = luma - blurred
    new_luma = np.clip(luma + detail * (1.6 * a), 0.0, 1.0)
    return _apply_luminance(image, new_luma).astype(np.float32)


def apply_white_balance(
    image: np.ndarray,
    source: np.ndarray,
    *,
    mode: str = "auto",
    strength: float = 0.8,
    red: float = 1.0,
    green: float = 1.0,
    blue: float = 1.0,
) -> np.ndarray:
    """Apply manual RGB balance plus optional auto-neutralization on planet midtones.

    ``strength`` blends between no automatic correction at 0 and the measured
    neutral midtone correction at 1.
    """

    if image.ndim != 3 or image.shape[-1] < 3:
        return image.astype(np.float32, copy=False)
    if mode not in {"none", "auto"}:
        raise ValueError("post white balance mode must be 'none' or 'auto'")

    multipliers = np.array([red, green, blue], dtype=np.float32)
    if np.any(~np.isfinite(multipliers)) or np.any(multipliers <= 0):
        raise ValueError("post white balance RGB multipliers must be positive")

    if mode == "auto":
        mask = planet_mask(luminance(source))
        luma = luminance(image)
        selected_luma = luma[mask]
        if selected_luma.size > 0:
            low, high = np.percentile(selected_luma, [15.0, 85.0])
            midtone_mask = mask & (luma >= low) & (luma <= high)
        else:
            midtone_mask = mask
        samples = image[..., :3][midtone_mask]
        samples = samples[np.all((samples > 1e-5) & (samples < 0.98), axis=1)]
        if samples.size > 0:
            medians = np.median(samples, axis=0)
            target = float(np.mean(medians))
            auto = target / np.maximum(medians, 1e-6)
            auto = np.clip(auto, 0.5, 2.0).astype(np.float32)
            blend = float(np.clip(strength, 0.0, 1.0))
            multipliers *= 1.0 + blend * (auto - 1.0)

    balanced = np.clip(image[..., :3] * multipliers, 0.0, 1.0)
    if image.shape[-1] > 3:
        balanced = np.concatenate([balanced, image[..., 3:]], axis=-1)
    return balanced.astype(np.float32)


def apply_color_boost(
    image: np.ndarray,
    *,
    saturation: float = 3.0,
    vibrance: float = 1.0,
) -> np.ndarray:
    """Boost chroma while preserving luminance structure."""

    if image.ndim != 3 or image.shape[-1] < 3:
        return image.astype(np.float32, copy=False)
    saturation = max(float(saturation), 0.0)
    vibrance = max(float(vibrance), 0.0)
    if saturation == 1.0 and vibrance == 0.0:
        return image.astype(np.float32, copy=False)

    luma = luminance(image)
    chroma = image - luma[..., None]
    if vibrance > 0:
        channel_range = np.max(image[..., :3], axis=-1) - np.min(image[..., :3], axis=-1)
        current_saturation = np.divide(
            channel_range,
            np.max(image[..., :3], axis=-1),
            out=np.zeros_like(channel_range, dtype=np.float32),
            where=np.max(image[..., :3], axis=-1) > 1e-6,
        )
        factor = saturation * (1.0 + vibrance * np.clip(1.0 - current_saturation, 0.0, 1.0))
        boosted = luma[..., None] + chroma * factor[..., None]
    else:
        boosted = luma[..., None] + chroma * saturation
    return np.clip(boosted, 0.0, 1.0).astype(np.float32)


def suppress_background(
    image: np.ndarray,
    source: np.ndarray,
    *,
    enabled: bool = True,
    dilate: int = 3,
    feather: float = 4.0,
) -> np.ndarray:
    """Feather processed output to black outside the planet disk."""

    if not enabled:
        return image.astype(np.float32, copy=False)
    mask = planet_mask(luminance(source))
    if dilate > 0:
        mask = ndi.binary_dilation(mask, iterations=int(dilate))
    alpha = mask.astype(np.float32)
    if feather > 0:
        alpha = ndi.gaussian_filter(alpha, sigma=feather)
        max_alpha = float(np.max(alpha))
        if max_alpha > 0:
            alpha /= max_alpha
    if image.ndim == 2:
        return np.clip(image * alpha, 0.0, 1.0).astype(np.float32)
    return np.clip(image * alpha[..., None], 0.0, 1.0).astype(np.float32)


def _valid_upsample_scale(scale: float) -> float:
    scale = float(scale)
    if not np.isfinite(scale) or scale < 1.0:
        raise ValueError("post upsample must be a finite value >= 1")
    return scale


def upsample_display_image(
    image: np.ndarray,
    *,
    scale: float = 1.0,
    smoothing: float = 0.35,
) -> np.ndarray:
    """Enlarge a display image without changing tone or color processing."""

    scale = _valid_upsample_scale(scale)
    smoothing = float(smoothing)
    if not np.isfinite(smoothing) or smoothing < 0:
        raise ValueError("post upsample smoothing must be a finite value >= 0")
    if scale == 1.0:
        return image.astype(np.float32, copy=False)
    factors = [scale, scale]
    factors.extend(1.0 for _ in image.shape[2:])
    resized = ndi.zoom(image, zoom=tuple(factors), order=3, mode="reflect").astype(np.float32)
    if smoothing > 0:
        sigmas = [smoothing, smoothing]
        sigmas.extend(0.0 for _ in image.shape[2:])
        resized = ndi.gaussian_filter(resized, sigma=tuple(sigmas), mode="nearest")
    return np.clip(resized, 0.0, 1.0).astype(np.float32)


def postprocess_image(
    image: np.ndarray,
    *,
    input_max: float = 65535.0,
    level_black: LevelValue = "auto",
    level_white: LevelValue = "auto",
    level_gamma: float = 1.0,
    exposure: float = 0.5,
    highlight_compression: float = 0.7,
    texture: float = 100.0,
    texture_radius: float = 2.0,
    dehaze: float = 100.0,
    dehaze_radius: float = 12.0,
    sharpening: float = 500.0,
    sharpening_radius: float = 0.8,
    white_balance: str = "auto",
    white_balance_strength: float = 1.0,
    red_balance: float = 1.0,
    green_balance: float = 1.0,
    blue_balance: float = 1.0,
    saturation: float = 3.0,
    vibrance: float = 1.0,
    background_black: bool = True,
    background_dilate: int = 3,
    background_feather: float = 4.0,
    upsample: float = 4.0,
    upsample_smoothing: float = 1.0,
) -> np.ndarray:
    """Apply an adjustable planetary display-processing recipe."""

    upsample = _valid_upsample_scale(upsample)
    source = _as_display_float(image, input_max)
    resolved_black, resolved_white = resolve_levels(
        image,
        input_max=input_max,
        black=level_black,
        white=level_white,
    )
    processed = source.copy()
    # Tone mapping happens before detail/color operations, matching how a user
    # would first set usable levels in an image editor and then sharpen/color it.
    processed = apply_levels(
        processed,
        black=resolved_black,
        white=resolved_white,
        gamma=level_gamma,
    )
    processed = apply_exposure(processed, stops=exposure)
    processed = compress_highlights(processed, amount=highlight_compression)
    if upsample > 1.0:
        # Upsample only the display path. The scientific stack remains whatever
        # grid `stack_upsample` produced in the stacking stage.
        source = upsample_display_image(source, scale=upsample, smoothing=upsample_smoothing)
        processed = upsample_display_image(processed, scale=upsample, smoothing=upsample_smoothing)
    processed = apply_texture(processed, amount=texture, radius=texture_radius * upsample)
    processed = apply_dehaze(processed, amount=dehaze, radius=dehaze_radius * upsample)
    processed = apply_sharpening(processed, amount=sharpening, radius=sharpening_radius * upsample)
    processed = apply_white_balance(
        processed,
        source,
        mode=white_balance,
        strength=white_balance_strength,
        red=red_balance,
        green=green_balance,
        blue=blue_balance,
    )
    processed = apply_color_boost(processed, saturation=saturation, vibrance=vibrance)
    processed = suppress_background(
        processed,
        source,
        enabled=background_black,
        dilate=int(round(background_dilate * upsample)),
        feather=background_feather * upsample,
    )
    return np.rint(np.clip(processed, 0.0, 1.0) * np.iinfo(np.uint16).max).astype(np.uint16)
