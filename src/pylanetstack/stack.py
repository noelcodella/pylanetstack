"""Frame quality selection, registration, stacking, and output writing.

The stacker only deals with already-cropped image frames. This keeps the heavy
registration logic independent of whether the original capture was RAW stills,
rendered images, or video frames decoded during the crop step.
"""

from __future__ import annotations

import os
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from .image_math import cast_output, high_frequency_score, image_stats, preview_stretch
from .io import ensure_dir, list_image_files, read_image, write_json, write_tiff
from .postprocess import LevelValue, level_preview_image, postprocess_image, resolve_levels
from .registration import align_color_channels, register_frame, registered_frame_to_grid
from .types import FrameRecord, path_text


def resolve_worker_count(workers: int | None) -> int:
    """Resolve a user worker setting into a concrete thread count."""

    if workers is None:
        return 1
    workers = int(workers)
    if workers < 0:
        raise ValueError("workers must be >= 0")
    if workers == 0:
        return max(1, os.cpu_count() or 1)
    return workers


def _format_duration(seconds: float) -> str:
    seconds = max(float(seconds), 0.0)
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remaining = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes}m {remaining:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def _quality_score_one(index: int, total: int, path: Path) -> tuple[int, float]:
    print(f"[quality] {index}/{total} {path.name}", flush=True)
    image = read_image(path)
    return index - 1, high_frequency_score(image)


def compute_quality_scores(paths: list[Path], *, workers: int = 1) -> list[float]:
    """Read crops once and compute detail scores."""

    worker_count = resolve_worker_count(workers)
    scores: list[float] = [0.0 for _path in paths]
    if worker_count == 1 or len(paths) <= 1:
        for index, path in enumerate(paths, start=1):
            score_index, score = _quality_score_one(index, len(paths), path)
            scores[score_index] = score
        return scores

    print(f"[quality] using {worker_count} worker threads", flush=True)
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(_quality_score_one, index, len(paths), path)
            for index, path in enumerate(paths, start=1)
        ]
        for future in futures:
            score_index, score = future.result()
            scores[score_index] = score
    return scores


def quality_weights(
    scores: list[float],
    *,
    enabled: bool = True,
    scale: float = 1.0,
    min_weight: float = 0.25,
    max_weight: float = 4.0,
) -> list[float]:
    """Convert quality scores to clipped relative stack weights."""

    if not enabled:
        return [1.0 for _ in scores]
    if scale < 0:
        raise ValueError("quality weight scale must be >= 0")
    if min_weight <= 0 or max_weight <= 0 or min_weight > max_weight:
        raise ValueError("quality weight min/max must be positive and min <= max")
    if scale == 0:
        return [1.0 for _ in scores]
    mean_score = float(np.mean(scores)) if scores else 0.0
    if mean_score <= 0.0 or not np.isfinite(mean_score):
        return [1.0 for _ in scores]
    # Normalize by the dataset mean so the scale parameter controls contrast
    # between frames without changing the average exposure of the final stack.
    weights = (np.asarray(scores, dtype=np.float32) / mean_score) ** scale
    mean_weight = float(np.mean(weights))
    if mean_weight > 0 and np.isfinite(mean_weight):
        weights /= mean_weight
    weights = np.clip(weights, min_weight, max_weight)
    return [float(weight) for weight in weights]


def choose_reference_index(scores: list[float], strategy: str) -> int:
    if not scores:
        raise ValueError("Cannot choose a reference without scores")
    if strategy == "first":
        return 0
    if strategy == "best":
        return int(np.argmax(scores))
    raise ValueError("reference must be 'best' or 'first'")


def select_quality_indices(
    scores: list[float],
    *,
    keep_best_percent: float | None = None,
    keep_best_frames: int | None = None,
) -> list[int]:
    """Choose which frame indices survive lucky-imaging quality culling."""

    if not scores:
        return []
    if keep_best_percent is not None and not (0.0 < keep_best_percent <= 100.0):
        raise ValueError("keep_best_percent must be greater than 0 and at most 100")
    if keep_best_frames is not None and keep_best_frames <= 0:
        raise ValueError("keep_best_frames must be greater than 0")

    selected_count = len(scores)
    if keep_best_frames is not None:
        selected_count = min(selected_count, keep_best_frames)
    elif keep_best_percent is not None:
        selected_count = min(selected_count, max(1, int(np.ceil(len(scores) * keep_best_percent / 100.0))))

    ranked = np.argsort(np.asarray(scores, dtype=np.float32))
    selected = set(int(index) for index in ranked[-selected_count:])
    return [index for index in range(len(scores)) if index in selected]


def validate_stack_upsample(scale: float) -> float:
    scale = float(scale)
    if not np.isfinite(scale) or scale < 1.0:
        raise ValueError("stack_upsample must be a finite value >= 1")
    return scale


def _process_stack_frame(
    *,
    index: int,
    total: int,
    path: Path,
    score: float,
    weight: float,
    reference_image: np.ndarray,
    method: str,
    upsample_factor: int,
    flow_warps: int,
    flow_iterations: int,
    flow_attachment: float,
    flow_tightness: float,
    flow_smoothing_sigma: float,
    flow_max_pixels: float,
    ap_patch_size: int,
    ap_spacing: int,
    ap_blend_sigma: float,
    ap_max_points: int,
    ap_min_mask_fraction: float,
    ap_min_contrast: float,
    channel_align: bool,
    channel_reference: str,
    channel_align_max_shift: float,
    channel_align_upsample_factor: int,
    stack_upsample: float,
    registration_max_shift: float,
) -> tuple[int, np.ndarray | None, np.ndarray | None, float, FrameRecord, float, dict[str, Any] | None]:
    frame_started = perf_counter()
    print(f"[stack] {index}/{total} {path.name}", flush=True)
    image = read_image(path).astype(np.float32)
    if image.shape != reference_image.shape:
        raise ValueError(f"Shape mismatch for {path}: got {image.shape}, expected {reference_image.shape}")

    channel_shifts: dict[str, list[float]] | None = None
    if channel_align:
        print(f"[stack:channel] {path.name}", flush=True)
        image, channel_shifts = align_color_channels(
            image,
            reference_channel=channel_reference,
            max_shift_pixels=channel_align_max_shift,
            upsample_factor=channel_align_upsample_factor,
        )
    print(
        f"[stack:register] {path.name} method={method} shape={image.shape[0]}x{image.shape[1]}",
        flush=True,
    )
    warped, info = register_frame(
        image,
        reference_image,
        method=method,
        upsample_factor=upsample_factor,
        max_global_shift=registration_max_shift,
        flow_warps=flow_warps,
        flow_iterations=flow_iterations,
        flow_attachment=flow_attachment,
        flow_tightness=flow_tightness,
        flow_smoothing_sigma=flow_smoothing_sigma,
        flow_max_pixels=flow_max_pixels,
        ap_patch_size=ap_patch_size,
        ap_spacing=ap_spacing,
        ap_blend_sigma=ap_blend_sigma,
        ap_max_points=ap_max_points,
        ap_min_mask_fraction=ap_min_mask_fraction,
        ap_min_contrast=ap_min_contrast,
    )
    shift_magnitude = float(np.hypot(info.shift_y, info.shift_x))
    frame = FrameRecord(
        path=path_text(path),
        weight=float(weight),
        quality_score=float(score),
        shift_y=info.shift_y,
        shift_x=info.shift_x,
        registration_error=info.error,
        method=info.method,
        flow_mean_magnitude=info.flow_mean_magnitude,
        flow_max_magnitude=info.flow_max_magnitude,
        alignment_points=info.alignment_points,
        channel_shifts=channel_shifts,
    )
    if registration_max_shift > 0 and shift_magnitude > registration_max_shift:
        rejection = {
            "path": path_text(path),
            "reason": "registration_shift_too_large",
            "shift_y": float(info.shift_y),
            "shift_x": float(info.shift_x),
            "shift_magnitude": shift_magnitude,
            "registration_max_shift": float(registration_max_shift),
            "quality_score": float(score),
            "weight": float(weight),
        }
        print(
            f"[stack:reject] {path.name} shift={shift_magnitude:.1f}px "
            f"exceeds {registration_max_shift:.1f}px",
            flush=True,
        )
        return index - 1, None, None, 0.0, frame, perf_counter() - frame_started, rejection

    if stack_upsample > 1.0:
        stack_frame = registered_frame_to_grid(image, info, scale=stack_upsample)
    else:
        stack_frame = warped
    elapsed = perf_counter() - frame_started
    return index - 1, stack_frame, warped, float(weight), frame, elapsed, None


def stack_crops(
    input_path: Path | str,
    output_path: Path | str,
    *,
    recursive: bool = False,
    method: str = "ap",
    quality_weighting: bool = True,
    quality_weight_scale: float = 1.5,
    quality_weight_min: float = 0.25,
    quality_weight_max: float = 4.0,
    keep_best_percent: float | None = 25.0,
    keep_best_frames: int | None = None,
    reference: str = "best",
    upsample_factor: int = 20,
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
    channel_align: bool = True,
    channel_reference: str = "green",
    channel_align_max_shift: float = 3.0,
    channel_align_upsample_factor: int = 20,
    update_reference_every: int = 0,
    max_frames: int | None = None,
    stack_upsample: float = 1.0,
    output_dtype: str = "uint16",
    compression: str | None = None,
    preview: bool = True,
    preview_low_percentile: float = 0.5,
    preview_high_percentile: float = 99.95,
    preview_path: Path | str | None = None,
    postprocess: bool = True,
    postprocess_path: Path | str | None = None,
    post_input_max: float = 65535.0,
    post_level_black: LevelValue = "auto",
    post_level_white: LevelValue = "auto",
    post_level_gamma: float = 1.0,
    post_exposure: float = 0.5,
    post_highlight_compression: float = 0.7,
    post_texture: float = 100.0,
    post_texture_radius: float = 2.0,
    post_dehaze: float = 100.0,
    post_dehaze_radius: float = 12.0,
    post_sharpening: float = 500.0,
    post_sharpening_radius: float = 0.8,
    post_wb_mode: str = "auto",
    post_wb_strength: float = 1.0,
    post_red_balance: float = 1.0,
    post_green_balance: float = 1.0,
    post_blue_balance: float = 1.0,
    post_saturation: float = 3.0,
    post_vibrance: float = 1.0,
    post_background_black: bool = True,
    post_background_dilate: int = 3,
    post_background_feather: float = 4.0,
    post_upsample: float = 4.0,
    post_upsample_smoothing: float = 1.0,
    workers: int = 8,
    registration_max_shift: float = 50.0,
    metadata_path: Path | str | None = None,
) -> dict[str, Any]:
    """Stack cropped image frames with optional local nonlinear registration."""

    stack_upsample = validate_stack_upsample(stack_upsample)
    registration_max_shift = float(registration_max_shift)
    if registration_max_shift < 0:
        raise ValueError("registration_max_shift must be >= 0")

    paths = list_image_files(input_path, recursive=recursive)
    paths = [path for path in paths if not path.stem.endswith("_raw_mosaic")]
    if max_frames is not None:
        paths = paths[:max_frames]
    if not paths:
        raise FileNotFoundError(f"No stackable image files found in {input_path}")

    worker_count = resolve_worker_count(workers)
    scores = compute_quality_scores(paths, workers=worker_count)
    input_frame_count = len(paths)
    selected_indices = select_quality_indices(
        scores,
        keep_best_percent=keep_best_percent,
        keep_best_frames=keep_best_frames,
    )
    paths = [paths[index] for index in selected_indices]
    scores = [scores[index] for index in selected_indices]
    if not paths:
        raise ValueError("Quality selection removed every frame")

    weights = quality_weights(
        scores,
        enabled=quality_weighting,
        scale=quality_weight_scale,
        min_weight=quality_weight_min,
        max_weight=quality_weight_max,
    )
    best_frame_index = choose_reference_index(scores, "best")
    reference_index = choose_reference_index(scores, reference)
    reference_image = read_image(paths[reference_index]).astype(np.float32)
    if channel_align:
        reference_image, _reference_channel_shifts = align_color_channels(
            reference_image,
            reference_channel=channel_reference,
            max_shift_pixels=channel_align_max_shift,
            upsample_factor=channel_align_upsample_factor,
        )

    if stack_upsample > 1.0:
        # The accumulator is the only large long-lived array. Each frame is
        # registered, added, and then released so memory usage scales mostly
        # with crop size rather than frame count.
        stack_reference_shape = registered_frame_to_grid(
            reference_image,
            register_frame(reference_image, reference_image, method="none")[1],
            scale=stack_upsample,
        ).shape
        accumulator = np.zeros(stack_reference_shape, dtype=np.float64)
    else:
        accumulator = np.zeros_like(reference_image, dtype=np.float64)
    weight_sum = 0.0
    reference_accumulator = np.zeros_like(reference_image, dtype=np.float64) if update_reference_every > 0 else None
    reference_weight_sum = 0.0
    frame_records: list[FrameRecord | None] = [None for _path in paths]
    registration_rejections: list[dict[str, Any]] = []
    accepted_frame_indices: list[int] = []

    effective_stack_workers = worker_count
    if update_reference_every > 0 and effective_stack_workers > 1:
        print("[stack] update_reference_every requires serial stacking; using 1 worker", flush=True)
        effective_stack_workers = 1
    if effective_stack_workers > 1:
        print(f"[stack] using {effective_stack_workers} worker threads", flush=True)

    stack_started = perf_counter()
    completed = 0

    def consume_result(
        result: tuple[int, np.ndarray | None, np.ndarray | None, float, FrameRecord, float, dict[str, Any] | None]
    ) -> None:
        nonlocal completed, reference_weight_sum, reference_image, weight_sum
        frame_index, stack_frame, warped, weight, frame, elapsed, rejection = result
        if rejection is None and stack_frame is not None and warped is not None:
            accumulator[:] += stack_frame.astype(np.float64) * weight
            weight_sum += weight
            frame_records[frame_index] = frame
            accepted_frame_indices.append(frame_index)
            if reference_accumulator is not None:
                reference_accumulator[:] += warped.astype(np.float64) * weight
                reference_weight_sum += weight
        else:
            if rejection is not None:
                registration_rejections.append(rejection)
        completed += 1
        if (
            update_reference_every > 0
            and completed % update_reference_every == 0
            and reference_accumulator is not None
            and reference_weight_sum > 0
        ):
            reference_image = (reference_accumulator / reference_weight_sum).astype(np.float32)
        average = (perf_counter() - stack_started) / completed
        remaining = average * (len(paths) - completed)
        print(
            f"[stack:done] {completed}/{len(paths)} {Path(frame.path).name} "
            f"elapsed={_format_duration(elapsed)} eta={_format_duration(remaining)}",
            flush=True,
        )

    if effective_stack_workers == 1 or len(paths) <= 1:
        for index, (path, score, weight) in enumerate(zip(paths, scores, weights), start=1):
            result = _process_stack_frame(
                index=index,
                total=len(paths),
                path=path,
                score=score,
                weight=weight,
                reference_image=reference_image,
                method=method,
                upsample_factor=upsample_factor,
                flow_warps=flow_warps,
                flow_iterations=flow_iterations,
                flow_attachment=flow_attachment,
                flow_tightness=flow_tightness,
                flow_smoothing_sigma=flow_smoothing_sigma,
                flow_max_pixels=flow_max_pixels,
                ap_patch_size=ap_patch_size,
                ap_spacing=ap_spacing,
                ap_blend_sigma=ap_blend_sigma,
                ap_max_points=ap_max_points,
                ap_min_mask_fraction=ap_min_mask_fraction,
                ap_min_contrast=ap_min_contrast,
                channel_align=channel_align,
                channel_reference=channel_reference,
                channel_align_max_shift=channel_align_max_shift,
                channel_align_upsample_factor=channel_align_upsample_factor,
                stack_upsample=stack_upsample,
                registration_max_shift=registration_max_shift,
            )
            consume_result(result)
    else:
        # Fixed-reference registration can run in parallel because every worker
        # reads the same reference image. Reference-updating mode stays serial
        # because each batch depends on the accumulated result so far.
        next_index = 0
        pending: dict[Any, None] = {}
        with ThreadPoolExecutor(max_workers=effective_stack_workers) as executor:
            while next_index < len(paths) or pending:
                while next_index < len(paths) and len(pending) < effective_stack_workers:
                    future = executor.submit(
                        _process_stack_frame,
                        index=next_index + 1,
                        total=len(paths),
                        path=paths[next_index],
                        score=scores[next_index],
                        weight=weights[next_index],
                        reference_image=reference_image,
                        method=method,
                        upsample_factor=upsample_factor,
                        flow_warps=flow_warps,
                        flow_iterations=flow_iterations,
                        flow_attachment=flow_attachment,
                        flow_tightness=flow_tightness,
                        flow_smoothing_sigma=flow_smoothing_sigma,
                        flow_max_pixels=flow_max_pixels,
                        ap_patch_size=ap_patch_size,
                        ap_spacing=ap_spacing,
                        ap_blend_sigma=ap_blend_sigma,
                        ap_max_points=ap_max_points,
                        ap_min_mask_fraction=ap_min_mask_fraction,
                        ap_min_contrast=ap_min_contrast,
                        channel_align=channel_align,
                        channel_reference=channel_reference,
                        channel_align_max_shift=channel_align_max_shift,
                        channel_align_upsample_factor=channel_align_upsample_factor,
                        stack_upsample=stack_upsample,
                        registration_max_shift=registration_max_shift,
                    )
                    pending[future] = None
                    next_index += 1
                done, _not_done = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    pending.pop(future)
                    consume_result(future.result())

    frames = [frame for frame in frame_records if frame is not None]
    if weight_sum <= 0 or not frames:
        raise RuntimeError("Registration filtering rejected every selected frame")

    stacked = (accumulator / weight_sum).astype(np.float32)
    final_channel_shifts: dict[str, list[float]] | None = None
    if channel_align:
        print("[stack:channel-final] aligning final stack channels", flush=True)
        stacked, final_channel_shifts = align_color_channels(
            stacked,
            reference_channel=channel_reference,
            max_shift_pixels=channel_align_max_shift,
            upsample_factor=channel_align_upsample_factor,
        )

    out_path = Path(output_path).expanduser()
    ensure_dir(out_path.parent)
    write_tiff(out_path, cast_output(stacked, output_dtype), compression=compression)

    postprocess_kwargs: dict[str, Any] = {
        "input_max": post_input_max,
        "level_black": post_level_black,
        "level_white": post_level_white,
        "level_gamma": post_level_gamma,
        "exposure": post_exposure,
        "highlight_compression": post_highlight_compression,
        "texture": post_texture,
        "texture_radius": post_texture_radius,
        "dehaze": post_dehaze,
        "dehaze_radius": post_dehaze_radius,
        "sharpening": post_sharpening,
        "sharpening_radius": post_sharpening_radius,
        "white_balance": post_wb_mode,
        "white_balance_strength": post_wb_strength,
        "red_balance": post_red_balance,
        "green_balance": post_green_balance,
        "blue_balance": post_blue_balance,
        "saturation": post_saturation,
        "vibrance": post_vibrance,
        "background_black": post_background_black,
        "background_dilate": post_background_dilate,
        "background_feather": post_background_feather,
        "upsample": post_upsample,
        "upsample_smoothing": post_upsample_smoothing,
    }
    resolved_post_level_black, resolved_post_level_white = resolve_levels(
        stacked,
        input_max=post_input_max,
        black=post_level_black,
        white=post_level_white,
    )
    resolved_postprocess_kwargs = {
        **postprocess_kwargs,
        "level_black": resolved_post_level_black,
        "level_white": resolved_post_level_white,
    }

    preview_path_text: str | None = None
    if preview:
        preview_out_path = Path(preview_path).expanduser() if preview_path else out_path.with_name(
            f"{out_path.stem}_preview.tif"
        )
        write_tiff(
            preview_out_path,
            preview_stretch(
                stacked,
                low_percentile=preview_low_percentile,
                high_percentile=preview_high_percentile,
            ),
            compression=compression,
        )
        preview_path_text = path_text(preview_out_path)

    postprocess_path_text: str | None = None
    if postprocess:
        postprocess_out_path = Path(postprocess_path).expanduser() if postprocess_path else out_path.with_name(
            f"{out_path.stem}_processed.tif"
        )
        processed = postprocess_image(stacked, **resolved_postprocess_kwargs)
        write_tiff(postprocess_out_path, processed, compression=compression)
        postprocess_path_text = path_text(postprocess_out_path)

    best_frame_index = max(accepted_frame_indices, key=lambda frame_index: scores[frame_index])
    best_frame_source = paths[best_frame_index]
    best_frame = read_image(best_frame_source)
    # Save the exact crop that scored best, plus leveled and fully processed
    # versions, so users can compare stacking against the strongest single
    # frame without needing to hunt through the crop directory.
    best_frame_out_path = out_path.with_name(f"{out_path.stem}_best_frame.tif")
    write_tiff(best_frame_out_path, best_frame, compression=compression)
    best_frame_path_text = path_text(best_frame_out_path)

    best_frame_leveled_out_path = out_path.with_name(f"{out_path.stem}_best_frame_leveled.tif")
    best_frame_leveled = level_preview_image(
        best_frame,
        input_max=post_input_max,
        level_black=resolved_post_level_black,
        level_white=resolved_post_level_white,
        level_gamma=post_level_gamma,
    )
    write_tiff(best_frame_leveled_out_path, best_frame_leveled, compression=compression)
    best_frame_leveled_path_text = path_text(best_frame_leveled_out_path)

    best_frame_processed_path_text: str | None = None
    if postprocess:
        best_frame_processed_out_path = out_path.with_name(f"{out_path.stem}_best_frame_processed.tif")
        best_frame_processed = postprocess_image(best_frame, **resolved_postprocess_kwargs)
        write_tiff(best_frame_processed_out_path, best_frame_processed, compression=compression)
        best_frame_processed_path_text = path_text(best_frame_processed_out_path)

    metadata = {
        "input_path": path_text(Path(input_path)),
        "output_path": path_text(out_path),
        "preview_path": preview_path_text,
        "postprocess_path": postprocess_path_text,
        "best_frame_source_path": path_text(best_frame_source),
        "best_frame_path": best_frame_path_text,
        "best_frame_leveled_path": best_frame_leveled_path_text,
        "best_frame_processed_path": best_frame_processed_path_text,
        "best_frame_quality_score": float(scores[best_frame_index]),
        "best_frame_selected_index": best_frame_index,
        "frame_count": len(frames),
        "accepted_frame_count": len(frames),
        "input_frame_count": input_frame_count,
        "selected_frame_count": len(paths),
        "rejected_frame_count": input_frame_count - len(paths),
        "registration_rejected_frame_count": len(registration_rejections),
        "registration_rejections": registration_rejections,
        "workers": worker_count,
        "stack_workers": effective_stack_workers,
        "method": method,
        "quality_weighting": quality_weighting,
        "quality_weight_scale": quality_weight_scale,
        "quality_weight_min": quality_weight_min,
        "quality_weight_max": quality_weight_max,
        "keep_best_percent": keep_best_percent,
        "keep_best_frames": keep_best_frames,
        "reference": reference,
        "reference_path": path_text(paths[reference_index]),
        "upsample_factor": upsample_factor,
        "flow_warps": flow_warps,
        "flow_iterations": flow_iterations,
        "flow_attachment": flow_attachment,
        "flow_tightness": flow_tightness,
        "flow_smoothing_sigma": flow_smoothing_sigma,
        "flow_max_pixels": flow_max_pixels,
        "registration_max_shift": registration_max_shift,
        "ap_patch_size": ap_patch_size,
        "ap_spacing": ap_spacing,
        "ap_blend_sigma": ap_blend_sigma,
        "ap_max_points": ap_max_points,
        "ap_min_mask_fraction": ap_min_mask_fraction,
        "ap_min_contrast": ap_min_contrast,
        "channel_align": channel_align,
        "channel_reference": channel_reference,
        "channel_align_max_shift": channel_align_max_shift,
        "channel_align_upsample_factor": channel_align_upsample_factor,
        "final_channel_shifts": final_channel_shifts,
        "update_reference_every": update_reference_every,
        "stack_upsample": stack_upsample,
        "output_dtype": output_dtype,
        "preview_low_percentile": preview_low_percentile,
        "preview_high_percentile": preview_high_percentile,
        "postprocess": postprocess,
        "post_input_max": post_input_max,
        "post_level_black": post_level_black,
        "post_level_white": post_level_white,
        "post_level_black_resolved": resolved_post_level_black,
        "post_level_white_resolved": resolved_post_level_white,
        "post_level_gamma": post_level_gamma,
        "post_exposure": post_exposure,
        "post_highlight_compression": post_highlight_compression,
        "post_texture": post_texture,
        "post_texture_radius": post_texture_radius,
        "post_dehaze": post_dehaze,
        "post_dehaze_radius": post_dehaze_radius,
        "post_sharpening": post_sharpening,
        "post_sharpening_radius": post_sharpening_radius,
        "post_wb_mode": post_wb_mode,
        "post_wb_strength": post_wb_strength,
        "post_red_balance": post_red_balance,
        "post_green_balance": post_green_balance,
        "post_blue_balance": post_blue_balance,
        "post_saturation": post_saturation,
        "post_vibrance": post_vibrance,
        "post_background_black": post_background_black,
        "post_background_dilate": post_background_dilate,
        "post_background_feather": post_background_feather,
        "post_upsample": post_upsample,
        "post_upsample_smoothing": post_upsample_smoothing,
        "stack_stats": image_stats(stacked),
        "weight_sum": float(weight_sum),
        "frames": [frame.to_json() for frame in frames],
    }
    json_path = Path(metadata_path).expanduser() if metadata_path else out_path.with_suffix(".json")
    write_json(json_path, metadata)
    return metadata
