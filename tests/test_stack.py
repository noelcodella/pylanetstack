from __future__ import annotations

import imageio.v3 as iio
import numpy as np
import tifffile
from scipy import ndimage as ndi

from pylanetstack.image_math import high_frequency_score, preview_stretch
from pylanetstack.io import read_image
from pylanetstack.postprocess import (
    apply_color_boost,
    apply_sharpening,
    apply_white_balance,
    level_preview_image,
    postprocess_image,
    resolve_levels,
    upsample_display_image,
)
from pylanetstack.registration import (
    align_color_channels,
    alignment_point_centers,
    register_frame,
    regularize_flow,
    warp_with_flow,
)
from pylanetstack.stack import quality_weights, select_quality_indices, stack_crops


def _planet_frame(shift_y: float = 0.0, shift_x: float = 0.0, blur: float = 0.0) -> np.ndarray:
    y, x = np.mgrid[:96, :96]
    disk = np.exp(-(((y - 48.0) ** 2) + ((x - 48.0) ** 2)) / (2 * 10.0**2))
    band = 0.18 * np.sin((y - 48.0) / 2.3) * disk
    spot = 0.35 * np.exp(-(((y - 44.0) ** 2) + ((x - 57.0) ** 2)) / (2 * 2.0**2))
    image = disk + band + spot
    if blur:
        image = ndi.gaussian_filter(image, blur)
    image = ndi.shift(image, (shift_y, shift_x), order=3, mode="nearest")
    rgb = np.stack([image * 0.8, image, image * 0.7], axis=-1)
    return (rgb * 20000).astype(np.float32)


def test_high_frequency_score_favors_sharp_frame() -> None:
    sharp = _planet_frame()
    soft = _planet_frame(blur=2.0)

    assert high_frequency_score(sharp) > high_frequency_score(soft)


def test_quality_weight_scale_controls_weight_spread() -> None:
    scores = [1.0, 2.0, 4.0]

    assert quality_weights(scores, scale=0.0) == [1.0, 1.0, 1.0]
    gentle = quality_weights(scores, scale=0.5, min_weight=0.01, max_weight=100.0)
    strong = quality_weights(scores, scale=2.0, min_weight=0.01, max_weight=100.0)

    assert strong[-1] / strong[0] > gentle[-1] / gentle[0]


def test_select_quality_indices_keeps_best_frames_in_original_order() -> None:
    scores = [0.4, 0.9, 0.1, 0.8, 0.3]

    assert select_quality_indices(scores, keep_best_frames=2) == [1, 3]
    assert select_quality_indices(scores, keep_best_percent=40) == [1, 3]
    assert select_quality_indices(scores, keep_best_percent=100) == [0, 1, 2, 3, 4]


def test_preview_stretch_makes_float_stack_viewable() -> None:
    image = np.linspace(500.0, 1700.0, 64, dtype=np.float32).reshape(8, 8)
    preview = preview_stretch(image)

    assert preview.dtype == np.uint16
    assert preview.min() == 0
    assert preview.max() == np.iinfo(np.uint16).max


def test_read_image_promotes_uint8_png_to_16bit_and_drops_alpha(tmp_path) -> None:
    rgba = np.zeros((8, 8, 4), dtype=np.uint8)
    rgba[..., 0] = 10
    rgba[..., 1] = 20
    rgba[..., 2] = 30
    rgba[..., 3] = 128
    path = tmp_path / "frame.png"
    iio.imwrite(path, rgba)

    image = read_image(path)

    assert image.shape == (8, 8, 3)
    assert image.dtype == np.uint16
    assert int(image[0, 0, 0]) == 10 * 257
    assert int(image[0, 0, 1]) == 20 * 257


def test_postprocess_image_writes_display_ready_uint16() -> None:
    image = _planet_frame() / 20000.0 * 1600.0
    processed = postprocess_image(
        image,
        level_black=1.0,
        level_white=10.0,
        exposure=1.0,
        texture=100.0,
        dehaze=100.0,
        sharpening=100.0,
    )

    assert processed.shape == image.shape
    assert processed.dtype == np.uint16
    assert int(processed.max()) > int(processed.min())


def test_auto_levels_resolve_from_image_percentiles() -> None:
    image = np.full((32, 32), 512.0, dtype=np.float32)
    image[8:24, 8:24] = np.linspace(1200.0, 8000.0, 16 * 16, dtype=np.float32).reshape(16, 16)

    black, white = resolve_levels(image, black="auto", white="auto")
    leveled = level_preview_image(image, level_black="auto", level_white="auto")

    assert 0.5 < black < 1.5
    assert 45.0 < white < 60.0
    assert leveled.dtype == np.uint16
    assert int(leveled.max()) > int(leveled.min())


def test_auto_levels_leave_headroom_for_dim_planets() -> None:
    y, x = np.mgrid[:250, :250]
    disk = np.exp(-(((y - 125.0) ** 2) + ((x - 125.0) ** 2)) / (2 * 16.0**2))
    image = np.full((250, 250, 3), 512.0, dtype=np.float32)
    image[..., 0] += disk * 980.0
    image[..., 1] += disk * 760.0
    image[..., 2] += disk * 620.0

    black, white = resolve_levels(image, black="auto", white="auto")

    assert 0.5 < black < 1.5
    assert 8.0 < white < 12.0


def test_auto_levels_use_less_headroom_for_bright_surfaces() -> None:
    image = np.full((128, 128, 3), 512.0, dtype=np.float32)
    y, x = np.mgrid[:128, :128]
    surface = (x > 24) & (x < 112) & (y > 16) & (y < 112)
    gradient = 3200.0 + 5200.0 * ((x + y) / float(x.max() + y.max()))
    image[surface] = gradient[surface, None]

    black, white = resolve_levels(image, black="auto", white="auto")

    assert 0.5 < black < 1.5
    assert 35.0 < white < 55.0


def test_auto_levels_keep_compact_bright_planets_conservative() -> None:
    y, x = np.mgrid[:250, :250]
    disk = np.exp(-(((y - 125.0) ** 2) + ((x - 125.0) ** 2)) / (2 * 18.0**2))
    image = np.full((250, 250, 3), 32.0, dtype=np.float32)
    image[..., 0] += disk * 27000.0
    image[..., 1] += disk * 23000.0
    image[..., 2] += disk * 18000.0

    black, white = resolve_levels(image, black="auto", white="auto")

    assert 0.0 <= black < 1.0
    assert 235.0 < white <= 255.0


def test_postprocess_image_accepts_auto_levels() -> None:
    image = _planet_frame() / 20000.0 * 1600.0

    processed = postprocess_image(
        image,
        level_black="auto",
        level_white="auto",
        exposure=0.0,
        texture=0.0,
        dehaze=0.0,
        sharpening=0.0,
        background_black=False,
    )

    assert processed.shape == image.shape
    assert processed.dtype == np.uint16
    assert int(processed.max()) > int(processed.min())


def test_postprocess_image_can_upsample_display_output() -> None:
    image = _planet_frame() / 20000.0 * 1600.0

    processed = postprocess_image(image, upsample=2.0)

    assert processed.shape == (192, 192, 3)
    assert processed.dtype == np.uint16


def test_upsample_smoothing_reduces_pixel_steps() -> None:
    image = np.zeros((16, 16), dtype=np.float32)
    image[:, 8:] = 1.0

    crisp = upsample_display_image(image, scale=4.0, smoothing=0.0)
    smooth = upsample_display_image(image, scale=4.0, smoothing=0.8)

    assert float(np.max(np.abs(np.diff(smooth, axis=1)))) < float(np.max(np.abs(np.diff(crisp, axis=1))))


def test_color_boost_increases_weak_chroma() -> None:
    image = _planet_frame() / 20000.0
    muted = image * np.array([1.0, 0.99, 0.98], dtype=np.float32)

    boosted = apply_color_boost(muted, saturation=3.0, vibrance=1.0)

    luma_before = np.mean(muted, axis=-1, keepdims=True)
    luma_after = np.mean(boosted, axis=-1, keepdims=True)
    chroma_before = float(np.mean(np.abs(muted - luma_before)))
    chroma_after = float(np.mean(np.abs(boosted - luma_after)))
    assert chroma_after > chroma_before * 2.0


def test_sharpening_allows_values_above_100() -> None:
    image = (_planet_frame() / 20000.0).astype(np.float32)

    sharpened_100 = apply_sharpening(image, amount=100.0)
    sharpened_300 = apply_sharpening(image, amount=300.0)

    residual_100 = float(np.mean(np.abs(sharpened_100 - image)))
    residual_300 = float(np.mean(np.abs(sharpened_300 - image)))
    assert residual_300 > residual_100 * 1.5


def test_auto_white_balance_reduces_green_cast() -> None:
    image = (_planet_frame() / 20000.0).astype(np.float32)
    green_cast = np.clip(image * np.array([0.9, 1.2, 0.85], dtype=np.float32), 0.0, 1.0)

    balanced = apply_white_balance(green_cast, image, mode="auto", strength=1.0)

    before = green_cast[..., :3].reshape(-1, 3).mean(axis=0)
    after = balanced[..., :3].reshape(-1, 3).mean(axis=0)
    before_spread = float(np.max(before) - np.min(before))
    after_spread = float(np.max(after) - np.min(after))
    assert after_spread < before_spread * 0.5


def test_regularize_flow_clips_vector_magnitude() -> None:
    flow_y = np.full((8, 8), 5.0, dtype=np.float32)
    flow_x = np.full((8, 8), 5.0, dtype=np.float32)

    clipped_y, clipped_x = regularize_flow(flow_y, flow_x, smoothing_sigma=0.0, max_pixels=2.0)

    assert float(np.max(np.hypot(clipped_y, clipped_x))) <= 2.01


def test_register_frame_shift_limit_skips_expensive_local_alignment() -> None:
    reference = _planet_frame()
    moving = _planet_frame(shift_y=24.0, shift_x=-22.0)

    registered, info = register_frame(
        moving,
        reference,
        method="ap",
        max_global_shift=10.0,
        ap_patch_size=32,
        ap_spacing=16,
        ap_min_contrast=0.001,
    )

    assert np.array_equal(registered, moving)
    assert float(np.hypot(info.shift_y, info.shift_x)) > 10.0
    assert info.alignment_points is None
    assert info.flow_mean_magnitude is None
    assert info.flow_max_magnitude is None


def test_alignment_point_centers_find_planet_grid() -> None:
    frame = _planet_frame()[..., 1]
    centers = alignment_point_centers(frame, patch_size=24, spacing=12, min_contrast=0.001)

    assert len(centers) > 4


def test_alignment_point_centers_adapts_to_max_points() -> None:
    frame = np.tile(_planet_frame()[..., 1], (6, 6))
    centers = alignment_point_centers(
        frame,
        patch_size=12,
        spacing=6,
        max_points=50,
        min_mask_fraction=0.0,
        min_contrast=0.0,
    )

    assert 1 < len(centers) <= 50


def test_align_color_channels_reduces_synthetic_dispersion() -> None:
    reference = _planet_frame()[..., 1]
    image = np.stack(
        [
            ndi.shift(reference, (1.4, -0.8), order=3, mode="nearest"),
            reference,
            ndi.shift(reference, (-1.1, 1.3), order=3, mode="nearest"),
        ],
        axis=-1,
    )
    before = float(np.mean((image[..., 2] - image[..., 1]) ** 2))

    aligned, shifts = align_color_channels(
        image,
        reference_channel="green",
        max_shift_pixels=3.0,
        upsample_factor=20,
    )
    after = float(np.mean((aligned[..., 2] - aligned[..., 1]) ** 2))

    assert after < before * 0.4
    assert "red" in shifts
    assert "blue" in shifts


def test_rigid_registration_reduces_error() -> None:
    reference = _planet_frame()
    moving = _planet_frame(shift_y=2.4, shift_x=-1.7)
    before = float(np.mean((reference - moving) ** 2))

    registered, info = register_frame(moving, reference, method="rigid", upsample_factor=20)
    after = float(np.mean((reference - registered) ** 2))

    assert after < before * 0.35
    assert abs(info.shift_y + 2.4) < 0.4
    assert abs(info.shift_x - 1.7) < 0.4


def test_ap_registration_improves_local_warp_more_than_rigid() -> None:
    reference = _planet_frame()
    height, width = reference.shape[:2]
    rows, cols = np.meshgrid(
        np.arange(height, dtype=np.float32),
        np.arange(width, dtype=np.float32),
        indexing="ij",
    )
    flow_y = 1.0 * np.sin((cols - width / 2) / 18.0).astype(np.float32)
    flow_x = 1.0 * np.sin((rows - height / 2) / 20.0).astype(np.float32)
    moving = warp_with_flow(reference, flow_y, flow_x)

    rigid, _rigid_info = register_frame(moving, reference, method="rigid", upsample_factor=10)
    ap_registered, ap_info = register_frame(
        moving,
        reference,
        method="ap",
        upsample_factor=10,
        flow_max_pixels=2.0,
        flow_smoothing_sigma=0.7,
        ap_patch_size=32,
        ap_spacing=16,
        ap_blend_sigma=14.0,
        ap_min_contrast=0.001,
    )

    rigid_error = float(np.mean((reference - rigid) ** 2))
    ap_error = float(np.mean((reference - ap_registered) ** 2))
    assert ap_info.alignment_points is not None
    assert ap_info.alignment_points > 4
    assert ap_error < rigid_error


def test_stack_crops_writes_output(tmp_path) -> None:
    input_dir = tmp_path / "crops"
    input_dir.mkdir()
    for index, shift in enumerate([(0.0, 0.0), (1.2, -0.5), (-0.7, 0.8)], start=1):
        tifffile.imwrite(input_dir / f"frame_{index:03d}.tif", _planet_frame(*shift).astype(np.uint16))

    output_path = tmp_path / "stack.tif"
    metadata = stack_crops(
        input_dir,
        output_path,
        method="ap",
        quality_weighting=True,
        quality_weight_scale=1.5,
        keep_best_frames=2,
        channel_align=True,
        post_upsample=2.0,
        post_upsample_smoothing=0.5,
        ap_patch_size=32,
        ap_spacing=16,
        ap_min_contrast=0.001,
        upsample_factor=10,
        workers=2,
    )

    stacked = tifffile.imread(output_path)
    preview = tifffile.imread(tmp_path / "stack_preview.tif")
    processed = tifffile.imread(tmp_path / "stack_processed.tif")
    best_frame = tifffile.imread(tmp_path / "stack_best_frame.tif")
    best_frame_leveled = tifffile.imread(tmp_path / "stack_best_frame_leveled.tif")
    best_frame_processed = tifffile.imread(tmp_path / "stack_best_frame_processed.tif")
    assert output_path.exists()
    assert stacked.shape == (96, 96, 3)
    assert processed.shape == (192, 192, 3)
    assert best_frame.shape == (96, 96, 3)
    assert best_frame_leveled.shape == (96, 96, 3)
    assert best_frame_processed.shape == (192, 192, 3)
    assert stacked.dtype == np.uint16
    assert preview.dtype == np.uint16
    assert processed.dtype == np.uint16
    assert best_frame.dtype == np.uint16
    assert best_frame_leveled.dtype == np.uint16
    assert best_frame_processed.dtype == np.uint16
    assert int(best_frame_leveled.max()) > int(best_frame_leveled.min())
    assert metadata["input_frame_count"] == 3
    assert metadata["frame_count"] == 2
    assert metadata["accepted_frame_count"] == 2
    assert metadata["selected_frame_count"] == 2
    assert metadata["rejected_frame_count"] == 1
    assert metadata["registration_rejected_frame_count"] == 0
    assert metadata["registration_rejections"] == []
    assert metadata["registration_max_shift"] == 50.0
    assert metadata["workers"] == 2
    assert metadata["stack_workers"] == 2
    assert metadata["output_dtype"] == "uint16"
    assert metadata["quality_weight_scale"] == 1.5
    assert metadata["keep_best_frames"] == 2
    assert metadata["ap_patch_size"] == 32
    assert metadata["ap_max_points"] == 2500
    assert metadata["frames"][0]["alignment_points"] is not None
    assert metadata["channel_align"] is True
    assert metadata["frames"][0]["channel_shifts"] is not None
    assert metadata["final_channel_shifts"] is not None
    assert "red" in metadata["final_channel_shifts"]
    assert metadata["preview_path"].endswith("stack_preview.tif")
    assert metadata["postprocess_path"].endswith("stack_processed.tif")
    assert metadata["best_frame_path"].endswith("stack_best_frame.tif")
    assert metadata["best_frame_leveled_path"].endswith("stack_best_frame_leveled.tif")
    assert metadata["best_frame_processed_path"].endswith("stack_best_frame_processed.tif")
    assert metadata["best_frame_source_path"].endswith(".tif")
    assert metadata["best_frame_quality_score"] > 0
    assert metadata["post_level_black"] == 1.0
    assert metadata["post_level_white"] == 10.0
    assert metadata["post_level_black_resolved"] == 1.0
    assert metadata["post_level_white_resolved"] == 10.0
    assert metadata["post_exposure"] == 0.5
    assert metadata["post_sharpening"] == 500.0
    assert metadata["post_wb_mode"] == "auto"
    assert metadata["post_wb_strength"] == 0.8
    assert metadata["post_saturation"] == 3.0
    assert metadata["post_vibrance"] == 1.0
    assert metadata["post_upsample"] == 2.0
    assert metadata["post_upsample_smoothing"] == 0.5
    assert metadata["stack_stats"]["max"] > metadata["stack_stats"]["min"]


def test_stack_crops_rejects_registration_shift_outliers(tmp_path) -> None:
    input_dir = tmp_path / "crops"
    input_dir.mkdir()
    tifffile.imwrite(input_dir / "a_frame_reference.tif", _planet_frame().astype(np.uint16))
    tifffile.imwrite(input_dir / "z_frame_outlier.tif", _planet_frame(shift_y=24.0, shift_x=-22.0).astype(np.uint16))

    output_path = tmp_path / "stack.tif"
    metadata = stack_crops(
        input_dir,
        output_path,
        method="rigid",
        quality_weighting=False,
        keep_best_percent=100,
        reference="first",
        channel_align=False,
        registration_max_shift=10.0,
        preview=False,
        postprocess=False,
        upsample_factor=10,
    )

    assert output_path.exists()
    assert metadata["input_frame_count"] == 2
    assert metadata["selected_frame_count"] == 2
    assert metadata["frame_count"] == 1
    assert metadata["accepted_frame_count"] == 1
    assert metadata["registration_rejected_frame_count"] == 1
    assert len(metadata["frames"]) == 1
    rejection = metadata["registration_rejections"][0]
    assert rejection["path"].endswith("z_frame_outlier.tif")
    assert rejection["reason"] == "registration_shift_too_large"
    assert rejection["shift_magnitude"] > 10.0
    assert rejection["registration_max_shift"] == 10.0
    assert metadata["best_frame_source_path"].endswith("a_frame_reference.tif")


def test_stack_crops_can_upsample_scientific_stack(tmp_path) -> None:
    input_dir = tmp_path / "crops"
    input_dir.mkdir()
    for index, shift in enumerate([(0.0, 0.0), (0.6, -0.4), (-0.5, 0.5)], start=1):
        tifffile.imwrite(input_dir / f"frame_{index:03d}.tif", _planet_frame(*shift).astype(np.uint16))

    output_path = tmp_path / "stack.tif"
    metadata = stack_crops(
        input_dir,
        output_path,
        method="rigid",
        quality_weighting=False,
        keep_best_frames=3,
        channel_align=False,
        stack_upsample=2.0,
        preview=False,
        postprocess=False,
        upsample_factor=20,
    )

    stacked = tifffile.imread(output_path)
    best_frame = tifffile.imread(tmp_path / "stack_best_frame.tif")
    best_frame_leveled = tifffile.imread(tmp_path / "stack_best_frame_leveled.tif")
    assert stacked.shape == (192, 192, 3)
    assert best_frame.shape == (96, 96, 3)
    assert best_frame_leveled.shape == (96, 96, 3)
    assert stacked.dtype == np.uint16
    assert metadata["stack_upsample"] == 2.0
    assert metadata["final_channel_shifts"] is None
    assert metadata["best_frame_path"].endswith("stack_best_frame.tif")
    assert metadata["best_frame_leveled_path"].endswith("stack_best_frame_leveled.tif")
    assert metadata["best_frame_processed_path"] is None


def test_stack_crops_best_frame_comes_from_selected_frames(tmp_path) -> None:
    input_dir = tmp_path / "crops"
    input_dir.mkdir()
    sharp = _planet_frame().astype(np.uint16)
    soft = _planet_frame(blur=2.5).astype(np.uint16)
    tifffile.imwrite(input_dir / "frame_sharp.tif", sharp)
    tifffile.imwrite(input_dir / "frame_soft.tif", soft)

    output_path = tmp_path / "stack.tif"
    metadata = stack_crops(
        input_dir,
        output_path,
        method="none",
        quality_weighting=False,
        keep_best_frames=1,
        channel_align=False,
        preview=False,
        postprocess=True,
        post_level_black="auto",
        post_level_white="auto",
    )

    best_frame = tifffile.imread(tmp_path / "stack_best_frame.tif")
    best_frame_leveled = tifffile.imread(tmp_path / "stack_best_frame_leveled.tif")
    best_frame_processed = tifffile.imread(tmp_path / "stack_best_frame_processed.tif")
    assert metadata["best_frame_source_path"].endswith("frame_sharp.tif")
    assert metadata["frame_count"] == 1
    assert metadata["post_level_black"] == "auto"
    assert metadata["post_level_white"] == "auto"
    assert isinstance(metadata["post_level_black_resolved"], float)
    assert isinstance(metadata["post_level_white_resolved"], float)
    assert metadata["post_level_white_resolved"] > metadata["post_level_black_resolved"]
    assert np.array_equal(best_frame, sharp)
    assert best_frame_leveled.shape == sharp.shape
    assert best_frame_leveled.dtype == np.uint16
    assert best_frame_processed.shape == sharp.shape
    assert best_frame_processed.dtype == np.uint16


def test_stack_crops_reads_png_inputs(tmp_path) -> None:
    input_dir = tmp_path / "pngs"
    input_dir.mkdir()
    for index, shift in enumerate([(0.0, 0.0), (0.8, -0.4), (-0.5, 0.7)], start=1):
        frame = np.clip(_planet_frame(*shift) / 20000.0 * 255.0, 0, 255).astype(np.uint8)
        iio.imwrite(input_dir / f"frame_{index:03d}.png", frame)

    output_path = tmp_path / "stack.tif"
    metadata = stack_crops(
        input_dir,
        output_path,
        method="rigid",
        quality_weighting=True,
        keep_best_frames=2,
        channel_align=False,
        preview=False,
        postprocess=False,
        upsample_factor=10,
    )

    stacked = tifffile.imread(output_path)
    assert output_path.exists()
    assert stacked.shape == (96, 96, 3)
    assert stacked.dtype == np.uint16
    assert int(stacked.max()) > 255
    assert metadata["input_frame_count"] == 3
    assert metadata["frame_count"] == 2
