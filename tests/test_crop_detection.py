from __future__ import annotations

import json

import imageio.v3 as iio
import numpy as np
import tifffile

from pylanetstack.crop import (
    apply_white_balance,
    auto_crop_geometry,
    crop_arw_directory,
    crop_bounds,
    crop_one_image,
    crop_with_padding,
    demosaic_bilinear,
    find_planet_center,
)
from pylanetstack.types import CropRecord


def test_find_planet_center_on_synthetic_mosaic() -> None:
    y, x = np.mgrid[:512, :768]
    center_y = 201.4
    center_x = 339.7
    planet = 1200.0 * np.exp(-(((y - center_y) ** 2) + ((x - center_x) ** 2)) / (2 * 18.0**2))
    background = np.full_like(planet, 512.0)
    noise = np.random.default_rng(42).normal(0, 4, size=planet.shape)
    mosaic = np.clip(background + planet + noise, 0, 65535).astype(np.uint16)

    found_y, found_x = find_planet_center(mosaic, crop_size=128, downsample=4)

    assert abs(found_y - center_y) < 1.0
    assert abs(found_x - center_x) < 1.0


def test_crop_with_padding_keeps_requested_shape() -> None:
    image = np.arange(100, dtype=np.uint16).reshape(10, 10)
    crop = crop_with_padding(image, y0=-2, y1=6, x0=7, x1=15)

    assert crop.shape == (8, 8)
    assert crop[0, 0] == 0
    assert crop[2, 0] == image[0, 7]


def test_crop_bounds_for_even_size() -> None:
    assert crop_bounds(100.2, 200.7, 50) == (75, 125, 176, 226)


def test_auto_crop_keeps_small_planet_at_minimum_size() -> None:
    y, x = np.mgrid[:512, :768]
    center_y = 201.4
    center_x = 339.7
    planet = 1200.0 * np.exp(-(((y - center_y) ** 2) + ((x - center_x) ** 2)) / (2 * 18.0**2))
    image = np.clip(512.0 + planet, 0, 65535).astype(np.uint16)

    found_y, found_x, crop_size = auto_crop_geometry(image, min_size=250)

    assert 250 <= crop_size <= 320
    assert abs(found_y - center_y) < 1.0
    assert abs(found_x - center_x) < 1.0


def test_auto_crop_expands_for_large_disk() -> None:
    y, x = np.mgrid[:600, :800]
    center_y = 300.0
    center_x = 410.0
    radius = 170.0
    disk = (((y - center_y) ** 2) + ((x - center_x) ** 2)) <= radius**2
    image = np.where(disk, 30000, 500).astype(np.uint16)

    found_y, found_x, crop_size = auto_crop_geometry(
        image,
        margin=1.25,
        min_size=250,
    )

    assert crop_size > 400
    assert crop_size < 560
    assert abs(found_y - center_y) < 8.0
    assert abs(found_x - center_x) < 8.0


def test_demosaic_bilinear_shape_and_dtype() -> None:
    mosaic = np.arange(64, dtype=np.uint16).reshape(8, 8)
    color_ids = np.indices((8, 8)).sum(axis=0) % 2
    color_ids = np.where(color_ids == 0, 0, 1).astype(np.int16)

    rgb = demosaic_bilinear(mosaic, color_ids, "RGGB")

    assert rgb.shape == (8, 8, 3)
    assert rgb.dtype == np.uint16


def test_apply_white_balance_scales_signal_above_black() -> None:
    image = np.array([[[612, 612, 612]]], dtype=np.uint16)
    balanced = apply_white_balance(image, multipliers=[2.0, 1.0, 1.5], black_level=512.0)

    assert balanced.tolist() == [[[712, 612, 662]]]


def test_crop_one_image_reads_png_and_writes_centered_tiff_crop(tmp_path) -> None:
    y, x = np.mgrid[:128, :160]
    center_y = 70.3
    center_x = 83.8
    planet = np.exp(-(((y - center_y) ** 2) + ((x - center_x) ** 2)) / (2 * 10.0**2))
    rgb = np.stack([planet * 0.9, planet, planet * 0.75], axis=-1)
    image = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    source = tmp_path / "jupiter.png"
    output_dir = tmp_path / "crops"
    iio.imwrite(source, image)

    record = crop_one_image(source, output_dir, crop_size=48, overwrite=True)

    crop = tifffile.imread(record.crop_path)
    assert crop.shape == (48, 48, 3)
    assert crop.dtype == np.uint16
    assert int(crop.max()) > 255
    assert 20.0 <= record.crop_center_x <= 28.0
    assert 20.0 <= record.crop_center_y <= 28.0
    assert record.output_mode == "image"


def test_crop_directory_accepts_png_files(tmp_path) -> None:
    input_dir = tmp_path / "images"
    output_dir = tmp_path / "crops"
    input_dir.mkdir()
    y, x = np.mgrid[:96, :96]
    planet = np.exp(-(((y - 45.0) ** 2) + ((x - 51.0) ** 2)) / (2 * 8.0**2))
    image = np.clip(np.stack([planet, planet, planet], axis=-1) * 255.0, 0, 255).astype(np.uint8)
    iio.imwrite(input_dir / "frame.png", image)
    iio.imwrite(input_dir / "frame_2.png", image)

    records = crop_arw_directory(input_dir, output_dir, crop_size=40, workers=2)

    assert len(records) == 2
    assert records[0].crop_path.endswith("frame_crop.tif")
    assert tifffile.imread(records[0].crop_path).shape == (40, 40, 3)


def test_crop_directory_accepts_video_frames(tmp_path, monkeypatch) -> None:
    input_dir = tmp_path / "videos"
    output_dir = tmp_path / "crops"
    input_dir.mkdir()
    video_path = input_dir / "capture.mp4"
    video_path.write_bytes(b"placeholder")

    y, x = np.mgrid[:96, :96]
    planet_a = np.exp(-(((y - 45.0) ** 2) + ((x - 51.0) ** 2)) / (2 * 8.0**2))
    planet_b = np.exp(-(((y - 47.0) ** 2) + ((x - 49.0) ** 2)) / (2 * 8.0**2))
    frame_a = np.clip(np.stack([planet_a, planet_a, planet_a], axis=-1) * 255.0, 0, 255).astype(np.uint8)
    frame_b = np.clip(np.stack([planet_b, planet_b, planet_b], axis=-1) * 255.0, 0, 255).astype(np.uint8)
    calls: dict[str, int] = {}

    def fake_iter_video_frames(path, *, frame_step=1):
        calls["frame_step"] = frame_step
        assert path == video_path
        yield 0, frame_a.astype(np.uint16) * np.uint16(257)
        yield 2, frame_b.astype(np.uint16) * np.uint16(257)

    monkeypatch.setattr("pylanetstack.crop.iter_video_frames", fake_iter_video_frames)

    records = crop_arw_directory(input_dir, output_dir, crop_size=40, video_frame_step=2)

    assert calls["frame_step"] == 2
    assert len(records) == 2
    assert records[0].source_type == "video"
    assert records[0].source_frame_index == 0
    assert records[1].source_frame_index == 2
    assert records[0].crop_path.endswith("capture_frame000000_crop.tif")
    assert tifffile.imread(records[0].crop_path).shape == (40, 40, 3)


def test_crop_directory_auto_sizes_png_files(tmp_path) -> None:
    input_dir = tmp_path / "images"
    output_dir = tmp_path / "crops"
    input_dir.mkdir()
    y, x = np.mgrid[:300, :420]
    disk = (((y - 150.0) ** 2) + ((x - 210.0) ** 2)) <= 90.0**2
    image = np.where(disk[..., None], np.array([220, 230, 210], dtype=np.uint8), 0)
    iio.imwrite(input_dir / "sun.png", image)

    records = crop_arw_directory(input_dir, output_dir, crop_size="auto", auto_crop_min_size=80)

    crop = tifffile.imread(records[0].crop_path)
    assert crop.shape[0] == crop.shape[1]
    assert crop.shape[0] > 200
    assert records[0].crop_size == crop.shape[0]


def test_crop_directory_auto_uses_one_dataset_size_for_all_png_files(tmp_path) -> None:
    input_dir = tmp_path / "images"
    output_dir = tmp_path / "crops"
    input_dir.mkdir()
    y, x = np.mgrid[:320, :420]
    small_disk = (((y - 145.0) ** 2) + ((x - 200.0) ** 2)) <= 25.0**2
    large_disk = (((y - 165.0) ** 2) + ((x - 225.0) ** 2)) <= 95.0**2
    small = np.where(small_disk[..., None], np.array([180, 190, 175], dtype=np.uint8), 0)
    large = np.where(large_disk[..., None], np.array([220, 230, 210], dtype=np.uint8), 0)
    iio.imwrite(input_dir / "frame_001.png", small)
    iio.imwrite(input_dir / "frame_002.png", large)

    records = crop_arw_directory(
        input_dir,
        output_dir,
        crop_size="auto",
        auto_crop_min_size=80,
        auto_crop_sample_frames=2,
    )

    crop_shapes = [tifffile.imread(record.crop_path).shape[:2] for record in records]
    crop_sizes = {record.crop_size for record in records}
    assert len(crop_sizes) == 1
    assert len(set(crop_shapes)) == 1
    assert crop_shapes[0][0] == crop_shapes[0][1]
    assert crop_shapes[0][0] > 220


def test_crop_directory_skips_bad_raw_and_removes_stale_output(tmp_path, monkeypatch) -> None:
    input_dir = tmp_path / "raw"
    input_dir.mkdir()
    good = input_dir / "JUN04189.ARW"
    bad = input_dir / "JUN04190.ARW"
    next_good = input_dir / "JUN04191.ARW"
    for path in (good, bad, next_good):
        path.write_bytes(b"raw")

    output_dir = tmp_path / "crops"
    output_dir.mkdir()
    stale_bad_crop = output_dir / "JUN04190_crop.tif"
    stale_bad_crop.write_bytes(b"stale")

    class LibRawIOError(Exception):
        pass

    def fake_crop_one_arw(source, out_dir, **_kwargs):
        if source == bad:
            raise LibRawIOError("Input/output error")
        crop_path = out_dir / f"{source.stem}_crop.tif"
        crop_path.write_bytes(b"crop")
        return CropRecord(
            source_path=str(source),
            crop_path=str(crop_path),
            output_mode="linear-rgb",
            crop_size=250,
            top_left_x=0,
            top_left_y=0,
            detected_center_x=10.0,
            detected_center_y=10.0,
            crop_center_x=10.0,
            crop_center_y=10.0,
            source_width=100,
            source_height=100,
            dtype="uint16",
        )

    monkeypatch.setattr("pylanetstack.crop.crop_one_arw", fake_crop_one_arw)

    records = crop_arw_directory(input_dir, output_dir, overwrite=True)

    errors = [json.loads(line) for line in (output_dir / "crop_errors.jsonl").read_text().splitlines()]
    assert len(records) == 2
    assert not stale_bad_crop.exists()
    assert errors[0]["source_path"].endswith("JUN04190.ARW")
    assert errors[0]["error_type"] == "LibRawIOError"
