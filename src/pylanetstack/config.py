"""YAML configuration loading.

The public CLI exposes many tuning knobs, so the YAML layer keeps repeatable
recipes readable by grouping related settings. The maps below translate those
human-oriented groups back into argparse destination names.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_NAMES = (
    "pylanetstack.yaml",
    "pylanetstack.yml",
    ".pylanetstack.yaml",
    ".pylanetstack.yml",
    "planetstack.yaml",
    "planetstack.yml",
    ".planetstack.yaml",
    ".planetstack.yml",
)


ConfigPath = tuple[str, ...]


BASE_CONFIG_MAP: dict[ConfigPath, str] = {
    ("performance", "workers"): "workers",
    ("filtering", "keep_best_percent"): "keep_best_percent",
    ("filtering", "keep_best_frames"): "keep_best_frames",
    ("filtering", "quality_weighting"): "quality_weighting",
    ("filtering", "quality_weight_scale"): "quality_weight_scale",
    ("filtering", "quality_weight_min"): "quality_weight_min",
    ("filtering", "quality_weight_max"): "quality_weight_max",
    ("filtering", "reference"): "reference",
    ("alignment", "method"): "method",
    ("alignment", "upsample_factor"): "upsample_factor",
    ("alignment", "update_reference_every"): "update_reference_every",
    ("alignment", "ap", "patch_size"): "ap_patch_size",
    ("alignment", "ap", "spacing"): "ap_spacing",
    ("alignment", "ap", "blend_sigma"): "ap_blend_sigma",
    ("alignment", "ap", "max_points"): "ap_max_points",
    ("alignment", "ap", "min_mask_fraction"): "ap_min_mask_fraction",
    ("alignment", "ap", "min_contrast"): "ap_min_contrast",
    ("alignment", "flow", "warps"): "flow_warps",
    ("alignment", "flow", "iterations"): "flow_iterations",
    ("alignment", "flow", "attachment"): "flow_attachment",
    ("alignment", "flow", "tightness"): "flow_tightness",
    ("alignment", "flow", "smoothing_sigma"): "flow_smoothing_sigma",
    ("alignment", "flow", "max_pixels"): "flow_max_pixels",
    ("alignment", "registration_max_shift"): "registration_max_shift",
    ("channel_alignment", "enabled"): "channel_align",
    ("channel_alignment", "reference"): "channel_reference",
    ("channel_alignment", "max_shift"): "channel_align_max_shift",
    ("channel_alignment", "upsample_factor"): "channel_align_upsample_factor",
    ("postprocessing", "enabled"): "postprocess",
    ("postprocessing", "input_max"): "post_input_max",
    ("postprocessing", "level_black"): "post_level_black",
    ("postprocessing", "level_white"): "post_level_white",
    ("postprocessing", "level_gamma"): "post_level_gamma",
    ("postprocessing", "exposure"): "post_exposure",
    ("postprocessing", "highlight_compression"): "post_highlight_compression",
    ("postprocessing", "texture"): "post_texture",
    ("postprocessing", "texture_radius"): "post_texture_radius",
    ("postprocessing", "dehaze"): "post_dehaze",
    ("postprocessing", "dehaze_radius"): "post_dehaze_radius",
    ("postprocessing", "sharpening"): "post_sharpening",
    ("postprocessing", "sharpening_radius"): "post_sharpening_radius",
    ("postprocessing", "wb_mode"): "post_wb_mode",
    ("postprocessing", "wb_strength"): "post_wb_strength",
    ("postprocessing", "red_balance"): "post_red_balance",
    ("postprocessing", "green_balance"): "post_green_balance",
    ("postprocessing", "blue_balance"): "post_blue_balance",
    ("postprocessing", "saturation"): "post_saturation",
    ("postprocessing", "vibrance"): "post_vibrance",
    ("postprocessing", "background_black"): "post_background_black",
    ("postprocessing", "background_dilate"): "post_background_dilate",
    ("postprocessing", "background_feather"): "post_background_feather",
    ("postprocessing", "upsample"): "post_upsample",
    ("postprocessing", "upsample_smoothing"): "post_upsample_smoothing",
    ("output", "dtype"): "output_dtype",
    ("output", "stack_upsample"): "stack_upsample",
    ("output", "preview"): "preview",
    ("output", "preview_low_percentile"): "preview_low_percentile",
    ("output", "preview_high_percentile"): "preview_high_percentile",
}


COMMAND_CONFIG_MAPS: dict[str, dict[ConfigPath, str]] = {
    "crop": {
        ("performance", "workers"): "workers",
        ("crop", "crop_size"): "crop_size",
        ("crop", "auto_crop_margin"): "auto_crop_margin",
        ("crop", "auto_crop_min_size"): "auto_crop_min_size",
        ("crop", "auto_crop_max_size"): "auto_crop_max_size",
        ("crop", "auto_crop_sample_frames"): "auto_crop_sample_frames",
        ("crop", "output_mode"): "output_mode",
        ("crop", "white_balance"): "white_balance",
        ("crop", "recursive"): "recursive",
        ("crop", "save_raw_crop"): "save_raw_crop",
        ("crop", "compression"): "compression",
        ("crop", "overwrite"): "overwrite",
        ("crop", "skip_bad_raw"): "skip_bad_raw",
        ("crop", "max_frames"): "max_frames",
        ("crop", "video_frame_step"): "video_frame_step",
    },
    "stack": {
        **BASE_CONFIG_MAP,
        ("stack", "recursive"): "recursive",
        ("stack", "max_frames"): "max_frames",
        ("stack", "compression"): "compression",
        ("output", "compression"): "compression",
    },
    "run": {
        **BASE_CONFIG_MAP,
        ("crop", "crop_size"): "crop_size",
        ("crop", "auto_crop_margin"): "auto_crop_margin",
        ("crop", "auto_crop_min_size"): "auto_crop_min_size",
        ("crop", "auto_crop_max_size"): "auto_crop_max_size",
        ("crop", "auto_crop_sample_frames"): "auto_crop_sample_frames",
        ("crop", "output_mode"): "output_mode",
        ("crop", "white_balance"): "white_balance",
        ("crop", "recursive"): "recursive",
        ("crop", "save_raw_crop"): "save_raw_crop",
        ("crop", "compression"): "crop_compression",
        ("crop", "overwrite"): "overwrite",
        ("crop", "skip_bad_raw"): "skip_bad_raw",
        ("crop", "video_frame_step"): "video_frame_step",
        ("run", "overwrite"): "overwrite",
        ("run", "max_frames"): "max_frames",
        ("run", "crop_compression"): "crop_compression",
        ("run", "stack_compression"): "stack_compression",
        ("output", "compression"): "stack_compression",
    },
}


KNOWN_CONFIG_PATHS = set().union(*(mapping.keys() for mapping in COMMAND_CONFIG_MAPS.values()))


def find_default_config(start: Path | None = None) -> Path | None:
    """Return the first default config file in the working directory."""

    root = Path.cwd() if start is None else start
    for name in DEFAULT_CONFIG_NAMES:
        path = root / name
        if path.is_file():
            return path
    return None


def load_config(path: Path | str) -> dict[str, Any]:
    """Load a YAML config file."""

    config_path = Path(path).expanduser()
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid YAML in {config_path}: {error}") from error
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config file {config_path} must contain a YAML mapping at the top level")
    return payload


def defaults_for_command(config: dict[str, Any], command: str) -> dict[str, Any]:
    """Translate grouped YAML config values to argparse destination defaults."""

    if command not in COMMAND_CONFIG_MAPS:
        return {}
    values: dict[str, Any] = {}
    unknown: list[str] = []
    mapping = COMMAND_CONFIG_MAPS[command]
    for path, value in _flatten_config(config):
        if path not in KNOWN_CONFIG_PATHS:
            unknown.append(".".join(path))
            continue
        dest = mapping.get(path)
        if dest is not None:
            values[dest] = value
    if unknown:
        keys = ", ".join(sorted(unknown))
        raise ValueError(f"Unknown config key(s): {keys}")
    return values


def _flatten_config(config: dict[str, Any]) -> list[tuple[ConfigPath, Any]]:
    leaves: list[tuple[ConfigPath, Any]] = []

    def visit(prefix: ConfigPath, value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if not isinstance(key, str):
                    raise ValueError("Config keys must be strings")
                visit((*prefix, key), child)
            return
        leaves.append((prefix, value))

    visit((), config)
    return leaves
