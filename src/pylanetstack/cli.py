"""Command-line interface for PyLanetStack.

The CLI is deliberately a thin translation layer: it gathers command-line and
YAML configuration values, normalizes a few user-friendly types, then hands the
work to the crop, stack, or run modules. Keeping it thin makes future GUI or
notebook frontends easier to build on the same core functions.
"""

from __future__ import annotations

import argparse
import math
import shutil
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from . import __version__
from .config import defaults_for_command, find_default_config, load_config
from .crop import crop_arw_directory
from .pipeline import run_pipeline
from .stack import stack_crops


COMMANDS = {"crop", "stack", "run"}


def _compression(value: str) -> str | None:
    if value.lower() in {"none", "off", ""}:
        return None
    return value


def _crop_size(value: object) -> int | str:
    if isinstance(value, str) and value.lower() == "auto":
        return "auto"
    try:
        size = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("crop size must be a positive integer or 'auto'") from error
    if size <= 0:
        raise argparse.ArgumentTypeError("crop size must be positive")
    return size


def _level_value(value: object) -> float | str:
    if isinstance(value, str) and value.strip().lower() == "auto":
        return "auto"
    try:
        level = float(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("level value must be a number on a 0-255 scale or 'auto'") from error
    if not math.isfinite(level):
        raise argparse.ArgumentTypeError("level value must be finite")
    return level


def _add_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        dest="config_path",
        type=Path,
        default=None,
        help="YAML config file. CLI arguments override config values.",
    )
    parser.add_argument(
        "--no-config",
        action="store_true",
        help="Ignore default config files in the current directory.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pylanetstack",
        description="Crop and stack planetary astrophotography frames.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    _add_config_arguments(parser)
    subparsers = parser.add_subparsers(dest="command", required=True)

    crop = subparsers.add_parser("crop", help="Find planets in RAW/image/video sources and save TIFF crops.")
    _add_config_arguments(crop)
    crop.add_argument("input", type=Path, help="RAW/image/video file or folder containing supported inputs.")
    crop.add_argument("output", type=Path, help="Folder for cropped TIFF files.")
    crop.add_argument("--crop-size", type=_crop_size, default=250, help="Square crop size in pixels, or auto.")
    crop.add_argument("--auto-crop-margin", type=float, default=1.35, help="Padding multiplier for auto crop.")
    crop.add_argument("--auto-crop-min-size", type=int, default=250, help="Minimum auto crop size in pixels.")
    crop.add_argument("--auto-crop-max-size", type=int, default=None, help="Maximum auto crop size in pixels.")
    crop.add_argument(
        "--auto-crop-sample-frames",
        type=int,
        default=12,
        help="Number of frames to inspect when choosing one dataset-wide auto crop size.",
    )
    crop.add_argument(
        "--output-mode",
        choices=["linear-rgb", "raw-mosaic"],
        default="linear-rgb",
        help="Write local linear RGB crops or literal raw Bayer mosaic crops.",
    )
    crop.add_argument(
        "--white-balance",
        choices=["camera", "auto", "none"],
        default="camera",
        help="Color-balance linear RGB crops. Camera WB is the default.",
    )
    crop.add_argument("--recursive", action="store_true", help="Search folders recursively.")
    crop.add_argument("--save-raw-crop", action="store_true", help="Also save the raw mosaic crop.")
    crop.add_argument("--compression", type=_compression, default=None, help="TIFF compression, or none.")
    crop.add_argument("--overwrite", action="store_true", help="Overwrite existing crops.")
    crop.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Worker threads for crop processing. Use 0 for all CPU cores.",
    )
    crop.add_argument(
        "--skip-bad-raw",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip RAW files that LibRaw/rawpy cannot read and continue the batch.",
    )
    crop.add_argument("--max-frames", type=int, default=None, help="Process at most N frames.")
    crop.add_argument(
        "--video-frame-step",
        type=int,
        default=1,
        help="For video inputs, keep every Nth decoded frame. Default 1 keeps every frame.",
    )

    stack = subparsers.add_parser("stack", help="Register and stack cropped image frames.")
    _add_config_arguments(stack)
    stack.add_argument("input", type=Path, help="Folder containing TIFF, PNG, JPEG, or BMP crops.")
    stack.add_argument("output", type=Path, help="Output stack TIFF path.")
    stack.add_argument("--recursive", action="store_true", help="Search folders recursively.")
    stack.add_argument("--method", choices=["ap", "flow", "rigid", "none"], default="ap")
    stack.add_argument(
        "--quality-weighting",
        dest="quality_weighting",
        action="store_true",
        default=True,
        help="Weight frames by high-frequency detail.",
    )
    stack.add_argument(
        "--no-quality-weighting",
        dest="quality_weighting",
        action="store_false",
        help="Give all frames equal weight.",
    )
    stack.add_argument(
        "--quality-weight-scale",
        type=float,
        default=1.5,
        help="Strength of high-frequency weighting. 0 disables weighting effect; 1 is neutral.",
    )
    stack.add_argument("--quality-weight-min", type=float, default=0.25)
    stack.add_argument("--quality-weight-max", type=float, default=4.0)
    stack.add_argument(
        "--keep-best-percent",
        type=float,
        default=25.0,
        help="Only stack the sharpest N percent of frames by quality score.",
    )
    stack.add_argument(
        "--keep-best-frames",
        type=int,
        default=None,
        help="Only stack the sharpest N frames by quality score.",
    )
    stack.add_argument("--reference", choices=["best", "first"], default="best")
    stack.add_argument("--upsample-factor", type=int, default=20)
    stack.add_argument("--flow-warps", type=int, default=5)
    stack.add_argument("--flow-iterations", type=int, default=10)
    stack.add_argument("--flow-attachment", type=float, default=15.0)
    stack.add_argument("--flow-tightness", type=float, default=0.3)
    stack.add_argument(
        "--flow-smoothing-sigma",
        type=float,
        default=0.1,
        help="Gaussian smoothing sigma applied to flow fields before warping.",
    )
    stack.add_argument(
        "--flow-max-pixels",
        type=float,
        default=2.0,
        help="Maximum local flow displacement in pixels. Use 0 for no cap.",
    )
    stack.add_argument(
        "--registration-max-shift",
        type=float,
        default=50.0,
        help="Reject selected frames whose global registration shift exceeds this many pixels. Use 0 to disable.",
    )
    stack.add_argument("--ap-patch-size", type=int, default=12, help="Alignment-point patch size in pixels.")
    stack.add_argument("--ap-spacing", type=int, default=6, help="Spacing between alignment-point centers.")
    stack.add_argument("--ap-blend-sigma", type=float, default=18.0, help="Blend radius for AP shifts.")
    stack.add_argument(
        "--ap-max-points",
        type=int,
        default=2500,
        help="Maximum AP points per frame. Use 0 to disable the cap.",
    )
    stack.add_argument("--ap-min-mask-fraction", type=float, default=0.25)
    stack.add_argument("--ap-min-contrast", type=float, default=0.015)
    stack.add_argument(
        "--channel-align",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Align RGB channels to reduce atmospheric color dispersion.",
    )
    stack.add_argument("--channel-reference", choices=["red", "green", "blue"], default="green")
    stack.add_argument("--channel-align-max-shift", type=float, default=3.0)
    stack.add_argument("--channel-align-upsample-factor", type=int, default=20)
    stack.add_argument("--update-reference-every", type=int, default=0)
    stack.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Worker threads for quality scoring and fixed-reference stack registration. Use 0 for all CPU cores.",
    )
    stack.add_argument("--max-frames", type=int, default=None)
    stack.add_argument(
        "--stack-upsample",
        type=float,
        default=1.0,
        help="Accumulate the scientific stack on a larger grid. 1 disables stack-time upsampling.",
    )
    stack.add_argument("--output-dtype", choices=["float32", "uint16"], default="uint16")
    stack.add_argument("--compression", type=_compression, default=None, help="TIFF compression, or none.")
    stack.add_argument(
        "--preview",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also write a display-stretched 16-bit preview TIFF.",
    )
    stack.add_argument("--preview-low-percentile", type=float, default=0.5)
    stack.add_argument("--preview-high-percentile", type=float, default=99.95)
    stack.add_argument(
        "--postprocess",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write a Photoshop-style processed output TIFF.",
    )
    stack.add_argument("--post-input-max", type=float, default=65535.0)
    stack.add_argument("--post-level-black", type=_level_value, default="auto")
    stack.add_argument("--post-level-white", type=_level_value, default="auto")
    stack.add_argument("--post-level-gamma", type=float, default=1.0)
    stack.add_argument("--post-exposure", type=float, default=0.5)
    stack.add_argument(
        "--post-highlight-compression",
        type=float,
        default=0.7,
        help="Soft-highlight compression after exposure. 0 disables it.",
    )
    stack.add_argument("--post-texture", type=float, default=100.0)
    stack.add_argument("--post-texture-radius", type=float, default=2.0)
    stack.add_argument("--post-dehaze", type=float, default=100.0)
    stack.add_argument("--post-dehaze-radius", type=float, default=12.0)
    stack.add_argument(
        "--post-sharpening",
        type=float,
        default=500.0,
        help="Luminance unsharp-mask strength. 500 is default; higher values are allowed.",
    )
    stack.add_argument("--post-sharpening-radius", type=float, default=0.8)
    stack.add_argument(
        "--post-wb-mode",
        choices=["auto", "none"],
        default="auto",
        help="Processed-output white balance. Auto neutralizes planet disk midtones.",
    )
    stack.add_argument(
        "--post-wb-strength",
        type=float,
        default=1.0,
        help="Blend strength for automatic processed-output white balance, 0 to 1.",
    )
    stack.add_argument("--post-red-balance", type=float, default=1.0)
    stack.add_argument("--post-green-balance", type=float, default=1.0)
    stack.add_argument("--post-blue-balance", type=float, default=1.0)
    stack.add_argument(
        "--post-saturation",
        type=float,
        default=3.0,
        help="Processed-output color saturation multiplier. 1 preserves source chroma.",
    )
    stack.add_argument(
        "--post-vibrance",
        type=float,
        default=1.0,
        help="Extra color boost for weakly saturated pixels. 0 disables it.",
    )
    stack.add_argument(
        "--post-background-black",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Feather processed output to black outside the detected planet.",
    )
    stack.add_argument("--post-background-dilate", type=int, default=3)
    stack.add_argument("--post-background-feather", type=float, default=4.0)
    stack.add_argument(
        "--post-upsample",
        type=float,
        default=4.0,
        help="Enlarge the processed output by this scale factor. 1 disables upsampling.",
    )
    stack.add_argument(
        "--post-upsample-smoothing",
        type=float,
        default=1.0,
        help="Gaussian smoothing sigma applied during processed-output upsampling. 0 disables it.",
    )

    run = subparsers.add_parser("run", help="Crop RAW/image/video sources and stack the crops.")
    _add_config_arguments(run)
    run.add_argument("input", type=Path, help="RAW/image/video file or folder containing supported inputs.")
    run.add_argument("output", type=Path, help="Output folder.")
    run.add_argument("--crop-size", type=_crop_size, default=250)
    run.add_argument("--auto-crop-margin", type=float, default=1.35)
    run.add_argument("--auto-crop-min-size", type=int, default=250)
    run.add_argument("--auto-crop-max-size", type=int, default=None)
    run.add_argument("--auto-crop-sample-frames", type=int, default=12)
    run.add_argument("--output-mode", choices=["linear-rgb", "raw-mosaic"], default="linear-rgb")
    run.add_argument(
        "--white-balance",
        choices=["camera", "auto", "none"],
        default="camera",
        help="Color-balance linear RGB crops. Camera WB is the default.",
    )
    run.add_argument("--recursive", action="store_true")
    run.add_argument("--save-raw-crop", action="store_true")
    run.add_argument("--crop-compression", type=_compression, default=None)
    run.add_argument("--stack-compression", type=_compression, default=None)
    run.add_argument("--overwrite", action="store_true")
    run.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Worker threads for crop, quality, and fixed-reference stack registration. Use 0 for all CPU cores.",
    )
    run.add_argument(
        "--skip-bad-raw",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip RAW files that LibRaw/rawpy cannot read and continue the batch.",
    )
    run.add_argument("--max-frames", type=int, default=None)
    run.add_argument(
        "--video-frame-step",
        type=int,
        default=1,
        help="For video inputs, keep every Nth decoded frame. Default 1 keeps every frame.",
    )
    run.add_argument("--method", choices=["ap", "flow", "rigid", "none"], default="ap")
    run.add_argument("--quality-weighting", dest="quality_weighting", action="store_true", default=True)
    run.add_argument("--no-quality-weighting", dest="quality_weighting", action="store_false")
    run.add_argument(
        "--quality-weight-scale",
        type=float,
        default=1.5,
        help="Strength of high-frequency weighting. 0 disables weighting effect; 1 is neutral.",
    )
    run.add_argument("--quality-weight-min", type=float, default=0.25)
    run.add_argument("--quality-weight-max", type=float, default=4.0)
    run.add_argument(
        "--keep-best-percent",
        type=float,
        default=25.0,
        help="Only stack the sharpest N percent of frames by quality score.",
    )
    run.add_argument(
        "--keep-best-frames",
        type=int,
        default=None,
        help="Only stack the sharpest N frames by quality score.",
    )
    run.add_argument("--reference", choices=["best", "first"], default="best")
    run.add_argument("--upsample-factor", type=int, default=20)
    run.add_argument("--flow-warps", type=int, default=5)
    run.add_argument("--flow-iterations", type=int, default=10)
    run.add_argument("--flow-attachment", type=float, default=15.0)
    run.add_argument("--flow-tightness", type=float, default=0.3)
    run.add_argument(
        "--flow-smoothing-sigma",
        type=float,
        default=0.1,
        help="Gaussian smoothing sigma applied to flow fields before warping.",
    )
    run.add_argument(
        "--flow-max-pixels",
        type=float,
        default=2.0,
        help="Maximum local flow displacement in pixels. Use 0 for no cap.",
    )
    run.add_argument(
        "--registration-max-shift",
        type=float,
        default=50.0,
        help="Reject selected frames whose global registration shift exceeds this many pixels. Use 0 to disable.",
    )
    run.add_argument("--ap-patch-size", type=int, default=12, help="Alignment-point patch size in pixels.")
    run.add_argument("--ap-spacing", type=int, default=6, help="Spacing between alignment-point centers.")
    run.add_argument("--ap-blend-sigma", type=float, default=18.0, help="Blend radius for AP shifts.")
    run.add_argument(
        "--ap-max-points",
        type=int,
        default=2500,
        help="Maximum AP points per frame. Use 0 to disable the cap.",
    )
    run.add_argument("--ap-min-mask-fraction", type=float, default=0.25)
    run.add_argument("--ap-min-contrast", type=float, default=0.015)
    run.add_argument(
        "--channel-align",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Align RGB channels to reduce atmospheric color dispersion.",
    )
    run.add_argument("--channel-reference", choices=["red", "green", "blue"], default="green")
    run.add_argument("--channel-align-max-shift", type=float, default=3.0)
    run.add_argument("--channel-align-upsample-factor", type=int, default=20)
    run.add_argument("--update-reference-every", type=int, default=0)
    run.add_argument(
        "--stack-upsample",
        type=float,
        default=1.0,
        help="Accumulate the scientific stack on a larger grid. 1 disables stack-time upsampling.",
    )
    run.add_argument("--output-dtype", choices=["float32", "uint16"], default="uint16")
    run.add_argument(
        "--preview",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also write a display-stretched 16-bit preview TIFF.",
    )
    run.add_argument("--preview-low-percentile", type=float, default=0.5)
    run.add_argument("--preview-high-percentile", type=float, default=99.95)
    run.add_argument(
        "--postprocess",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write a Photoshop-style processed output TIFF.",
    )
    run.add_argument("--post-input-max", type=float, default=65535.0)
    run.add_argument("--post-level-black", type=_level_value, default="auto")
    run.add_argument("--post-level-white", type=_level_value, default="auto")
    run.add_argument("--post-level-gamma", type=float, default=1.0)
    run.add_argument("--post-exposure", type=float, default=0.5)
    run.add_argument(
        "--post-highlight-compression",
        type=float,
        default=0.7,
        help="Soft-highlight compression after exposure. 0 disables it.",
    )
    run.add_argument("--post-texture", type=float, default=100.0)
    run.add_argument("--post-texture-radius", type=float, default=2.0)
    run.add_argument("--post-dehaze", type=float, default=100.0)
    run.add_argument("--post-dehaze-radius", type=float, default=12.0)
    run.add_argument(
        "--post-sharpening",
        type=float,
        default=500.0,
        help="Luminance unsharp-mask strength. 500 is default; higher values are allowed.",
    )
    run.add_argument("--post-sharpening-radius", type=float, default=0.8)
    run.add_argument(
        "--post-wb-mode",
        choices=["auto", "none"],
        default="auto",
        help="Processed-output white balance. Auto neutralizes planet disk midtones.",
    )
    run.add_argument(
        "--post-wb-strength",
        type=float,
        default=1.0,
        help="Blend strength for automatic processed-output white balance, 0 to 1.",
    )
    run.add_argument("--post-red-balance", type=float, default=1.0)
    run.add_argument("--post-green-balance", type=float, default=1.0)
    run.add_argument("--post-blue-balance", type=float, default=1.0)
    run.add_argument(
        "--post-saturation",
        type=float,
        default=3.0,
        help="Processed-output color saturation multiplier. 1 preserves source chroma.",
    )
    run.add_argument(
        "--post-vibrance",
        type=float,
        default=1.0,
        help="Extra color boost for weakly saturated pixels. 0 disables it.",
    )
    run.add_argument(
        "--post-background-black",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Feather processed output to black outside the detected planet.",
    )
    run.add_argument("--post-background-dilate", type=int, default=3)
    run.add_argument("--post-background-feather", type=float, default=4.0)
    run.add_argument(
        "--post-upsample",
        type=float,
        default=4.0,
        help="Enlarge the processed output by this scale factor. 1 disables upsampling.",
    )
    run.add_argument(
        "--post-upsample-smoothing",
        type=float,
        default=1.0,
        help="Gaussian smoothing sigma applied during processed-output upsampling. 0 disables it.",
    )

    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    argv = sys.argv[1:] if argv is None else list(argv)
    parser = build_parser()
    try:
        command = _command_from_argv(argv)
        config_path = _requested_config_path(argv)
        no_config = _has_flag(argv, "--no-config")
        active_config_path: Path | None = None
        loaded_config_path: Path | None = None
        if command is not None and not no_config:
            active_config_path = config_path or find_default_config()
            if active_config_path is not None:
                config = load_config(active_config_path)
                loaded_config_path = active_config_path
                defaults = defaults_for_command(config, command)
                defaults = _normalize_config_defaults(parser, defaults)
                defaults["config_path"] = active_config_path
                defaults["loaded_config_path"] = loaded_config_path
                _command_parser(parser, command).set_defaults(**defaults)
        elif config_path is not None:
            active_config_path = config_path

        args = parser.parse_args(argv)
        if active_config_path is not None:
            args.config_path = active_config_path
        args.loaded_config_path = loaded_config_path
        return args
    except ValueError as error:
        parser.error(str(error))
        raise AssertionError("unreachable")


def _command_from_argv(argv: Iterable[str]) -> str | None:
    skip_next = False
    for token in argv:
        if skip_next:
            skip_next = False
            continue
        if token == "--config":
            skip_next = True
            continue
        if token in COMMANDS:
            return token
    return None


def _requested_config_path(argv: list[str]) -> Path | None:
    for index, token in enumerate(argv):
        if token == "--config":
            if index + 1 >= len(argv):
                raise ValueError("--config requires a path")
            return Path(argv[index + 1]).expanduser()
        if token.startswith("--config="):
            return Path(token.split("=", 1)[1]).expanduser()
    return None


def _has_flag(argv: Iterable[str], flag: str) -> bool:
    return any(token == flag for token in argv)


def _normalize_config_defaults(
    parser: argparse.ArgumentParser,
    defaults: dict[str, Any],
) -> dict[str, Any]:
    actions = _actions_by_dest(parser)
    normalized: dict[str, Any] = {}
    for dest, value in defaults.items():
        action = actions.get(dest)
        if action is None:
            normalized[dest] = value
            continue
        normalized[dest] = _normalize_config_value(action, value)
    return normalized


def _actions_by_dest(parser: argparse.ArgumentParser) -> dict[str, argparse.Action]:
    actions: dict[str, argparse.Action] = {}

    def visit(current: argparse.ArgumentParser) -> None:
        for action in current._actions:
            if action.dest != argparse.SUPPRESS:
                actions.setdefault(action.dest, action)
            if isinstance(action, argparse._SubParsersAction):
                for subparser in action.choices.values():
                    visit(subparser)

    visit(parser)
    return actions


def _command_parser(parser: argparse.ArgumentParser, command: str) -> argparse.ArgumentParser:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            subparser = action.choices.get(command)
            if subparser is not None:
                return subparser
    raise ValueError(f"Unknown command: {command}")


def _normalize_config_value(action: argparse.Action, value: Any) -> Any:
    if value is None:
        return None
    if _is_bool_action(action):
        return _coerce_bool(action.dest, value)
    if action.type is not None:
        try:
            value = action.type(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid config value for {action.dest}: {value!r}") from error
    if action.choices is not None and value not in action.choices:
        choices = ", ".join(str(choice) for choice in action.choices)
        raise ValueError(f"Invalid config value for {action.dest}: {value!r}; choose one of {choices}")
    return value


def _is_bool_action(action: argparse.Action) -> bool:
    if isinstance(action, argparse.BooleanOptionalAction):
        return True
    return action.nargs == 0 and action.const in {True, False}


def _coerce_bool(dest: str, value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "on", "1"}:
            return True
        if normalized in {"false", "no", "off", "0"}:
            return False
    raise ValueError(f"Invalid config value for {dest}: expected boolean, got {value!r}")


def _copy_loaded_config(args: argparse.Namespace, output_dir: Path | str) -> Path | None:
    """Copy the YAML file that was actually loaded into an output directory."""

    config_path = getattr(args, "loaded_config_path", None)
    if config_path is None:
        return None
    source = Path(config_path).expanduser()
    destination_dir = Path(output_dir).expanduser()
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / "pylanetstack_used_config.yaml"
    if source.resolve() == destination.resolve():
        return destination
    shutil.copy2(source, destination)
    return destination


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.command == "crop":
        _copy_loaded_config(args, args.output)
        crop_arw_directory(
            args.input,
            args.output,
            crop_size=args.crop_size,
            auto_crop_margin=args.auto_crop_margin,
            auto_crop_min_size=args.auto_crop_min_size,
            auto_crop_max_size=args.auto_crop_max_size,
            auto_crop_sample_frames=args.auto_crop_sample_frames,
            output_mode=args.output_mode,
            white_balance=args.white_balance,
            recursive=args.recursive,
            save_raw_crop=args.save_raw_crop,
            compression=args.compression,
            overwrite=args.overwrite,
            skip_bad_raw=args.skip_bad_raw,
            max_frames=args.max_frames,
            workers=args.workers,
            video_frame_step=args.video_frame_step,
        )
        return 0

    if args.command == "stack":
        _copy_loaded_config(args, args.output.parent)
        stack_crops(
            args.input,
            args.output,
            recursive=args.recursive,
            method=args.method,
            quality_weighting=args.quality_weighting,
            quality_weight_scale=args.quality_weight_scale,
            quality_weight_min=args.quality_weight_min,
            quality_weight_max=args.quality_weight_max,
            keep_best_percent=args.keep_best_percent,
            keep_best_frames=args.keep_best_frames,
            reference=args.reference,
            upsample_factor=args.upsample_factor,
            flow_warps=args.flow_warps,
            flow_iterations=args.flow_iterations,
            flow_attachment=args.flow_attachment,
            flow_tightness=args.flow_tightness,
            flow_smoothing_sigma=args.flow_smoothing_sigma,
            flow_max_pixels=args.flow_max_pixels,
            registration_max_shift=args.registration_max_shift,
            ap_patch_size=args.ap_patch_size,
            ap_spacing=args.ap_spacing,
            ap_blend_sigma=args.ap_blend_sigma,
            ap_max_points=args.ap_max_points,
            ap_min_mask_fraction=args.ap_min_mask_fraction,
            ap_min_contrast=args.ap_min_contrast,
            channel_align=args.channel_align,
            channel_reference=args.channel_reference,
            channel_align_max_shift=args.channel_align_max_shift,
            channel_align_upsample_factor=args.channel_align_upsample_factor,
            update_reference_every=args.update_reference_every,
            stack_upsample=args.stack_upsample,
            max_frames=args.max_frames,
            output_dtype=args.output_dtype,
            compression=args.compression,
            preview=args.preview,
            preview_low_percentile=args.preview_low_percentile,
            preview_high_percentile=args.preview_high_percentile,
            postprocess=args.postprocess,
            post_input_max=args.post_input_max,
            post_level_black=args.post_level_black,
            post_level_white=args.post_level_white,
            post_level_gamma=args.post_level_gamma,
            post_exposure=args.post_exposure,
            post_highlight_compression=args.post_highlight_compression,
            post_texture=args.post_texture,
            post_texture_radius=args.post_texture_radius,
            post_dehaze=args.post_dehaze,
            post_dehaze_radius=args.post_dehaze_radius,
            post_sharpening=args.post_sharpening,
            post_sharpening_radius=args.post_sharpening_radius,
            post_wb_mode=args.post_wb_mode,
            post_wb_strength=args.post_wb_strength,
            post_red_balance=args.post_red_balance,
            post_green_balance=args.post_green_balance,
            post_blue_balance=args.post_blue_balance,
            post_saturation=args.post_saturation,
            post_vibrance=args.post_vibrance,
            post_background_black=args.post_background_black,
            post_background_dilate=args.post_background_dilate,
            post_background_feather=args.post_background_feather,
            post_upsample=args.post_upsample,
            post_upsample_smoothing=args.post_upsample_smoothing,
            workers=args.workers,
        )
        return 0

    if args.command == "run":
        _copy_loaded_config(args, args.output)
        run_pipeline(
            args.input,
            args.output,
            crop_size=args.crop_size,
            auto_crop_margin=args.auto_crop_margin,
            auto_crop_min_size=args.auto_crop_min_size,
            auto_crop_max_size=args.auto_crop_max_size,
            auto_crop_sample_frames=args.auto_crop_sample_frames,
            output_mode=args.output_mode,
            white_balance=args.white_balance,
            recursive=args.recursive,
            save_raw_crop=args.save_raw_crop,
            crop_compression=args.crop_compression,
            stack_compression=args.stack_compression,
            overwrite=args.overwrite,
            skip_bad_raw=args.skip_bad_raw,
            max_frames=args.max_frames,
            video_frame_step=args.video_frame_step,
            method=args.method,
            quality_weighting=args.quality_weighting,
            quality_weight_scale=args.quality_weight_scale,
            quality_weight_min=args.quality_weight_min,
            quality_weight_max=args.quality_weight_max,
            keep_best_percent=args.keep_best_percent,
            keep_best_frames=args.keep_best_frames,
            reference=args.reference,
            upsample_factor=args.upsample_factor,
            flow_warps=args.flow_warps,
            flow_iterations=args.flow_iterations,
            flow_attachment=args.flow_attachment,
            flow_tightness=args.flow_tightness,
            flow_smoothing_sigma=args.flow_smoothing_sigma,
            flow_max_pixels=args.flow_max_pixels,
            registration_max_shift=args.registration_max_shift,
            ap_patch_size=args.ap_patch_size,
            ap_spacing=args.ap_spacing,
            ap_blend_sigma=args.ap_blend_sigma,
            ap_max_points=args.ap_max_points,
            ap_min_mask_fraction=args.ap_min_mask_fraction,
            ap_min_contrast=args.ap_min_contrast,
            channel_align=args.channel_align,
            channel_reference=args.channel_reference,
            channel_align_max_shift=args.channel_align_max_shift,
            channel_align_upsample_factor=args.channel_align_upsample_factor,
            update_reference_every=args.update_reference_every,
            stack_upsample=args.stack_upsample,
            output_dtype=args.output_dtype,
            preview=args.preview,
            preview_low_percentile=args.preview_low_percentile,
            preview_high_percentile=args.preview_high_percentile,
            postprocess=args.postprocess,
            post_input_max=args.post_input_max,
            post_level_black=args.post_level_black,
            post_level_white=args.post_level_white,
            post_level_gamma=args.post_level_gamma,
            post_exposure=args.post_exposure,
            post_highlight_compression=args.post_highlight_compression,
            post_texture=args.post_texture,
            post_texture_radius=args.post_texture_radius,
            post_dehaze=args.post_dehaze,
            post_dehaze_radius=args.post_dehaze_radius,
            post_sharpening=args.post_sharpening,
            post_sharpening_radius=args.post_sharpening_radius,
            post_wb_mode=args.post_wb_mode,
            post_wb_strength=args.post_wb_strength,
            post_red_balance=args.post_red_balance,
            post_green_balance=args.post_green_balance,
            post_blue_balance=args.post_blue_balance,
            post_saturation=args.post_saturation,
            post_vibrance=args.post_vibrance,
            post_background_black=args.post_background_black,
            post_background_dilate=args.post_background_dilate,
            post_background_feather=args.post_background_feather,
            post_upsample=args.post_upsample,
            post_upsample_smoothing=args.post_upsample_smoothing,
            workers=args.workers,
        )
        return 0

    build_parser().error(f"Unknown command: {args.command}")
    return 2
