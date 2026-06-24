from __future__ import annotations

from pathlib import Path

import pytest

from pylanetstack.cli import _copy_loaded_config, parse_args
from pylanetstack.config import KNOWN_CONFIG_PATHS, _flatten_config, defaults_for_command, load_config


def _assert_config_matches_hardcoded_defaults(config_path: Path) -> None:
    config = load_config(config_path)
    commands = {
        "crop": ["crop", "input", "output"],
        "stack": ["stack", "crops", "stack.tif"],
        "run": ["run", "input", "output"],
    }

    for command, argv in commands.items():
        builtin = vars(parse_args([*argv, "--no-config"]))
        configured = vars(parse_args([*argv, "--config", str(config_path)]))
        config_keys = defaults_for_command(config, command)
        mismatches = {
            key: (builtin[key], configured[key])
            for key in config_keys
            if builtin.get(key) != configured.get(key)
        }

        assert mismatches == {}


def test_default_config_file_applies_to_stack(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pylanetstack.yaml").write_text(
        """
filtering:
  keep_best_percent: 10
channel_alignment:
  enabled: false
output:
  stack_upsample: 2
postprocessing:
  sharpening: 700
  upsample: 2
  upsample_smoothing: 0.5
""",
        encoding="utf-8",
    )

    args = parse_args(["stack", "crops", "stack.tif"])

    assert args.keep_best_percent == 10
    assert args.channel_align is False
    assert args.stack_upsample == 2
    assert args.post_sharpening == 700
    assert args.post_upsample == 2
    assert args.post_upsample_smoothing == 0.5
    assert args.config_path == tmp_path / "pylanetstack.yaml"
    assert args.loaded_config_path == tmp_path / "pylanetstack.yaml"


def test_legacy_default_config_file_name_still_loads(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "planetstack.yaml").write_text(
        """
filtering:
  keep_best_percent: 12
""",
        encoding="utf-8",
    )

    args = parse_args(["stack", "crops", "stack.tif"])

    assert args.keep_best_percent == 12
    assert args.config_path == tmp_path / "planetstack.yaml"
    assert args.loaded_config_path == tmp_path / "planetstack.yaml"


def test_cli_values_override_config_file(tmp_path) -> None:
    config_path = tmp_path / "pylanetstack.yaml"
    config_path.write_text(
        """
postprocessing:
  sharpening: 700
  wb_strength: 1
""",
        encoding="utf-8",
    )

    args = parse_args(
        [
            "stack",
            "crops",
            "stack.tif",
            "--config",
            str(config_path),
            "--post-sharpening",
            "900",
        ]
    )

    assert args.post_sharpening == 900
    assert args.post_wb_strength == 1


def test_auto_level_values_parse_from_cli_and_config(tmp_path) -> None:
    config_path = tmp_path / "pylanetstack.yaml"
    config_path.write_text(
        """
postprocessing:
  level_black: auto
  level_white: auto
""",
        encoding="utf-8",
    )

    args = parse_args(["stack", "crops", "stack.tif", "--config", str(config_path)])
    overridden = parse_args(
        [
            "stack",
            "crops",
            "stack.tif",
            "--config",
            str(config_path),
            "--post-level-white",
            "42",
        ]
    )

    assert args.post_level_black == "auto"
    assert args.post_level_white == "auto"
    assert overridden.post_level_black == "auto"
    assert overridden.post_level_white == 42.0


def test_no_config_ignores_default_config_file(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pylanetstack.yaml").write_text(
        """
postprocessing:
  sharpening: 700
""",
        encoding="utf-8",
    )

    args = parse_args(["stack", "crops", "stack.tif", "--no-config"])

    assert args.post_sharpening == 500
    assert args.loaded_config_path is None


def test_explicit_config_applies_to_run_crop_and_stack_sections(tmp_path) -> None:
    config_path = tmp_path / "custom.yaml"
    config_path.write_text(
        """
crop:
  crop_size: auto
  auto_crop_margin: 1.5
  auto_crop_min_size: 128
  auto_crop_max_size: 2048
  auto_crop_sample_frames: 5
  skip_bad_raw: false
performance:
  workers: 3
filtering:
  keep_best_percent: 15
output:
  compression: zlib
run:
  overwrite: true
""",
        encoding="utf-8",
    )

    args = parse_args(["run", "input", "output", "--config", str(config_path)])

    assert args.crop_size == "auto"
    assert args.auto_crop_margin == 1.5
    assert args.auto_crop_min_size == 128
    assert args.auto_crop_max_size == 2048
    assert args.auto_crop_sample_frames == 5
    assert args.skip_bad_raw is False
    assert args.workers == 3
    assert args.keep_best_percent == 15
    assert args.stack_compression == "zlib"
    assert args.overwrite is True
    assert args.loaded_config_path == config_path


def test_loaded_config_can_be_copied_to_output_folder(tmp_path) -> None:
    config_path = tmp_path / "jupiter.yaml"
    config_text = """
filtering:
  keep_best_percent: 20
"""
    config_path.write_text(config_text, encoding="utf-8")
    output_dir = tmp_path / "out"

    args = parse_args(["stack", "crops", str(output_dir / "stack.tif"), "--config", str(config_path)])
    copied_path = _copy_loaded_config(args, output_dir)

    assert copied_path == output_dir / "pylanetstack_used_config.yaml"
    assert copied_path.read_text(encoding="utf-8") == config_text


def test_no_config_does_not_write_used_config_copy(tmp_path) -> None:
    args = parse_args(["stack", "crops", str(tmp_path / "stack.tif"), "--no-config"])

    copied_path = _copy_loaded_config(args, tmp_path)

    assert copied_path is None
    assert not (tmp_path / "pylanetstack_used_config.yaml").exists()


def test_invalid_config_choice_fails_fast(tmp_path) -> None:
    config_path = tmp_path / "bad.yaml"
    config_path.write_text(
        """
alignment:
  method: sparkle
""",
        encoding="utf-8",
    )

    with pytest.raises(SystemExit):
        parse_args(["stack", "crops", "stack.tif", "--config", str(config_path)])


def test_unknown_config_key_fails_fast(tmp_path) -> None:
    config_path = tmp_path / "bad.yaml"
    config_path.write_text(
        """
postprocessing:
  extra_magic: 42
""",
        encoding="utf-8",
    )

    with pytest.raises(SystemExit):
        parse_args(["stack", "crops", "stack.tif", "--config", str(config_path)])


def test_grouped_config_translates_to_destinations(tmp_path) -> None:
    config_path = tmp_path / "pylanetstack.yaml"
    config_path.write_text(
        """
alignment:
  registration_max_shift: 40
  ap:
    patch_size: 24
    max_points: 100
postprocessing:
  background_black: false
""",
        encoding="utf-8",
    )
    config = load_config(config_path)

    defaults = defaults_for_command(config, "stack")

    assert defaults["ap_patch_size"] == 24
    assert defaults["ap_max_points"] == 100
    assert defaults["registration_max_shift"] == 40
    assert defaults["post_background_black"] is False


def test_example_yaml_includes_every_supported_config_key() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config = load_config(repo_root / "pylanetstack.example.yaml")
    example_paths = {path for path, _value in _flatten_config(config)}

    assert example_paths == KNOWN_CONFIG_PATHS


def test_workflow_example_configs_are_valid() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    for name in ["pylanetstack.planets.yaml", "pylanetstack.moon.yaml"]:
        config = load_config(repo_root / "examples" / name)
        stack_defaults = defaults_for_command(config, "stack")
        run_defaults = defaults_for_command(config, "run")

        assert "post_level_black" in stack_defaults
        assert "post_level_white" in stack_defaults
        assert "post_level_black" in run_defaults
        assert "post_level_white" in run_defaults


def test_hardcoded_defaults_match_planet_example_config() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    _assert_config_matches_hardcoded_defaults(repo_root / "examples" / "pylanetstack.planets.yaml")


def test_complete_example_yaml_matches_hardcoded_defaults() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    _assert_config_matches_hardcoded_defaults(repo_root / "pylanetstack.example.yaml")


def test_planet_example_uses_mars_safe_postprocessing() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config = load_config(repo_root / "examples" / "pylanetstack.planets.yaml")
    defaults = defaults_for_command(config, "stack")

    assert defaults["post_level_black"] == "auto"
    assert defaults["post_level_white"] == "auto"
    assert defaults["post_exposure"] == 0.5
    assert defaults["post_highlight_compression"] == 0.7
    assert defaults["post_wb_mode"] == "auto"
    assert defaults["post_wb_strength"] == 1
    assert defaults["post_saturation"] == 3
    assert defaults["post_vibrance"] == 1
    assert defaults["post_upsample"] == 4
    assert defaults["post_upsample_smoothing"] == 1.0


def test_moon_example_uses_auto_levels() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config = load_config(repo_root / "examples" / "pylanetstack.moon.yaml")
    defaults = defaults_for_command(config, "stack")

    assert defaults["post_level_black"] == "auto"
    assert defaults["post_level_white"] == "auto"
