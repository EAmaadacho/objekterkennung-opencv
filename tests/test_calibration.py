"""Tests for the calibration mode: slider values and saving to config.json (the window itself needs a camera)."""

import json

from objekterkennung.calibration import bounds_from_hue_ranges, hue_ranges_from_bounds, save_colors
from objekterkennung.config import DEFAULT_CONFIG_PATH, load_settings


def test_A7_normal_slider_range_gives_one_hue_range():
    ranges = hue_ranges_from_bounds(95, 130)

    assert ranges == [(95, 130)]


def test_A7_wrapping_slider_range_gives_two_hue_ranges_for_red():
    ranges = hue_ranges_from_bounds(170, 10)

    assert ranges == [(170, 180), (0, 10)]


def test_A7_red_ranges_go_back_onto_the_sliders():
    bounds = bounds_from_hue_ranges([(0, 10), (170, 180)])

    assert bounds == (170, 10)


def test_A7_saved_colors_are_loaded_again(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    hue_ranges = dict(load_settings(path).hue_ranges)
    hue_ranges["blue"] = [(100, 125)]

    save_colors(path, hue_ranges, saturation_min=90, value_min=50)

    settings = load_settings(path)
    assert settings.hue_ranges["blue"] == [(100, 125)]
    assert (settings.saturation_min, settings.value_min) == (90, 50)
    assert settings.camera_index == json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))["camera_index"]


def test_A7_saving_unchanged_values_keeps_config_json_as_it_is(tmp_path):
    original = DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")
    path = tmp_path / "config.json"
    path.write_text(original, encoding="utf-8")
    settings = load_settings(path)

    save_colors(path, settings.hue_ranges, settings.saturation_min, settings.value_min)

    assert path.read_text(encoding="utf-8") == original
