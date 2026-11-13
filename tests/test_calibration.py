"""Tests for the calibration mode: slider values and saving to config.json (the window itself needs a camera)."""

import json

from objekterkennung.capture.calibration import (
    bounds_from_hue_ranges,
    hue_ranges_from_bounds,
    save_colors,
)
from objekterkennung.config import DEFAULT_CONFIG_PATH, Colors, load_settings


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
    colors = load_settings(path).colors
    hue_ranges = dict(colors.hue_ranges)
    hue_ranges["blue"] = [(100, 125)]
    min_saturation = {"red": 90, "yellow": 90, "green": 90, "blue": 150}
    min_brightness = {"red": 60, "yellow": 60, "green": 40, "blue": 50}

    save_colors(path, Colors(hue_ranges, min_saturation, min_brightness))

    settings = load_settings(path)
    assert settings.colors.hue_ranges["blue"] == [(100, 125)]
    assert (settings.colors.min_saturation["blue"], settings.colors.min_brightness["green"]) == (150, 40)
    assert settings.camera.number == json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))["camera"]["number"]


def test_A7_saving_unchanged_values_keeps_config_json_as_it_is(tmp_path):
    original = DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")
    path = tmp_path / "config.json"
    path.write_text(original, encoding="utf-8")
    settings = load_settings(path)

    save_colors(path, settings.colors)

    assert path.read_text(encoding="utf-8") == original
