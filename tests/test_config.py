"""Tests for the settings read from config.json."""

import json

import pytest

from objekterkennung.config import COLOR_NAMES, DEFAULT_CONFIG_PATH, load_settings

A4_HEIGHT_MM = 297


def test_A7_default_config_has_all_four_colors():
    settings = load_settings()

    assert set(settings.colors.hue_ranges) == set(COLOR_NAMES)


def test_A7_red_needs_two_hue_ranges():
    # Red lies at both ends of the OpenCV hue scale (0-180).
    settings = load_settings()

    assert len(settings.colors.hue_ranges["red"]) == 2


def test_A7_missing_color_is_rejected(tmp_path):
    raw = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    del raw["colors"]["hue_ranges"]["blue"]
    path = tmp_path / "config.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="blue"):
        load_settings(path)


def test_A3_markers_lie_outside_the_field():
    board = load_settings().board

    centers = board.marker_centers_mm()

    for _, y in centers.values():
        assert y < 0 or y > board.field_size_mm


def test_A3_board_fits_on_a4():
    board = load_settings().board

    height = board.field_size_mm + 2 * (board.marker_distance_to_field_mm + board.marker_size_mm)

    assert height <= A4_HEIGHT_MM
