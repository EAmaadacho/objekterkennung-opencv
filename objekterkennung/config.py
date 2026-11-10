"""Settings of the application, read from config.json (A7)."""

import json
from dataclasses import dataclass
from pathlib import Path

PROJECT_FOLDER = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_FOLDER / "config.json"
COLOR_NAMES = ("red", "yellow", "green", "blue")


@dataclass(frozen=True)
class Board:
    """Geometry of the printed board in millimetres, e.g. a 180 mm field with 30 mm markers."""

    field_mm: float
    marker_mm: float
    marker_gap_mm: float
    warp_px: int

    def marker_centers_mm(self):
        """Centre of every ArUco marker, measured from the top-left corner of the field.

        Marker 0 sits above the top-left corner, so its centre has a negative y,
        e.g. (15, -23) for the default board. The y axis points down like in an image.
        """
        half = self.marker_mm / 2
        above = -(self.marker_gap_mm + half)
        below = self.field_mm + self.marker_gap_mm + half
        right = self.field_mm - half
        return {
            0: (half, above),
            1: (right, above),
            2: (right, below),
            3: (half, below),
        }


@dataclass(frozen=True)
class Settings:
    """All settings of one run: camera, board, colour ranges and detection thresholds."""

    camera_index: int
    board: Board
    saturation_min: dict
    value_min: int
    hue_ranges: dict
    min_area_px: int
    stable_frames: int
    move_threshold: float


def load_settings(path=DEFAULT_CONFIG_PATH):
    """Reads config.json and returns the settings, e.g. camera_index 0 for the first webcam."""
    with open(path, encoding="utf-8") as file:
        raw = json.load(file)
    _check_colors(raw["hue_ranges"])
    return Settings(
        camera_index=raw["camera_index"],
        board=Board(**raw["board"]),
        saturation_min=_per_color(raw["saturation_min"]),
        value_min=raw["value_min"],
        hue_ranges={name: [tuple(r) for r in ranges] for name, ranges in raw["hue_ranges"].items()},
        min_area_px=raw["min_area_px"],
        stable_frames=raw["stable_frames"],
        move_threshold=raw["move_threshold"],
    )


def _per_color(value):
    """Allows one number for all colours or one number per colour, e.g. {"blue": 140, ...}."""
    if isinstance(value, dict):
        return {name: value[name] for name in COLOR_NAMES}
    return {name: value for name in COLOR_NAMES}


def _check_colors(hue_ranges):
    """Every object colour needs at least one hue range, otherwise it could never be found."""
    missing = [name for name in COLOR_NAMES if not hue_ranges.get(name)]
    if missing:
        raise ValueError(f"config.json has no hue range for: {', '.join(missing)}")
