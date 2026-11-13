"""Settings of the application, read from config.json (A7).

config.json holds everything that depends on the physical setup: camera, printed board,
colours of the blocks, size of the sphere and the lamp. The code reads these values as
settings.<section>.<name>, e.g. settings.colors.min_saturation, with exactly the names
used in config.json. Every other number in the code is a named constant at the top of
its module, together with the reason for its value.
"""

import json
from dataclasses import dataclass
from pathlib import Path

PROJECT_FOLDER = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_FOLDER / "config.json"
COLOR_NAMES = ("red", "yellow", "green", "blue")


@dataclass(frozen=True)
class Camera:
    """Which webcam is used and which picture size is requested from it."""

    number: int
    width_px: int
    height_px: int


@dataclass(frozen=True)
class Board:
    """Geometry of the printed board in millimetres, e.g. a 180 mm field with 30 mm markers."""

    field_size_mm: float
    margin_around_field_mm: float
    marker_size_mm: float
    marker_distance_to_field_mm: float
    field_size_in_image_px: int

    def px_per_mm(self):
        """Pixels per millimetre in the straight field image, e.g. 600 / 180 = 3.33."""
        return self.field_size_in_image_px / self.field_size_mm

    def marker_centers_mm(self):
        """Centre of every ArUco marker, measured from the top-left corner of the field.

        Marker 0 sits above the top-left corner, so its centre has a negative y,
        e.g. (15, -23) for the default board. The y axis points down like in an image.
        """
        half = self.marker_size_mm / 2
        above = -(self.marker_distance_to_field_mm + half)
        below = self.field_size_mm + self.marker_distance_to_field_mm + half
        right = self.field_size_mm - half
        return {
            0: (half, above),
            1: (right, above),
            2: (right, below),
            3: (half, below),
        }


@dataclass(frozen=True)
class Colors:
    """HSV ranges per colour: hue range(s), minimum saturation and minimum brightness."""

    hue_ranges: dict
    min_saturation: dict
    min_brightness: dict


@dataclass(frozen=True)
class Objects:
    """Facts about the blocks that the detection needs."""

    sphere_diameter_mm: float
    prism_pair_colors: tuple


@dataclass(frozen=True)
class Light:
    """Target brightness of the white paper (0-255) for the setup mode."""

    paper_brightness_min: int
    paper_brightness_max: int


@dataclass(frozen=True)
class Detection:
    """Smallest area in the straight field image that counts as an object."""

    min_object_area_px: int


@dataclass(frozen=True)
class Changes:
    """When a change is reported: after how many equal frames and from which distance on."""

    frames_until_reported: int
    min_move_share_of_field: float


@dataclass(frozen=True)
class Settings:
    """All sections of config.json."""

    camera: Camera
    board: Board
    colors: Colors
    objects: Objects
    light: Light
    detection: Detection
    changes: Changes


def load_settings(path=DEFAULT_CONFIG_PATH):
    """Reads config.json and returns the settings, e.g. settings.camera.number 1 for the first USB webcam."""
    with open(path, encoding="utf-8") as file:
        raw = json.load(file)
    colors = raw["colors"]
    _check_colors(colors["hue_ranges"])
    return Settings(
        camera=Camera(**raw["camera"]),
        board=Board(**raw["board"]),
        colors=Colors(
            hue_ranges={name: [tuple(r) for r in colors["hue_ranges"][name]] for name in COLOR_NAMES},
            min_saturation=_per_color(colors["min_saturation"]),
            min_brightness=_per_color(colors["min_brightness"]),
        ),
        objects=Objects(
            sphere_diameter_mm=raw["objects"]["sphere_diameter_mm"],
            prism_pair_colors=tuple(raw["objects"]["prism_pair_colors"]),
        ),
        light=Light(**raw["light"]),
        detection=Detection(**raw["detection"]),
        changes=Changes(**raw["changes"]),
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
