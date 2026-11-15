"""Tests for the whole detection on a drawn field: colour, shape and position together."""

import cv2
import numpy as np

from objekterkennung.config import load_settings
from objekterkennung.detection.detector import _about_square, detect_objects

# The straight image is the 600 px field plus a 50 px margin on every side.
FIELD_PX = 700
MARGIN_PX = 50
WHITE = (255, 255, 255)
RED = (40, 30, 210)
YELLOW = (40, 200, 230)
GREEN = (60, 160, 40)
BLUE = (200, 90, 30)
CUBE_PX = 83


def test_A1_A2_A3_four_objects_are_found_with_shape_colour_and_position():
    settings = load_settings()
    field = np.full((FIELD_PX, FIELD_PX, 3), WHITE, dtype=np.uint8)
    _square(field, center=(150 + MARGIN_PX, 150 + MARGIN_PX), size=CUBE_PX, color=RED)
    cv2.rectangle(field, (390, 130), (390 + 2 * CUBE_PX, 130 + CUBE_PX), YELLOW, -1)
    cv2.circle(field, (170, 470), 75, GREEN, -1)
    cv2.fillPoly(field, [np.array([[430, 510], [513, 510], [471, 438]])], BLUE)

    objects = detect_objects(field, settings)

    found = {(item.color, item.shape) for item in objects}
    assert found == {("red", "cube"), ("yellow", "cuboid"), ("green", "sphere"), ("blue", "prism")}
    red_cube = next(item for item in objects if item.color == "red")
    assert abs(red_cube.x - 0.25) < 0.01
    assert abs(red_cube.y - 0.75) < 0.01


def test_A5_two_touching_red_cubes_are_two_objects():
    settings = load_settings()
    field = np.full((FIELD_PX, FIELD_PX, 3), WHITE, dtype=np.uint8)
    _square(field, center=(200, 300), size=CUBE_PX, color=RED)
    _square(field, center=(200 + CUBE_PX, 340), size=CUBE_PX, color=RED)

    objects = detect_objects(field, settings)

    assert [(item.color, item.shape) for item in objects] == [("red", "cube"), ("red", "cube")]


def test_A5_touching_objects_of_different_colours_are_separate():
    settings = load_settings()
    field = np.full((FIELD_PX, FIELD_PX, 3), WHITE, dtype=np.uint8)
    _square(field, center=(250, 300), size=CUBE_PX, color=RED)
    _square(field, center=(250 + CUBE_PX, 300), size=CUBE_PX, color=BLUE)

    objects = detect_objects(field, settings)

    assert sorted(item.color for item in objects) == ["blue", "red"]


def test_A1_square_with_side_strip_is_still_checked_for_two_prisms():
    # Test 15.11.: two prisms as a square plus the side face of the cube next to them
    # measured 1.51 : 1 and counted as a cuboid, so the joint was never looked for.
    square_with_strip = np.zeros((FIELD_PX, FIELD_PX), np.uint8)
    cv2.rectangle(square_with_strip, (100, 100), (220, 220), 255, -1)
    cv2.rectangle(square_with_strip, (100, 220), (220, 280), 255, -1)
    cuboid = np.zeros((FIELD_PX, FIELD_PX), np.uint8)
    cv2.rectangle(cuboid, (100, 100), (340, 200), 255, -1)

    assert _about_square(square_with_strip)
    assert not _about_square(cuboid)


def _square(field, center, size, color):
    half = size // 2
    x, y = center
    cv2.rectangle(field, (x - half, y - half), (x + half, y + half), color, -1)
