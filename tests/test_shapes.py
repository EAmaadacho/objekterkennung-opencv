"""Tests for shape recognition; the outlines are drawn with cv2 at real size (600 px = 180 mm)."""

import cv2
import numpy as np
import pytest

from objekterkennung.shapes import center_px, classify_shape

FIELD_PX = 600
CUBE_PX = 83
SPHERE_RADIUS_PX = 75


def test_A1_square_is_a_cube():
    mask = _rotated_box(CUBE_PX, CUBE_PX, angle=0)

    shape = classify_shape(mask)

    assert shape == "cube"


def test_A1_rotated_square_is_still_a_cube():
    mask = _rotated_box(CUBE_PX, CUBE_PX, angle=35)

    shape = classify_shape(mask)

    assert shape == "cube"


def test_A1_cube_with_visible_side_face_is_still_a_cube():
    # Near the field edge the camera also sees one side of the cube: the outline becomes a hexagon.
    mask = _empty_mask()
    hexagon = np.array([[200, 200], [283, 200], [293, 210], [293, 293], [210, 293], [200, 283]])
    cv2.fillPoly(mask, [hexagon], 255)

    shape = classify_shape(mask)

    assert shape == "cube"


def test_A1_cube_with_dark_corners_is_not_a_sphere():
    # Two dark corners drop out of the colour mask, so the outline becomes almost round.
    mask = _empty_mask()
    octagon = np.array([[235, 200], [283, 200], [283, 248], [248, 283], [200, 283], [200, 235]])
    cv2.fillPoly(mask, [octagon], 255)

    shape = classify_shape(mask)

    assert shape == "cube"


def test_A1_long_rectangle_is_a_cuboid():
    mask = _rotated_box(2 * CUBE_PX, CUBE_PX, angle=20)

    shape = classify_shape(mask)

    assert shape == "cuboid"


@pytest.mark.parametrize(
    "corners",
    [
        [[200, 300], [283, 300], [241, 228]],
        [[200, 300], [283, 300], [200, 217]],
        [[150, 300], [316, 300], [233, 217]],
    ],
)
def test_A1_triangle_is_a_prism(corners):
    mask = _empty_mask()
    cv2.fillPoly(mask, [np.array(corners)], 255)

    shape = classify_shape(mask)

    assert shape == "prism"


def test_A1_triangle_with_blunt_tip_is_still_a_prism():
    # The real prisms have bevelled edges, so the camera sees the tip cut off.
    mask = _empty_mask()
    blunt_triangle = np.array([[160, 300], [320, 300], [271, 245], [209, 245]])
    cv2.fillPoly(mask, [blunt_triangle], 255)

    shape = classify_shape(mask)

    assert shape == "prism"


def test_A1_circle_is_a_sphere():
    mask = _empty_mask()
    cv2.circle(mask, (300, 300), SPHERE_RADIUS_PX, 255, -1)

    shape = classify_shape(mask)

    assert shape == "sphere"


def test_A1_empty_mask_is_unknown():
    mask = _empty_mask()

    shape = classify_shape(mask)

    assert shape == "unknown"


def test_A3_center_of_a_square_is_its_middle():
    mask = _empty_mask()
    cv2.rectangle(mask, (100, 200), (200, 300), 255, -1)

    x, y = center_px(mask)

    assert (round(x), round(y)) == (150, 250)


def _empty_mask():
    return np.zeros((FIELD_PX, FIELD_PX), dtype=np.uint8)


def _rotated_box(width, height, angle):
    mask = _empty_mask()
    corners = cv2.boxPoints(((300, 300), (width, height), angle)).astype(np.int32)
    cv2.fillPoly(mask, [corners], 255)
    return mask
