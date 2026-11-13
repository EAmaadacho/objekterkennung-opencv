"""Tests for separating coloured objects from the paper; the field images are drawn with cv2."""

import cv2
import numpy as np

from objekterkennung.config import load_settings
from objekterkennung.detection.segmentation import (
    clean_mask,
    color_mask,
    split_touching,
)
from objekterkennung.detection.shapes import classify_shape

FIELD_PX = 600
WHITE = (255, 255, 255)
RED = (40, 30, 210)
DARK_RED = (25, 20, 130)
SEAM = (20, 15, 90)
# Only slightly darker than RED: the seam between two flush blocks is often this faint.
FAINT_SEAM = (40, 30, 180)
BLUE = (200, 90, 30)
SHADOW_GREY = (150, 150, 150)
# Measured with the ring light above the field: paper and shadows have hue 100-130 and
# a saturation of at most about 30; this test shadow has hue 107, saturation 44.
SHADOW_BLUISH = (150, 135, 124)
# Measured on the real objects (13.11.): the red sphere and red blocks have hue 170-178,
# the yellow cuboids hue 23-27.
RED_SPHERE = (75, 43, 200)  # HSV (174, 200, 200)
YELLOW_CUBOID = (50, 200, 230)  # HSV (25, 200, 230)


def test_A2_red_block_is_found_in_red_mask():
    settings = load_settings()
    field = _empty_field()
    cv2.rectangle(field, (100, 100), (180, 180), RED, -1)

    mask = color_mask(field, "red", settings)

    assert mask[140, 140] == 255
    assert mask[300, 300] == 0


def test_A2_blue_block_is_not_in_red_mask():
    settings = load_settings()
    field = _empty_field()
    cv2.circle(field, (300, 300), 40, BLUE, -1)

    mask = color_mask(field, "red", settings)

    assert not mask.any()


def test_A2_grey_shadow_is_ignored():
    settings = load_settings()
    field = _empty_field()
    cv2.rectangle(field, (200, 200), (400, 400), SHADOW_GREY, -1)

    masks = [color_mask(field, color, settings) for color in settings.colors.hue_ranges]

    assert not any(mask.any() for mask in masks)


def test_A2_bluish_shadow_on_paper_is_not_blue():
    settings = load_settings()
    field = _empty_field()
    cv2.rectangle(field, (200, 200), (400, 400), SHADOW_BLUISH, -1)

    mask = color_mask(field, "blue", settings)

    assert not mask.any()


def test_A2_clean_mask_removes_noise_and_fills_holes():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[5, 5] = 255
    mask[40:80, 40:80] = 255
    mask[60, 60] = 0

    cleaned = clean_mask(mask)

    assert cleaned[5, 5] == 0
    assert cleaned[60, 60] == 255


def test_A5_two_touching_cubes_are_split():
    mask = np.zeros((FIELD_PX, FIELD_PX), dtype=np.uint8)
    cv2.rectangle(mask, (100, 100), (180, 180), 255, -1)
    cv2.rectangle(mask, (181, 130), (261, 210), 255, -1)

    objects = split_touching(mask, load_settings())

    assert len(objects) == 2


def test_A5_cuboid_stays_one_object():
    mask = np.zeros((FIELD_PX, FIELD_PX), dtype=np.uint8)
    cv2.rectangle(mask, (100, 100), (266, 183), 255, -1)

    objects = split_touching(mask, load_settings())

    assert len(objects) == 1


def test_A5_two_touching_spheres_are_split():
    mask = np.zeros((FIELD_PX, FIELD_PX), dtype=np.uint8)
    cv2.circle(mask, (200, 300), 75, 255, -1)
    cv2.circle(mask, (348, 300), 75, 255, -1)

    objects = split_touching(mask, load_settings())

    assert len(objects) == 2


def test_A5_two_flush_cubes_are_split_at_the_visible_seam():
    # Without an offset the outline is a plain 2:1 rectangle; only the dark seam shows two cubes.
    settings = load_settings()
    field = _empty_field()
    cv2.rectangle(field, (100, 100), (180, 180), RED, -1)
    cv2.rectangle(field, (182, 100), (262, 180), RED, -1)
    cv2.line(field, (181, 100), (181, 180), SEAM, 2)

    objects = split_touching(color_mask(field, "red", settings), settings, field)

    assert len(objects) == 2


def test_A5_two_flush_cubes_with_faint_seam_are_split():
    # The normal edge search misses this seam; the finer one for large areas finds it.
    settings = load_settings()
    field = _empty_field()
    cv2.rectangle(field, (100, 100), (210, 210), RED, -1)
    cv2.rectangle(field, (212, 100), (322, 210), RED, -1)
    cv2.line(field, (211, 100), (211, 210), FAINT_SEAM, 2)

    objects = split_touching(color_mask(field, "red", settings), settings, field)

    assert len(objects) == 2


def test_A5_large_cuboid_without_seam_stays_one_object():
    settings = load_settings()
    field = _empty_field()
    cv2.rectangle(field, (100, 100), (322, 210), RED, -1)

    objects = split_touching(color_mask(field, "red", settings), settings, field)

    assert len(objects) == 1


def test_A5_darker_side_face_stays_with_its_cube():
    # The camera sees one side face of the cube below the top face, darker and divided by an edge.
    settings = load_settings()
    field = _empty_field()
    cv2.rectangle(field, (100, 100), (180, 180), RED, -1)
    cv2.rectangle(field, (100, 183), (180, 240), DARK_RED, -1)
    cv2.line(field, (100, 181), (180, 181), SEAM, 2)

    objects = split_touching(color_mask(field, "red", settings), settings, field)

    assert len(objects) == 1


def test_A5_ball_with_reflection_stays_one_object():
    # A bright reflection has almost no colour, so it leaves a hole in the mask.
    mask = np.zeros((FIELD_PX, FIELD_PX), dtype=np.uint8)
    cv2.circle(mask, (300, 300), 75, 255, -1)
    cv2.circle(mask, (292, 300), 10, 0, -1)

    objects = split_touching(mask, load_settings())

    assert len(objects) == 1


def test_A5_tiny_specks_are_dropped():
    mask = np.zeros((FIELD_PX, FIELD_PX), dtype=np.uint8)
    cv2.rectangle(mask, (10, 10), (15, 15), 255, -1)

    objects = split_touching(mask, load_settings())

    assert objects == []


def test_A5_cuboid_gets_back_the_strip_hidden_by_the_ball():
    # The ball covers a 22 px strip of the cuboid's left side, as seen by the camera.
    mask = np.zeros((FIELD_PX, FIELD_PX), dtype=np.uint8)
    cv2.rectangle(mask, (250, 200), (330, 393), 255, -1)
    cv2.circle(mask, (195, 240), 77, 255, -1)

    objects = split_touching(mask, load_settings())

    cuboid = [piece for piece in objects if classify_shape(piece) == "cuboid"]
    assert sorted(classify_shape(piece) for piece in objects) == ["cuboid", "sphere"]
    assert cuboid[0][240, 255] == 255


def test_A5_prism_next_to_the_ball_is_not_filled_up_to_a_square():
    # The ball lies in the open corner of the prism's square, like in the test recordings.
    mask = np.zeros((FIELD_PX, FIELD_PX), dtype=np.uint8)
    cv2.fillPoly(mask, [np.array([[200, 200], [200, 330], [330, 330]], np.int32)], 255)
    cv2.circle(mask, (330, 200), 77, 255, -1)

    objects = split_touching(mask, load_settings())

    assert sorted(classify_shape(piece) for piece in objects) == ["prism", "sphere"]


def test_A2_red_sphere_counts_as_red_and_yellow_cuboid_as_yellow():
    settings = load_settings()
    field = _empty_field()
    cv2.circle(field, (200, 300), 70, RED_SPHERE, -1)
    cv2.rectangle(field, (350, 260), (540, 340), YELLOW_CUBOID, -1)

    red = color_mask(field, "red", settings)
    yellow = color_mask(field, "yellow", settings)

    assert red[300, 200] == 255 and red[300, 450] == 0
    assert yellow[300, 450] == 255 and yellow[300, 200] == 0


def _empty_field():
    return np.full((FIELD_PX, FIELD_PX, 3), WHITE, dtype=np.uint8)
