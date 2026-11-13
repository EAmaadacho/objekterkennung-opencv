"""Tests for cutting a ball out of an area where it touches a block of the same colour (A1, A5)."""

import cv2
import numpy as np

from objekterkennung.detection.ball import OUTLINE_MATCH_MIN, OUTLINE_TOLERANCE_PX, cut_ball, outline_match, refine_circle
from objekterkennung.detection.segmentation import split_touching
from objekterkennung.detection.shapes import classify_shape

FIELD_PX = 600
BALL_RADIUS_PX = 75
MIN_AREA_PX = 3000


def test_A5_ball_touching_a_cuboid_is_cut_out():
    mask = np.zeros((FIELD_PX, FIELD_PX), dtype=np.uint8)
    cv2.rectangle(mask, (100, 150), (183, 316), 255, -1)
    cv2.circle(mask, (183 + BALL_RADIUS_PX - 10, 300), BALL_RADIUS_PX, 255, -1)

    pieces = split_touching(mask, MIN_AREA_PX)

    assert sorted(classify_shape(piece) for piece in pieces) == ["cuboid", "sphere"]


def test_A1_cube_alone_has_no_ball():
    mask = np.zeros((FIELD_PX, FIELD_PX), dtype=np.uint8)
    cv2.rectangle(mask, (200, 200), (333, 333), 255, -1)

    ball, rest = cut_ball(mask)

    assert ball is None
    assert (rest == mask).all()


def test_A5_roughly_found_circle_is_moved_onto_the_ball():
    mask = np.zeros((FIELD_PX, FIELD_PX), dtype=np.uint8)
    cv2.rectangle(mask, (100, 150), (183, 316), 255, -1)
    cv2.circle(mask, (250, 300), BALL_RADIUS_PX, 255, -1)

    x, y, radius = refine_circle(mask, 256, 294, BALL_RADIUS_PX - 6)

    assert outline_match(mask, 256, 294, BALL_RADIUS_PX - 6) < OUTLINE_MATCH_MIN
    assert outline_match(mask, x, y, radius) >= OUTLINE_MATCH_MIN
    assert abs(x - 250) <= OUTLINE_TOLERANCE_PX and abs(y - 300) <= OUTLINE_TOLERANCE_PX
