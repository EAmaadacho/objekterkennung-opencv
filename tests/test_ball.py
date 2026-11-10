"""Tests for cutting a ball out of an area where it touches a block of the same colour (A1, A5)."""

import cv2
import numpy as np

from objekterkennung.ball import cut_ball
from objekterkennung.segmentation import split_touching
from objekterkennung.shapes import classify_shape

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
