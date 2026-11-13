"""Tests for splitting two prisms that lie together as a square (A1, A5)."""

import cv2
import numpy as np

from objekterkennung.detection.prism_pair import split_prism_pair
from objekterkennung.detection.shapes import classify_shape

PAPER = (230, 230, 230)
BLUE = (200, 80, 20)
DARK_BLUE = (120, 40, 10)


def _square(angle, joint_color=None):
    """A blue square on paper, optionally with a darker line on one diagonal, and its mask."""
    image = np.full((300, 300, 3), PAPER, dtype=np.uint8)
    corners = cv2.boxPoints(((150, 150), (120, 120), angle)).astype(np.int32)
    cv2.fillPoly(image, [corners], BLUE)
    if joint_color is not None:
        cv2.line(image, tuple(corners[0]), tuple(corners[2]), joint_color, 2)
    mask = np.zeros((300, 300), dtype=np.uint8)
    cv2.fillPoly(mask, [corners], 255)
    return image, mask


def test_A1_square_with_dark_diagonal_becomes_two_prisms():
    image, mask = _square(20, joint_color=DARK_BLUE)

    pieces = split_prism_pair(mask, image)

    assert [classify_shape(piece) for piece in pieces] == ["prism", "prism"]


def test_A1_square_without_joint_stays_one_cube():
    image, mask = _square(20)

    pieces = split_prism_pair(mask, image)

    assert len(pieces) == 1
    assert classify_shape(pieces[0]) == "cube"


def test_A1_line_across_the_middle_is_not_a_joint():
    image, mask = _square(0)
    cv2.line(image, (90, 150), (210, 150), DARK_BLUE, 2)

    pieces = split_prism_pair(mask, image)

    assert len(pieces) == 1


def test_A1_dark_spot_on_half_the_diagonal_is_not_a_joint():
    image, mask = _square(20)
    corners = cv2.boxPoints(((150, 150), (120, 120), 20)).astype(np.int32)
    middle = tuple(((corners[0] + corners[2]) // 2).tolist())
    cv2.line(image, tuple(corners[0].tolist()), middle, DARK_BLUE, 2)

    pieces = split_prism_pair(mask, image)

    assert len(pieces) == 1
