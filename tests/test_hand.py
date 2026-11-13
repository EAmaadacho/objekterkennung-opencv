"""Tests for noticing a hand in the field; hand and blocks are drawn with cv2."""

import cv2
import numpy as np

from objekterkennung.config import load_settings
from objekterkennung.hand import hand_in_field

FIELD_PX = 600
WHITE = (235, 225, 220)
# Measured on the real field: skin has a warm hue but much less saturation than the red blocks.
SKIN = (150, 175, 215)
RED = (40, 30, 210)


def test_A4_hand_reaching_in_from_the_edge_is_noticed():
    settings = load_settings()
    field = _empty_field()
    cv2.ellipse(field, (560, 300), (120, 60), 0, 0, 360, SKIN, -1)

    found = hand_in_field(field, settings)

    assert found


def test_A4_red_block_at_the_edge_is_not_a_hand():
    settings = load_settings()
    field = _empty_field()
    cv2.rectangle(field, (0, 250), (85, 335), RED, -1)

    found = hand_in_field(field, settings)

    assert not found


def test_A4_empty_field_has_no_hand():
    settings = load_settings()

    found = hand_in_field(_empty_field(), settings)

    assert not found


def _empty_field():
    return np.full((FIELD_PX, FIELD_PX, 3), WHITE, dtype=np.uint8)


def test_A4_marker_strips_in_the_margin_are_no_hand():
    # The 15 mm margin shows the lower 7 mm of the markers above and the upper 7 mm below the field.
    settings = load_settings()
    field = np.full((700, 700, 3), 180, dtype=np.uint8)
    for left in (50, 550):
        field[0:23, left : left + 100] = 0
        field[677:700, left : left + 100] = 0

    hand = hand_in_field(field, settings)

    assert not hand
