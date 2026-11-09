"""Tests for separating coloured objects from the paper; the field images are drawn with cv2."""

import cv2
import numpy as np

from objekterkennung.config import load_settings
from objekterkennung.segmentation import clean_mask, color_mask

FIELD_PX = 600
WHITE = (255, 255, 255)
RED = (40, 30, 210)
BLUE = (200, 90, 30)
SHADOW_GREY = (150, 150, 150)


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

    masks = [color_mask(field, color, settings) for color in settings.hue_ranges]

    assert not any(mask.any() for mask in masks)


def test_A2_clean_mask_removes_noise_and_fills_holes():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[5, 5] = 255
    mask[40:80, 40:80] = 255
    mask[60, 60] = 0

    cleaned = clean_mask(mask)

    assert cleaned[5, 5] == 0
    assert cleaned[60, 60] == 255


def _empty_field():
    return np.full((FIELD_PX, FIELD_PX, 3), WHITE, dtype=np.uint8)
