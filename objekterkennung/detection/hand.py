"""Notices a hand or arm in the field, so that the program pauses its reports meanwhile (A4).

HSV thresholds (script ch. 7.2), opening (ch. 5.2) and contours (ch. 6.1).
"""

import dataclasses

import cv2
import numpy as np

from objekterkennung.detection.segmentation import color_mask

# Skin and a dark sleeve: a warm hue (red to orange) with some colour, or very dark pixels.
WARM_HUE_MAX = 25
WARM_HUE_MIN = 165
SKIN_SATURATION_MIN = 25
SKIN_VALUE_MIN = 50
DARK_VALUE_MAX = 60
# Skin under the lamp can have a red hue with saturation 80-100. The red blocks count from
# saturation 80 (config.json); for the hand check a red block must reach 100, otherwise a
# pale hand would be taken out as a block.
KNOWN_SATURATION_MIN = {"red": 100}
# Around every block 7 px on each side (a 15 px kernel) are counted as the block too,
# because its border is often blurred.
KNOWN_MARGIN_PX = 15
# Small dark or warm specks are removed before the areas are measured (opening, script ch. 5).
OPENING_PX = 7
# A hand always reaches in from the edge of the image. 8000 px are about 27 x 27 mm.
HAND_AREA_MIN_PX = 8000
BORDER_PX = 3
# A strip of table or cardboard can show along the edge of the image if the paper ends there.
# It is long but thin; a hand or arm is at least this wide (30 px are 9 mm).
STRIP_WIDTH_MAX_PX = 30


def hand_in_field(field_image, settings):
    """True if a large skin-coloured or dark area that is no block touches the edge of the field.

    The blocks themselves are found by their colour masks first and taken out,
    so a block lying near the edge is not mistaken for a hand. Skin and dark areas are
    measured each on their own, so that slightly pink paper next to a dark strip of table
    at the edge does not add up to a hand.
    """
    hsv = cv2.cvtColor(field_image, cv2.COLOR_BGR2HSV)
    # A warm hue lies at both ends of the hue scale, like red, so skin needs two ranges.
    skin = cv2.inRange(hsv, (0, SKIN_SATURATION_MIN, SKIN_VALUE_MIN), (WARM_HUE_MAX, 255, 255))
    skin = skin | cv2.inRange(hsv, (WARM_HUE_MIN, SKIN_SATURATION_MIN, SKIN_VALUE_MIN), (180, 255, 255))
    dark = cv2.inRange(hsv, (0, 0, 0), (180, 255, DARK_VALUE_MAX - 1))
    known = np.zeros(hsv.shape[:2], dtype=np.uint8)
    block_settings = _block_settings(settings)
    for color in settings.colors.hue_ranges:
        known = known | color_mask(field_image, color, block_settings)
    known = cv2.dilate(known, np.ones((KNOWN_MARGIN_PX, KNOWN_MARGIN_PX), np.uint8))
    not_known = cv2.bitwise_not(known)
    return any(_large_area_at_edge(candidate & not_known) for candidate in (skin, dark))


def _block_settings(settings):
    """The settings with the stricter saturation of KNOWN_SATURATION_MIN, used only to take the blocks out."""
    saturation = dict(settings.colors.min_saturation)
    for color, minimum in KNOWN_SATURATION_MIN.items():
        saturation[color] = max(saturation[color], minimum)
    colors = dataclasses.replace(settings.colors, min_saturation=saturation)
    return dataclasses.replace(settings, colors=colors)


def _large_area_at_edge(candidate):
    """True if the candidate mask has an area that touches the edge and is as large and wide as a hand."""
    rest = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, np.ones((OPENING_PX, OPENING_PX), np.uint8))
    contours, _ = cv2.findContours(rest, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = rest.shape
    for contour in contours:
        left, top, box_width, box_height = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        touches_border = (
            left <= BORDER_PX
            or top <= BORDER_PX
            or left + box_width >= width - BORDER_PX
            or top + box_height >= height - BORDER_PX
        )
        (_, _), (side_a, side_b), _ = cv2.minAreaRect(contour)
        if touches_border and area >= HAND_AREA_MIN_PX and min(side_a, side_b) > STRIP_WIDTH_MAX_PX:
            return True
    return False
