"""Notices a hand or arm in the field, so that the program pauses its reports meanwhile (A4).

HSV thresholds (script ch. 7.2), opening (ch. 5.2) and contours (ch. 6.1).
"""

import cv2
import numpy as np

from objekterkennung.detection.segmentation import color_mask

# Skin and a dark sleeve: a warm hue (red to orange) with some colour, or very dark pixels.
WARM_HUE_MAX = 25
WARM_HUE_MIN = 165
SKIN_SATURATION_MIN = 25
SKIN_VALUE_MIN = 50
DARK_VALUE_MAX = 60
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
    measured each on their own: with many red blocks in the image the camera tints the
    paper slightly pink, and next to a dark strip of table at the edge the two together
    looked like a hand (live test 14.11.). Measured apart, 76 of 77 hands in the recordings
    were still found, and the false alarms dropped from 11 to 2 of 213 frames without a hand.
    """
    hsv = cv2.cvtColor(field_image, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    warm = (hue <= WARM_HUE_MAX) | (hue >= WARM_HUE_MIN)
    skin = warm & (saturation >= SKIN_SATURATION_MIN) & (value >= SKIN_VALUE_MIN)
    dark = value < DARK_VALUE_MAX
    known = np.zeros(hue.shape, dtype=np.uint8)
    for color in settings.colors.hue_ranges:
        known = known | color_mask(field_image, color, settings)
    known = cv2.dilate(known, np.ones((KNOWN_MARGIN_PX, KNOWN_MARGIN_PX), np.uint8))
    return any(_large_area_at_edge(candidate & (known == 0)) for candidate in (skin, dark))


def _large_area_at_edge(candidate):
    """True if the candidate pixels form an area that touches the edge and is as large and wide as a hand."""
    rest = np.where(candidate, 255, 0).astype(np.uint8)
    rest = cv2.morphologyEx(rest, cv2.MORPH_OPEN, np.ones((OPENING_PX, OPENING_PX), np.uint8))
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
