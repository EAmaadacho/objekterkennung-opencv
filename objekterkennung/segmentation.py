"""Separates the coloured objects from the white paper with HSV colour ranges (A2, A5)."""

import cv2
import numpy as np

# 5 px in the 600 px field is 1.5 mm: big enough to remove noise, too small to eat an object.
MORPH_KERNEL_PX = 5
HUE_MAX = 180
# A seed is the inner part of an object: every pixel with at least this share of the
# largest distance to the border. 0.8 keeps one seed for a cuboid, but two for two
# cubes that touch with an offset.
SEED_RATIO = 0.8


def color_mask(field_image, color, settings):
    """Returns a mask that is 255 where the field shows the given colour and 0 elsewhere.

    The image is converted to HSV first, because there the colour (hue) is one
    number, e.g. about 120 for blue, no matter how bright the light is. White
    paper and grey shadows have almost no saturation, so saturation_min removes them.
    """
    hsv = cv2.cvtColor(field_image, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    # Red needs two ranges, because its hue lies at both ends of the scale (0-10 and 170-180).
    for hue_low, hue_high in settings.hue_ranges[color]:
        lower = (hue_low, settings.saturation_min, settings.value_min)
        upper = (min(hue_high, HUE_MAX), 255, 255)
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))
    return clean_mask(mask)


def clean_mask(mask):
    """Removes single noise pixels and closes small holes inside the objects.

    Opening (erode, then dilate) deletes specks smaller than the kernel.
    Closing (dilate, then erode) fills small gaps, e.g. wood grain or a reflection.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL_PX, MORPH_KERNEL_PX))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)


def split_touching(mask, min_area_px):
    """Splits a colour mask into one mask per object, also when objects touch.

    Uses the watershed method: the distance transform finds the centre of
    every object as a "hill", and watershed draws the border in the "valley"
    between two hills. Areas smaller than min_area_px are dropped as noise.
    """
    count, labels = cv2.connectedComponents(mask)
    objects = []
    for label in range(1, count):
        part = np.where(labels == label, 255, 0).astype(np.uint8)
        if cv2.countNonZero(part) < min_area_px:
            continue
        objects.extend(_split_one_part(part, min_area_px))
    return objects


def _split_one_part(part, min_area_px):
    """Watershed on one connected area; returns one mask per found object."""
    distance = cv2.distanceTransform(part, cv2.DIST_L2, 5)
    seeds = np.where(distance >= SEED_RATIO * distance.max(), 255, 0).astype(np.uint8)
    seed_count, markers = cv2.connectedComponents(seeds)
    if seed_count <= 2:
        return [part]
    # Marker 1 is the background, the seeds are 2, 3, ... and 0 means "still unknown".
    markers = markers + 1
    markers[(part == 255) & (seeds == 0)] = 0
    cv2.watershed(cv2.cvtColor(part, cv2.COLOR_GRAY2BGR), markers)
    pieces = []
    for marker in range(2, seed_count + 1):
        piece = np.where(markers == marker, 255, 0).astype(np.uint8)
        if cv2.countNonZero(piece) >= min_area_px:
            pieces.append(piece)
    return pieces
