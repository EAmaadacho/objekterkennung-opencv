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
# Canny thresholds for the visible border between two touching blocks, e.g. two flush cubes.
EDGE_LOW = 30
EDGE_HIGH = 90
# A seed found between the edges must be a real top face: at least 800 px and 40 px wide.
# A visible side face is a thin strip and is not used as a seed of its own.
EDGE_SEED_MIN_PX = 800
EDGE_SEED_MIN_WIDTH_PX = 40
# A visible side face is also darker than the top face: under the lamp it reaches only about
# 60 % of the top's brightness, while two touching top faces are about equally bright.
SIDE_FACE_BRIGHTNESS_MAX = 0.8


def color_mask(field_image, color, settings):
    """Returns a mask that is 255 where the field shows the given colour and 0 elsewhere.

    The image is converted to HSV first, because there the colour (hue) is one
    number, e.g. about 120 for blue, no matter how bright the light is. White
    paper and grey shadows have almost no saturation, so saturation_min removes them.
    It is set per colour: the webcam makes the paper slightly blue, so shadows on it
    are pale blue, and blue needs a higher limit than the other colours.
    """
    hsv = cv2.cvtColor(field_image, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    # Red needs two ranges, because its hue lies at both ends of the scale (0-10 and 170-180).
    for hue_low, hue_high in settings.hue_ranges[color]:
        lower = (hue_low, settings.saturation_min[color], settings.value_min)
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


def split_touching(mask, min_area_px, field_image=None):
    """Splits a colour mask into one mask per object, also when objects touch.

    First the visible edges of the field image cut the area into pieces: two
    cubes lying flush show a dark line between them. If that gives fewer than
    two seeds, the distance transform finds the centre of every object as a
    "hill" instead. In both cases watershed draws the final border.
    Areas smaller than min_area_px are dropped as noise.
    """
    count, labels = cv2.connectedComponents(mask)
    edges = None if field_image is None else _edges(field_image)
    objects = []
    for label in range(1, count):
        part = np.where(labels == label, 255, 0).astype(np.uint8)
        if cv2.countNonZero(part) < min_area_px:
            continue
        part = fill_holes(part)
        seeds = None if edges is None else _edge_seeds(part, edges, field_image)
        if seeds is not None and cv2.connectedComponents(seeds)[0] > 2:
            objects.extend(_watershed(part, seeds, field_image, min_area_px))
        else:
            objects.extend(_split_one_part(part, min_area_px))
    return objects


def _edges(field_image):
    """Canny edges of the field, made 3 px wide so that a thin border line closes.

    Every colour channel is searched on its own, because the orange ball and a
    yellow cuboid are almost equally bright and differ only in the green channel.
    """
    blurred = cv2.GaussianBlur(field_image, (5, 5), 0)
    edges = np.zeros(field_image.shape[:2], dtype=np.uint8)
    for channel in cv2.split(blurred):
        edges = cv2.bitwise_or(edges, cv2.Canny(channel, EDGE_LOW, EDGE_HIGH))
    return cv2.dilate(edges, np.ones((3, 3), np.uint8))


def _edge_seeds(part, edges, field_image):
    """The top faces inside one area, found between the edges.

    Only large, wide and bright pieces count: a side face is a thin or dark strip
    and belongs to the top face next to it.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL_PX, MORPH_KERNEL_PX))
    inner = cv2.erode(cv2.bitwise_and(part, cv2.bitwise_not(edges)), kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(inner)
    brightness = cv2.cvtColor(field_image, cv2.COLOR_BGR2HSV)[:, :, 2]
    candidates = []
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] < EDGE_SEED_MIN_PX:
            continue
        piece = labels == label
        (_, _), (width, height), _ = cv2.minAreaRect(cv2.findNonZero(piece.astype(np.uint8)))
        if min(width, height) >= EDGE_SEED_MIN_WIDTH_PX:
            candidates.append((piece, np.median(brightness[piece])))
    seeds = np.zeros_like(part)
    brightest = max((value for _, value in candidates), default=0)
    for piece, value in candidates:
        if value >= SIDE_FACE_BRIGHTNESS_MAX * brightest:
            seeds[piece] = 255
    return seeds


def fill_holes(part):
    """Fills every hole inside one object, e.g. a bright reflection on the ball.

    A hole near the middle would give the distance transform two "hills",
    and watershed would then cut one ball into two objects.
    """
    contours, _ = cv2.findContours(part, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(part)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return filled


def _split_one_part(part, min_area_px):
    """Watershed on one connected area; returns one mask per found object."""
    distance = cv2.distanceTransform(part, cv2.DIST_L2, 5)
    seeds = np.where(distance >= SEED_RATIO * distance.max(), 255, 0).astype(np.uint8)
    seed_count, markers = cv2.connectedComponents(seeds)
    if seed_count <= 2:
        return [part]
    return _watershed(part, seeds, cv2.cvtColor(part, cv2.COLOR_GRAY2BGR), min_area_px)


def _watershed(part, seeds, image, min_area_px):
    """Grows every seed until it meets its neighbour; returns one mask per seed."""
    seed_count, markers = cv2.connectedComponents(seeds)
    # Marker 1 is the background, the seeds are 2, 3, ... and 0 means "still unknown".
    markers = markers + 1
    markers[(part == 255) & (seeds == 0)] = 0
    cv2.watershed(image, markers)
    pieces = []
    for marker in range(2, seed_count + 1):
        piece = np.where(markers == marker, 255, 0).astype(np.uint8)
        if cv2.countNonZero(piece) >= min_area_px:
            pieces.append(piece)
    return pieces
