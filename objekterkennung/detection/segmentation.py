"""Finds the objects of one colour and splits objects that touch (A2, A5).

Colour: HSV and cv2.inRange (script ch. 2.2, 7.2), cleaned by opening and closing (ch. 5.2).
Touching objects: Canny edges (ch. 4.1, 4.2), then watershed, or the distance transform if the
edges show no border. Both go beyond the script; the README says why.
"""

import cv2
import numpy as np

from objekterkennung.detection.ball import CUT_MARGIN_PX, cut_ball
from objekterkennung.detection.shapes import CUBOID_ASPECT_MIN

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
# Two blocks of the same colour lying flush often show only a faint seam that these
# thresholds miss. A second, finer edge search finds it. It is only used for large areas,
# because inside a small block it finds wood grain as well: 16000 px lie above a cube top
# (32 x 32 mm = 11 400 px) and just above a cuboid top (58 x 24 mm = 15 500 px). A cuboid
# with a visible side face can be larger; its result is used only if the finer search finds
# more wide, bright top faces than the normal one, which wood grain did not do in the recordings.
FINE_EDGE_LOW = 10
FINE_EDGE_HIGH = 30
FINE_EDGE_AREA_MIN_PX = 16000
# A seed found between the edges must be a real top face: at least 800 px and 40 px wide.
# A visible side face is a thin strip and is not used as a seed of its own.
EDGE_SEED_MIN_PX = 800
EDGE_SEED_MIN_WIDTH_PX = 40
# A visible side face is also darker than the top face: under the lamp it reaches only about
# 60 % of the top's brightness, while two touching top faces are about equally bright.
SIDE_FACE_BRIGHTNESS_MAX = 0.8
# Around a cut-out ball lies a gap of CUT_MARGIN_PX (ball.py); the area hidden by the
# ball is taken a bit larger than that, so a block next to it touches it.
HIDDEN_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * CUT_MARGIN_PX + 3, 2 * CUT_MARGIN_PX + 3))
# The ball is higher than the blocks, so the camera sees it a little from the side and it can
# hide a strip of a block next to it: at most about 46 mm x 127 mm / 480 mm = 12 mm at the corner
# of the field (ball height x distance from the field centre / camera height). In the recordings
# the hidden strips of the green cuboid were up to 26 px (8 mm) wide; wider gaps under a ball
# were the open half of a prism's rectangle.
HIDDEN_STRIP_MAX_PX = 30
# Two pieces next to a ball are one cuboid if the ball covers at least 18 % of the rectangle
# around them and they fill 90 % of the rest. In the recordings the ball covered 21-25 % of a
# green cuboid cut in two, but only up to 15 % of two red cubes lying beside the red sphere.
JOIN_COVER_MIN = 0.18
JOIN_FILL_MIN = 0.9
# Wood grain on some blocks gives Canny many short lines, so a top face shows no smooth
# part to start from. In a row of three flush blocks of one colour (test 15.11., two blue
# cubes and the blue prism) only the left cube kept a seed, and the whole row became one
# "cuboid". For an area longer than any single block (a cuboid is 58 mm = 193 px) the edges
# are searched once more without lines shorter than GRAIN_EDGE_MIN_PX; seams and outlines
# run across a whole block and stay. In all recordings this split 4 such rows and changed
# nothing else (20 px split only one, 50 px the same 4).
ROW_LENGTH_MIN_PX = 230
GRAIN_EDGE_MIN_PX = 35
# What the background takes inside an area counts as a block only if it is at least this wide.
# A visible side face is a thin strip; the smallest block, a prism, is about 70 px wide.
# In the recordings 55 px kept every side face out (40 px let 9 more through).
LOST_MIN_WIDTH_PX = 55


def color_mask(field_image, color, settings):
    """Returns a mask that is 255 where the field shows the given colour and 0 elsewhere.

    The image is converted to HSV first, because there the colour (hue) is one
    number, e.g. about 120 for blue, no matter how bright the light is. White
    paper and grey shadows have almost no saturation, so min_saturation removes them.
    It is set per colour in config.json, because it depends on the light: strong light
    makes the tops of the blocks paler, and skin is close to red, so red needs a higher
    limit than the other colours. min_brightness is set per colour as well: the dark
    green blocks reach only 27-54 in their shadowed parts, the other colours stay above 60.
    All three values come from config.json (section "colors").
    """
    hsv = cv2.cvtColor(field_image, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    # Red needs two ranges, because its hue lies at both ends of the scale (0-10 and 170-180).
    for hue_low, hue_high in settings.colors.hue_ranges[color]:
        lower = (hue_low, settings.colors.min_saturation[color], settings.colors.min_brightness[color])
        upper = (min(hue_high, HUE_MAX), 255, 255)
        # An image is a numpy array, so | joins two masks pixel by pixel (white where either is white).
        mask = mask | cv2.inRange(hsv, lower, upper)
    return clean_mask(mask)


def clean_mask(mask):
    """Removes single noise pixels and closes small holes inside the objects.

    Opening (erode, then dilate) deletes specks smaller than the kernel.
    Closing (dilate, then erode) fills small gaps, e.g. wood grain or a reflection.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL_PX, MORPH_KERNEL_PX))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)


def split_touching(mask, settings, field_image=None):
    """Splits a colour mask into one mask per object, also when objects touch.

    Every connected area goes through these steps:
    1. Areas smaller than min_object_area_px (config.json, "detection") are noise and dropped.
    2. A ball is cut out first, because it can melt together with a block of the same
       colour, e.g. the green sphere and the green cuboid. The rest starts again at step 1;
       Two pieces of it that form one cuboid partly under the ball are joined again.
    3. The visible edges of the field image cut the area into top faces ("seeds"): two cubes
       lying flush show a dark line between them. Large areas are also searched with finer
       edges, because that seam can be very faint. An area longer than one block is
       searched once more without short lines, because wood grain can hide every top face.
    4. With two or more seeds, watershed draws the border between them. Otherwise the
       distance transform looks for one "hill" per object and watershed splits there.
    """
    min_area_px = settings.detection.min_object_area_px
    sphere_radius_px = settings.objects.sphere_diameter_mm / 2 * settings.board.px_per_mm()
    edges = None if field_image is None else _edges(field_image, EDGE_LOW, EDGE_HIGH)
    fine_edges = None if field_image is None else _edges(field_image, FINE_EDGE_LOW, FINE_EDGE_HIGH)
    objects = []
    for part in areas(mask):
        if np.count_nonzero(part) < min_area_px:
            continue
        part = fill_holes(part)
        ball, rest = cut_ball(part, sphere_radius_px)
        if ball is not None:
            objects.append(ball)
            # The rest may still hold several blocks, so it goes through the same steps.
            # Each block then gets back what the ball hides of it.
            hidden = cv2.dilate(ball, HIDDEN_KERNEL)
            pieces = _join_halves(split_touching(clean_mask(rest), settings, field_image), hidden)
            for piece in pieces:
                objects.append(_add_hidden_part(piece, hidden))
            continue
        seeds = None if edges is None else _edge_seeds(part, edges, field_image)
        count = 0 if seeds is None else len(areas(seeds))
        if fine_edges is not None and np.count_nonzero(part) >= FINE_EDGE_AREA_MIN_PX:
            fine_seeds = _edge_seeds(part, fine_edges, field_image)
            fine_count = len(areas(fine_seeds))
            # The finer search is only trusted if it finds more top faces.
            if fine_count > count:
                seeds, count = fine_seeds, fine_count
        if count < 2 and field_image is not None and _longest_side(part) > ROW_LENGTH_MIN_PX:
            grain_free = _edges(field_image, EDGE_LOW, EDGE_HIGH, GRAIN_EDGE_MIN_PX)
            grain_free_seeds = _edge_seeds(part, grain_free, field_image)
            if len(areas(grain_free_seeds)) >= 2:
                seeds, count = grain_free_seeds, len(areas(grain_free_seeds))
        if count >= 2:
            objects.extend(_watershed(part, seeds, field_image, min_area_px))
        else:
            objects.extend(_split_one_part(part, min_area_px))
    return objects


def _join_halves(pieces, hidden):
    """Joins two pieces next to the ball that are really one cuboid partly under it.

    If the ball lies on a cuboid or throws its shadow across it, the edge search can cut
    the cuboid into two pieces that each look like a cube (test 15.11.). Two pieces are
    joined if both touch the ball, together they are long like a cuboid, the ball covers at
    least JOIN_COVER_MIN of the rectangle around them (cv2.minAreaRect), and they fill at
    least JOIN_FILL_MIN of the part of that rectangle the ball leaves visible.
    """
    pieces = list(pieces)
    for first in range(len(pieces)):
        for second in range(first + 1, len(pieces)):
            joined = pieces[first] | pieces[second]
            if _is_cuboid_under_ball(pieces[first], pieces[second], joined, hidden):
                rest = [piece for index, piece in enumerate(pieces) if index not in (first, second)]
                return _join_halves(rest + [joined], hidden)
    return pieces


def _is_cuboid_under_ball(first, second, joined, hidden):
    """The checks of _join_halves for one pair of pieces."""
    if not all(np.any(cv2.dilate(piece, HIDDEN_KERNEL) & hidden) for piece in (first, second)):
        return False
    rectangle_box = cv2.minAreaRect(outline_points(joined))
    (_, _), (width, height), _ = rectangle_box
    if max(width, height) < CUBOID_ASPECT_MIN * min(width, height):
        return False
    rectangle = np.zeros_like(joined)
    cv2.fillPoly(rectangle, [np.round(cv2.boxPoints(rectangle_box)).astype(np.int32)], 255)
    if np.count_nonzero(rectangle & hidden) < JOIN_COVER_MIN * np.count_nonzero(rectangle):
        return False
    visible = rectangle & ~hidden
    return np.count_nonzero(joined & visible) >= JOIN_FILL_MIN * np.count_nonzero(visible)


def _add_hidden_part(piece, hidden):
    """Gives a cuboid back the narrow strip that the ball in front of it hides.

    Seen from above, the ball can cover an edge or a corner of a block next to it.
    The rest of the block then has a round bite and can look like a prism. A cuboid is a
    rectangle from above, so the smallest rectangle around its visible part
    (cv2.minAreaRect) shows where it lies; the pixels of that
    rectangle under the ball are added to it again. Two cases are left as they are:
    a rectangle that is about square, because a prism is half of a square and its
    open half must stay open, and a missing part wider than HIDDEN_STRIP_MAX_PX.
    """
    if not np.any(cv2.dilate(piece, HIDDEN_KERNEL) & hidden):
        return piece
    rectangle_box = cv2.minAreaRect(outline_points(piece))
    (_, _), (width, height), _ = rectangle_box
    if max(width, height) < CUBOID_ASPECT_MIN * min(width, height):
        return piece
    corners = cv2.boxPoints(rectangle_box)
    rectangle = np.zeros_like(piece)
    cv2.fillPoly(rectangle, [np.round(corners).astype(np.int32)], 255)
    added = rectangle & hidden & ~piece
    # The distance transform gives half the width of the added part at its widest place.
    if 2 * cv2.distanceTransform(added, cv2.DIST_L2, 5).max() > HIDDEN_STRIP_MAX_PX:
        return piece
    return piece | added


def _edges(field_image, low, high, min_length_px=0):
    """Canny edges of the field with the thresholds low and high, made 3 px wide so that a thin border line closes.

    Every colour channel is searched on its own, because two neighbouring areas can be
    equally bright in the grey image and differ only in one channel (script ch. 4.1, 4.2).
    With min_length_px, connected lines that are shorter in x and y are left out (wood grain).
    """
    blurred = cv2.GaussianBlur(field_image, (5, 5), 0)
    edges = np.zeros(field_image.shape[:2], dtype=np.uint8)
    for channel in cv2.split(blurred):
        edges = edges | cv2.Canny(channel, low, high)
    if min_length_px:
        _, labels, stats, _ = cv2.connectedComponentsWithStats(edges)
        length = np.maximum(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT])
        long_enough = length >= min_length_px
        long_enough[0] = False  # label 0 is the background
        edges = np.where(long_enough[labels], 255, 0).astype(np.uint8)
    return cv2.dilate(edges, np.ones((3, 3), np.uint8))


def _longest_side(part):
    """Length of the longer side of the smallest rectangle around the area (cv2.minAreaRect)."""
    (_, _), (width, height), _ = cv2.minAreaRect(outline_points(part))
    return max(width, height)


def _edge_seeds(part, edges, field_image):
    """The top faces inside one area, found between the edges.

    Only large, wide and bright pieces count: a side face is a thin or dark strip
    and belongs to the top face next to it.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL_PX, MORPH_KERNEL_PX))
    inner = cv2.erode(cv2.bitwise_and(part, 255 - edges), kernel)
    brightness = cv2.cvtColor(field_image, cv2.COLOR_BGR2HSV)[:, :, 2]
    candidates = []
    for piece_mask in areas(inner):
        if np.count_nonzero(piece_mask) < EDGE_SEED_MIN_PX:
            continue
        piece = piece_mask > 0
        (_, _), (width, height), _ = cv2.minAreaRect(outline_points(piece_mask))
        if min(width, height) >= EDGE_SEED_MIN_WIDTH_PX:
            candidates.append((piece, np.median(brightness[piece])))
    seeds = np.zeros_like(part)
    brightest = max((value for _, value in candidates), default=0)
    for piece, value in candidates:
        if value >= SIDE_FACE_BRIGHTNESS_MAX * brightest:
            seeds[piece] = 255
    return seeds


def areas(mask):
    """One mask per connected white area, found as outer contours (cv2.findContours).

    Each contour is filled and cut with the original mask again, so holes of
    the area stay holes.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = []
    for contour in contours:
        filled = np.zeros_like(mask)
        cv2.drawContours(filled, [contour], -1, 255, thickness=cv2.FILLED)
        result.append(cv2.bitwise_and(filled, mask))
    return result


def outline_points(mask):
    """All points on the outer contours of a mask, e.g. as input for cv2.minAreaRect.

    The smallest rectangle around these points is the same as around all white pixels.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return np.vstack(contours)


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
    if len(areas(seeds)) < 2:
        return [part]
    return _watershed(part, seeds, cv2.cvtColor(part, cv2.COLOR_GRAY2BGR), min_area_px)


def _watershed(part, seeds, image, min_area_px):
    """Grows every seed until it meets its neighbour; returns one mask per seed.

    The background grows as well. A block whose top face gave no seed, e.g. a prism next
    to two cubes, can then be taken by the background instead of a seed. What the background
    took inside the area is therefore kept as an object of its own if it is large enough.
    """
    # Marker 1 is the background, the seeds are 2, 3, ... and 0 means "still unknown".
    markers = np.ones(part.shape, dtype=np.int32)
    markers[part == 255] = 0
    seed_masks = areas(seeds)
    for number, seed in enumerate(seed_masks, start=2):
        markers[seed == 255] = number
    cv2.watershed(image, markers)
    pieces = []
    for marker in range(2, len(seed_masks) + 2):
        piece = np.where(markers == marker, 255, 0).astype(np.uint8)
        if np.count_nonzero(piece) >= min_area_px:
            pieces.append(piece)
    lost = np.where((markers == 1) & (part == 255), 255, 0).astype(np.uint8)
    for piece in areas(clean_mask(lost)):
        if np.count_nonzero(piece) >= min_area_px and 2 * cv2.distanceTransform(piece, cv2.DIST_L2, 5).max() >= LOST_MIN_WIDTH_PX:
            pieces.append(piece)
    return pieces
