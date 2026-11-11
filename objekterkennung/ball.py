"""Cuts a ball out of an area where it touches a block of the same colour (A1, A5)."""

import cv2
import numpy as np

# The balls are 45 mm wide, i.e. a radius of about 75 px in the 600 px field (180 mm).
# The range allows for the camera height: the top of the ball is closer and looks bigger.
BALL_RADIUS_MIN_PX = 65
BALL_RADIUS_MAX_PX = 90
# Share of the circle along which the outline of the area really runs. A ball that
# touches one or two blocks still shows most of its round outline (0.74 to 0.86 in
# the test recordings); circles found in blocks reached at most 0.47.
OUTLINE_MATCH_MIN = 0.6
# How far the outline may lie from the circle, in pixels.
OUTLINE_TOLERANCE_PX = 4
# Pixels around the ball that are given to neither object, so no thin seam is left over.
CUT_MARGIN_PX = 4
# HoughCircles only finds the circle roughly. Centre and radius are then moved step by
# step to where the outline fits best.
REFINE_GRID_PX = 6
REFINE_GRID_STEP_PX = 3
ANGLES = np.linspace(0, 2 * np.pi, 360, endpoint=False)


def cut_ball(part):
    """Returns (ball mask, rest of the area); (None, part) if there is no ball in the area.

    cv2.HoughCircles looks for circles of ball size along the edge of the mask.
    A circle only counts as a ball if the outline of the area follows it for most
    of its length, because a circle can also be fitted into the corner of a cube.
    """
    blurred = cv2.GaussianBlur(part, (9, 9), 2)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=2, minDist=120, param1=100, param2=30,
        minRadius=BALL_RADIUS_MIN_PX, maxRadius=BALL_RADIUS_MAX_PX,
    )
    if circles is None:
        return None, part
    for x, y, radius in circles[0]:
        if outline_match(part, x, y, radius) < OUTLINE_MATCH_MIN:
            x, y, radius = refine_circle(part, x, y, radius)
        if outline_match(part, x, y, radius) >= OUTLINE_MATCH_MIN:
            disc = np.zeros_like(part)
            cv2.circle(disc, (round(x), round(y)), round(radius), 255, -1)
            ball = cv2.bitwise_and(part, disc)
            cv2.circle(disc, (round(x), round(y)), round(radius) + CUT_MARGIN_PX, 255, -1)
            rest = cv2.bitwise_and(part, 255 - disc)
            return ball, rest
    return None, part


def refine_circle(part, x, y, radius):
    """Moves centre and radius to where the outline follows the circle best.

    First a coarse grid in steps of REFINE_GRID_STEP_PX, then 1 px steps from the
    best grid point for as long as the match still gets better.
    """
    grid = range(-REFINE_GRID_PX, REFINE_GRID_PX + 1, REFINE_GRID_STEP_PX)
    best = max(
        (outline_match(part, x + dx, y + dy, radius + dr), x + dx, y + dy, radius + dr)
        for dx in grid for dy in grid for dr in grid
    )
    while True:
        match, bx, by, br = best
        neighbours = [(bx + 1, by, br), (bx - 1, by, br), (bx, by + 1, br),
                      (bx, by - 1, br), (bx, by, br + 1), (bx, by, br - 1)]
        better = max((outline_match(part, *c), *c) for c in neighbours)
        if better[0] <= match:
            return bx, by, br
        best = better


def outline_match(part, x, y, radius):
    """Share of the circle where the area is inside just within it and outside just beyond it."""
    inside = _on_circle(part, x, y, radius - OUTLINE_TOLERANCE_PX)
    outside = ~_on_circle(part, x, y, radius + OUTLINE_TOLERANCE_PX)
    return float((inside & outside).mean())


def _on_circle(part, x, y, radius):
    """For 360 points on the circle: True where the mask is set."""
    xs = np.clip((x + radius * np.cos(ANGLES)).astype(int), 0, part.shape[1] - 1)
    ys = np.clip((y + radius * np.sin(ANGLES)).astype(int), 0, part.shape[0] - 1)
    return part[ys, xs] > 0
