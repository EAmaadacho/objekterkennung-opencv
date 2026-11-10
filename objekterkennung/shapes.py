"""Recognises the shape of one object from its outline seen from above (A1)."""

import math

import cv2

# How much of its smallest surrounding rectangle a shape fills:
# square or rectangle about 1.0, circle pi/4 = 0.79, any triangle exactly 0.5.
# A real prism has bevelled edges and a blurred tip, so it reaches up to about 0.7.
BOX_FILL_MIN = 0.86
TRIANGLE_FILL_MAX = 0.72
# 4*pi*area/perimeter^2 is 1.0 for a circle and 0.79 for a square.
CIRCLE_ROUNDNESS_MIN = 0.80
# A cube is about 1:1 from above, a cuboid about 2:1.
CUBOID_ASPECT_MIN = 1.5


def classify_shape(object_mask):
    """Returns "cube", "cuboid", "prism", "sphere" or "unknown" for one object mask.

    The convex hull is used instead of the raw outline, so a small dent from a
    reflection or a visible side face does not change the result.
    """
    contours, _ = cv2.findContours(object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return "unknown"
    hull = cv2.convexHull(max(contours, key=cv2.contourArea))
    area = cv2.contourArea(hull)
    (_, _), (width, height), _ = cv2.minAreaRect(hull)
    if area == 0 or width == 0 or height == 0:
        return "unknown"
    fill = area / (width * height)
    perimeter = cv2.arcLength(hull, closed=True)
    roundness = 4 * math.pi * area / perimeter**2
    if fill >= BOX_FILL_MIN:
        aspect = max(width, height) / min(width, height)
        return "cuboid" if aspect >= CUBOID_ASPECT_MIN else "cube"
    if fill <= TRIANGLE_FILL_MAX:
        return "prism"
    if roundness >= CIRCLE_ROUNDNESS_MIN:
        return "sphere"
    return "unknown"


def center_px(object_mask):
    """Centre of mass of the object in pixels of the straight field, from the image moments."""
    moments = cv2.moments(object_mask, binaryImage=True)
    return moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]
