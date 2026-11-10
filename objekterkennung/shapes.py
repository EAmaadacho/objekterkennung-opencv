"""Recognises the shape of one object from its outline seen from above (A1)."""

import math

import cv2

# How much of its smallest surrounding rectangle a shape fills:
# square or rectangle about 1.0, any triangle exactly 0.5. The limit lies in between,
# because shading and bevelled edges make real objects a bit worse than the ideal.
BOX_FILL_MIN = 0.75
# How much of its smallest surrounding circle a shape fills:
# circle 1.0, square 0.64, cuboid 0.51, triangle at most 0.41.
CIRCLE_FILL_MIN = 0.80
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
    (_, _), radius = cv2.minEnclosingCircle(hull)
    if area / (math.pi * radius**2) >= CIRCLE_FILL_MIN:
        return "sphere"
    if area / (width * height) >= BOX_FILL_MIN:
        aspect = max(width, height) / min(width, height)
        return "cuboid" if aspect >= CUBOID_ASPECT_MIN else "cube"
    return "prism"


def center_px(object_mask):
    """Centre of mass of the object in pixels of the straight field, from the image moments."""
    moments = cv2.moments(object_mask, binaryImage=True)
    return moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]
