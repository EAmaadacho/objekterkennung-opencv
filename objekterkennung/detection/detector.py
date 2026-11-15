"""Finds all objects in the straight field image: colour, shape and position (A1, A2, A3, A5).

This is where the detection starts. It calls segmentation.py (colour masks, touching
objects), prism_pair.py (two prisms that form a square) and shapes.py (the shape).
"""

import cv2

from objekterkennung.capture.board import pixel_to_unit
from objekterkennung.detection.prism_pair import split_prism_pair
from objekterkennung.detection.segmentation import color_mask, outline_points, split_touching
from objekterkennung.detection.shapes import center_px, classify_shape
from objekterkennung.output.models import DetectedObject

# Two prisms as a square can carry a thin side face of the block next to them and then measure
# just over the cube limit, e.g. 1.51 : 1 (test 15.11.). Up to this ratio a "cuboid" is still
# checked for a joint; a real cuboid is 2.4 : 1.
PAIR_ASPECT_MAX = 1.7


def detect_objects(field_image, settings):
    """Returns every object on the field, e.g. [DetectedObject("cube", "red", 0.42, 0.71)].

    Each colour is searched on its own, so two touching objects of different
    colours are separate right away. Touching objects of the same colour are
    split by split_touching. The image includes a small margin around the
    field, so a block at the edge is seen whole.
    """
    objects = []
    for color in settings.colors.hue_ranges:
        mask = color_mask(field_image, color, settings)
        for object_mask in _split_all(mask, color, field_image, settings):
            x, y = pixel_to_unit(*center_px(object_mask), settings.board)
            # Something whose centre lies in the margin, e.g. a finger, is not on the field.
            if not (0 <= x <= 1 and 0 <= y <= 1):
                continue
            objects.append(DetectedObject(classify_shape(object_mask), color, x, y))
    # Sorted from top-left to bottom-right, so the console list is easy to compare with the field.
    return sorted(objects, key=lambda item: (-round(item.y, 1), item.x))


def _split_all(mask, color, field_image, settings):
    """All object masks of one colour; a square is checked once more for being two prisms.

    Only colours with two prisms (config.json, "objects": "prism_pair_colors") can form
    such a square; for all others a dark line across a cube is never a joint. In the
    current set every prism exists once, so the list is empty and no square is checked.
    """
    for object_mask in split_touching(mask, settings, field_image):
        if color in settings.objects.prism_pair_colors and _about_square(object_mask):
            yield from split_prism_pair(object_mask, field_image)
        else:
            yield object_mask


def _about_square(object_mask):
    """True for a "cube" and for a "cuboid" that is shorter than PAIR_ASPECT_MAX : 1."""
    shape = classify_shape(object_mask)
    if shape == "cube":
        return True
    if shape != "cuboid":
        return False
    (_, _), (width, height), _ = cv2.minAreaRect(outline_points(object_mask))
    return max(width, height) < PAIR_ASPECT_MAX * min(width, height)
