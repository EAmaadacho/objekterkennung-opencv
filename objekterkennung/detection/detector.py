"""Finds all objects in the straight field image: colour, shape and position (A1, A2, A3, A5)."""

from objekterkennung.capture.board import pixel_to_unit
from objekterkennung.output.models import DetectedObject
from objekterkennung.detection.prism_pair import split_prism_pair
from objekterkennung.detection.segmentation import color_mask, split_touching
from objekterkennung.detection.shapes import center_px, classify_shape


def detect_objects(field_image, settings):
    """Returns every object on the field, e.g. [DetectedObject("cube", "red", 0.42, 0.71)].

    Each colour is searched on its own, so two touching objects of different
    colours are separate right away. Touching objects of the same colour are
    split by split_touching. The image includes a small margin around the
    field, so a block at the edge is seen whole.
    """
    objects = []
    for color in settings.hue_ranges:
        mask = color_mask(field_image, color, settings)
        for object_mask in _split_all(mask, field_image, settings):
            x, y = pixel_to_unit(*center_px(object_mask), settings.board)
            # Something whose centre lies in the margin, e.g. a finger, is not on the field.
            if not (0 <= x <= 1 and 0 <= y <= 1):
                continue
            objects.append(DetectedObject(classify_shape(object_mask), color, x, y))
    # Sorted from top-left to bottom-right, so the console list is easy to compare with the field.
    return sorted(objects, key=lambda item: (-round(item.y, 1), item.x))


def _split_all(mask, field_image, settings):
    """All object masks of one colour; a "cube" is checked once more for being two prisms."""
    for object_mask in split_touching(mask, settings.min_area_px, field_image):
        if classify_shape(object_mask) == "cube":
            yield from split_prism_pair(object_mask, field_image)
        else:
            yield object_mask
