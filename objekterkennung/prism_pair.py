"""Splits two triangle prisms that lie together as a square, which from above looks like a cube (A1, A5)."""

import cv2
import numpy as np

from objekterkennung.segmentation import outline_points
from objekterkennung.shapes import classify_shape

# The joint between the two prisms runs along a diagonal of the square, i.e. about
# 45 degrees to its sides. A few degrees of play allow for an imperfect fit.
DIAGONAL_ANGLES = (43, 45, 47)
# How far the joint may lie beside the centre of the square, in pixels.
MAX_OFFSET_PX = 7
# The joint is compared with the strips this far away on both sides.
NEIGHBOUR_PX = 6
# The joint must be at least this much darker (grey levels 0-255) than its
# neighbours. In the test recordings joints were 9.6 to 56 darker, the darkest
# line through the middle of a real cube only 7.6.
JOINT_CONTRAST_MIN = 9
# Rows with fewer object pixels are ignored, e.g. the tips of the rotated square.
MIN_ROW_PX = 30
# The border of the object is left out, because it is dark from the side faces.
BORDER_PX = 11


def split_prism_pair(object_mask, field_image):
    """Returns two masks if the "cube" is really two prisms with a dark joint on its diagonal, else [object_mask]."""
    joint = find_joint(object_mask, field_image)
    if joint is None:
        return [object_mask]
    halves = _cut_along(object_mask, *joint)
    # Both halves must look like triangles, otherwise the line was something else.
    if all(classify_shape(half) == "prism" for half in halves):
        return halves
    return [object_mask]


def find_joint(object_mask, field_image):
    """Looks for a dark straight line through the middle along a diagonal.

    Returns (centre, angle in degrees, offset in pixels) of the line, or None.
    The image is turned around the centre of the object so that the tested line
    lies horizontal; then the mean brightness of every row is one value of a profile.
    """
    x, y, width, height = cv2.boundingRect(object_mask)
    gray = cv2.cvtColor(field_image[y : y + height, x : x + width], cv2.COLOR_BGR2GRAY).astype(np.float32)
    inner = cv2.erode(object_mask[y : y + height, x : x + width], np.ones((BORDER_PX, BORDER_PX), np.uint8))
    moments = cv2.moments(inner, binaryImage=True)
    if moments["m00"] == 0:
        return None
    center = (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])
    _, _, side_angle = cv2.minAreaRect(outline_points(object_mask))
    best = None
    for diagonal in (45, 135):
        for delta in DIAGONAL_ANGLES:
            angle = side_angle + diagonal + delta - 45
            contrast, offset = _darkest_row(gray, inner, center, angle)
            if contrast >= JOINT_CONTRAST_MIN and (best is None or contrast > best[0]):
                best = (contrast, angle, offset)
    if best is None:
        return None
    return (center[0] + x, center[1] + y), best[1], best[2]


def _darkest_row(gray, inner, center, angle):
    """Turns the crop by angle and returns (contrast, offset) of the darkest row near the centre."""
    rotation = cv2.getRotationMatrix2D(center, angle, 1.0)
    size = (gray.shape[1], gray.shape[0])
    turned = cv2.warpAffine(gray, rotation, size)
    turned_mask = cv2.warpAffine(inner, rotation, size, flags=cv2.INTER_NEAREST)
    middle = int(round(center[1]))
    best = (0.0, 0)
    for offset in range(-MAX_OFFSET_PX, MAX_OFFSET_PX + 1):
        rows = [middle + offset - NEIGHBOUR_PX, middle + offset, middle + offset + NEIGHBOUR_PX]
        means = [_row_mean(turned, turned_mask, row) for row in rows]
        if None in means:
            continue
        contrast = (means[0] + means[2]) / 2 - means[1]
        if contrast > best[0]:
            best = (contrast, offset)
    return best


def _row_mean(image, mask, row):
    """Mean brightness of the object pixels in one row, or None if there are too few."""
    if not 0 <= row < image.shape[0]:
        return None
    pixels = image[row][mask[row] > 0]
    return float(pixels.mean()) if len(pixels) >= MIN_ROW_PX else None


def _cut_along(object_mask, center, angle, offset):
    """Cuts the mask into the parts on both sides of the line."""
    rows, cols = np.indices(object_mask.shape)
    # The row direction of the turned image, seen in the original image.
    radians = np.radians(angle)
    normal_x, normal_y = -np.sin(radians), np.cos(radians)
    side = (cols - center[0]) * normal_x + (rows - center[1]) * normal_y - offset
    first = np.where((object_mask > 0) & (side < 0), 255, 0).astype(np.uint8)
    second = np.where((object_mask > 0) & (side >= 0), 255, 0).astype(np.uint8)
    return [first, second]
