"""Finds the four ArUco markers and straightens the field into a square image (A3)."""

import cv2
import numpy as np

# The same marker family that tools/make_board.py prints on the board.
ARUCO_DICTIONARY = cv2.aruco.DICT_4X4_50
MARKER_IDS = (0, 1, 2, 3)


def find_markers(image):
    """Returns the four pixel corners of every visible marker, e.g. {0: array of 4 (x, y) points}.

    The corners of each marker come in the order top-left, top-right,
    bottom-right, bottom-left, as printed on the paper.
    """
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICTIONARY)
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(image)
    if ids is None:
        return {}
    return {int(marker_id): marker_corners.reshape(4, 2) for marker_id, marker_corners in zip(ids.flatten(), corners)}


def find_homography(image, board, last_homography=None):
    """Computes the transformation from the camera image to the straight field image.

    Only a view with all four markers gives a new homography. If a hand covers
    a marker, the last valid homography is used again, e.g. while the lecturer
    places a block. Without any valid homography so far the result is None.
    """
    markers = find_markers(image)
    if not all(marker_id in markers for marker_id in MARKER_IDS):
        return last_homography
    image_points = np.concatenate([markers[marker_id] for marker_id in MARKER_IDS])
    field_points = np.concatenate([_marker_corners_px(board, marker_id) for marker_id in MARKER_IDS])
    homography, _ = cv2.findHomography(image_points, field_points)
    return homography


def straighten_field(image, homography, board):
    """Cuts out the field with its margin and turns it into a square, e.g. 700 x 700 pixels for field_size_in_image_px 600."""
    size = field_image_size(board)
    return cv2.warpPerspective(image, homography, (size, size))


def upright_turns(homography, board):
    """How many quarter turns (np.rot90) make the camera image show the field upright, i.e. 0, 1, 2 or 3.

    Upright means as seen from the front of the field: (0,0) bottom left, x to the right.
    The bottom edge of the field is mapped back into the camera image with the inverse
    homography; its direction says how the camera is turned against the field. Only
    quarter turns are used, because they keep every pixel (no new pixels are calculated);
    a rest of up to 45 degrees is left to the setup mode ("turn paper").
    """
    margin = margin_px(board)
    far = margin + board.field_size_in_image_px
    bottom_edge = np.float32([[margin, far], [far, far]]).reshape(-1, 1, 2)
    left, right = cv2.perspectiveTransform(bottom_edge, np.linalg.inv(homography)).reshape(2, 2)
    # The y axis of an image points down, so an edge pointing down has an angle of +90 degrees.
    angle = np.degrees(np.arctan2(right[1] - left[1], right[0] - left[0]))
    return round(angle / 90) % 4


def turn_upright(image, turns):
    """Turns an image by quarter turns counter-clockwise (np.rot90), e.g. turns from upright_turns."""
    return np.ascontiguousarray(np.rot90(image, turns))


def margin_px(board):
    """Width of the paper margin around the field in the straight image, e.g. 50 px for margin_around_field_mm 15.

    The straight image shows some paper around the field as well, so a block at the edge is
    still seen whole: seen from the camera, the top of a high block lies a bit further out than
    its base. With 15 mm the margin reaches 7 mm into the markers above and below. They are black
    and white, so no colour mask finds them, and their strip is far smaller than a hand.
    """
    return round(board.margin_around_field_mm * board.field_size_in_image_px / board.field_size_mm)


def field_image_size(board):
    """Side length of the straight image: the field plus the margin on both sides, e.g. 700 px."""
    return board.field_size_in_image_px + 2 * margin_px(board)


def pixel_to_unit(pixel_x, pixel_y, board):
    """Converts a pixel of the straight image into unit-square coordinates.

    The margin is taken off first. The image y axis points down, the field y axis
    points up, so y is flipped: pixel (350, 200) with a 50 px margin is (0.5, 0.75).
    """
    size = board.field_size_in_image_px
    margin = margin_px(board)
    return (pixel_x - margin) / size, 1 - (pixel_y - margin) / size


def unit_to_pixel(x, y, board):
    """The way back from unit-square coordinates to a pixel of the straight image.

    (0.5, 0.75) is pixel (350, 200) for a 600 px field with a 50 px margin.
    """
    size = board.field_size_in_image_px
    margin = margin_px(board)
    return round(x * size) + margin, round((1 - y) * size) + margin


def _marker_corners_px(board, marker_id):
    """Where the four corners of a marker lie in the straight field image, in pixels.

    Markers lie outside the field, so some values are negative or larger than
    the image. That is fine: they only serve as reference points.
    """
    scale = board.field_size_in_image_px / board.field_size_mm
    shift_mm = board.margin_around_field_mm
    center_x, center_y = board.marker_centers_mm()[marker_id]
    half = board.marker_size_mm / 2
    corners_mm = [
        (center_x - half, center_y - half),
        (center_x + half, center_y - half),
        (center_x + half, center_y + half),
        (center_x - half, center_y + half),
    ]
    return (np.array(corners_mm, dtype=np.float32) + shift_mm) * scale
