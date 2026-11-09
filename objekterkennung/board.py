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
    """Cuts out the field and turns it into a square, e.g. 600 x 600 pixels for warp_px 600."""
    size = board.warp_px
    return cv2.warpPerspective(image, homography, (size, size))


def pixel_to_unit(pixel_x, pixel_y, board):
    """Converts a pixel of the straight field into unit-square coordinates.

    The image y axis points down, the field y axis points up, so y is flipped:
    pixel (300, 150) in a 600 px field is (0.5, 0.75).
    """
    size = board.warp_px
    return pixel_x / size, 1 - pixel_y / size


def _marker_corners_px(board, marker_id):
    """Where the four corners of a marker lie in the straight field image, in pixels.

    Markers lie outside the field, so some values are negative or larger than
    warp_px. That is fine: they only serve as reference points.
    """
    scale = board.warp_px / board.field_mm
    center_x, center_y = board.marker_centers_mm()[marker_id]
    half = board.marker_mm / 2
    corners_mm = [
        (center_x - half, center_y - half),
        (center_x + half, center_y - half),
        (center_x + half, center_y + half),
        (center_x - half, center_y + half),
    ]
    return np.array(corners_mm, dtype=np.float32) * scale
