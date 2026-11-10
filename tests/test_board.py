"""Tests for finding the board and straightening the field; the board is drawn with cv2."""

import cv2
import numpy as np

from objekterkennung.board import (
    ARUCO_DICTIONARY,
    find_homography,
    find_markers,
    pixel_to_unit,
    straighten_field,
    unit_to_pixel,
)
from objekterkennung.config import load_settings

PX_PER_MM = 3
MARGIN_MM = 50
RED = (0, 0, 255)
WHITE = 255


def test_A3_finds_all_four_markers():
    board = load_settings().board
    image = _draw_board(board)

    markers = find_markers(image)

    assert sorted(markers) == [0, 1, 2, 3]


def test_A3_tilted_board_is_straightened():
    # A red dot at (0.25 | 0.75) must land there again after straightening.
    board = load_settings().board
    image = _tilt(_draw_board(board, red_dot_at=(0.25, 0.75)))

    homography = find_homography(image, board)
    field = straighten_field(image, homography, board)
    x, y = pixel_to_unit(*_red_center(field), board)

    assert abs(x - 0.25) < 0.01
    assert abs(y - 0.75) < 0.01


def test_A3_covered_marker_keeps_last_homography():
    board = load_settings().board
    last = np.eye(3)
    image = _draw_board(board)
    cv2.rectangle(image, (0, 0), (image.shape[1] // 2, 400), (128, 128, 128), -1)

    homography = find_homography(image, board, last_homography=last)

    assert homography is last


def test_A3_no_markers_and_no_history_gives_none():
    board = load_settings().board
    empty = np.full((480, 640, 3), WHITE, dtype=np.uint8)

    homography = find_homography(empty, board)

    assert homography is None


def test_A3_pixel_to_unit_flips_y():
    board = load_settings().board

    x, y = pixel_to_unit(320, 170, board)

    assert (x, y) == (0.5, 0.75)


def test_A3_unit_to_pixel_is_the_way_back():
    board = load_settings().board

    pixel = unit_to_pixel(0.5, 0.75, board)

    assert pixel == (320, 170)


def _draw_board(board, red_dot_at=None):
    """Draws the board like the printed page: white paper, four markers around the field."""
    size_mm = board.field_mm + 2 * MARGIN_MM
    image = np.full((size_mm * PX_PER_MM, size_mm * PX_PER_MM, 3), WHITE, dtype=np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICTIONARY)
    marker_px = int(board.marker_mm * PX_PER_MM)
    for marker_id, (center_x, center_y) in board.marker_centers_mm().items():
        marker = cv2.aruco.generateImageMarker(dictionary, marker_id, marker_px)
        left = int((MARGIN_MM + center_x - board.marker_mm / 2) * PX_PER_MM)
        top = int((MARGIN_MM + center_y - board.marker_mm / 2) * PX_PER_MM)
        image[top : top + marker_px, left : left + marker_px] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
    if red_dot_at is not None:
        x, y = red_dot_at
        center = (
            int((MARGIN_MM + x * board.field_mm) * PX_PER_MM),
            int((MARGIN_MM + (1 - y) * board.field_mm) * PX_PER_MM),
        )
        cv2.circle(image, center, 15, RED, -1)
    return image


def _tilt(image):
    """Simulates a camera that looks at the paper from slightly to the side."""
    height, width = image.shape[:2]
    source = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    target = np.float32([[40, 10], [width - 10, 60], [width - 60, height - 20], [10, height - 50]])
    matrix = cv2.getPerspectiveTransform(source, target)
    return cv2.warpPerspective(image, matrix, (width, height), borderValue=(WHITE, WHITE, WHITE))


def _red_center(image):
    """Pixel centre of all strongly red pixels."""
    mask = cv2.inRange(image, (0, 0, 200), (60, 60, 255))
    moments = cv2.moments(mask)
    return moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]
