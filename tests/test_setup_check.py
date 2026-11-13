"""Tests for the setup mode; the board is drawn with cv2, no camera is needed."""

import cv2
import numpy as np

from objekterkennung.board import ARUCO_DICTIONARY
from objekterkennung.config import load_settings
from objekterkennung.setup_check import (
    BAD,
    OK,
    START_BUTTON,
    WARN,
    WINDOW_MAX_WIDTH,
    check_light,
    check_piece_glare,
    check_setup,
    draw_setup,
    is_start_click,
)

PX_PER_MM = 3
MARGIN_MM = 50
# Grey paper like under the lamp; pure white 255 would count as glare.
PAPER = 180


def test_A6_straight_centred_board_is_all_green():
    board = load_settings().board
    frame = _draw_board(board)
    # One block in the field, so the glare check has something to measure.
    cv2.rectangle(frame, (400, 400), (480, 480), (40, 30, 210), -1)

    checks, corners = check_setup(frame, board)

    assert [check.level for check in checks] == [OK, OK, OK, OK, OK, OK]
    assert corners is not None


def test_A6_small_board_at_the_side_asks_to_move_the_camera():
    board = load_settings().board
    small = cv2.resize(_draw_board(board), (300, 300))
    frame = np.full((1080, 1920, 3), PAPER, dtype=np.uint8)
    frame[100:400, 100:400] = small

    checks, _ = check_setup(frame, board)

    size, centre = checks[1], checks[3]
    assert size.level == BAD and "lower" in size.text
    assert centre.level == BAD and "left" in centre.text


def test_A6_covered_marker_is_named():
    board = load_settings().board
    frame = _draw_board(board)
    frame[700:, 600:] = PAPER

    checks, corners = check_setup(frame, board)

    assert checks[0].level == BAD and "missing: 2" in checks[0].text
    assert corners is None


def test_A6_tilted_camera_is_found():
    board = load_settings().board
    frame = _draw_board(board)
    height, width = frame.shape[:2]
    source = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    target = np.float32([[90, 0], [width - 90, 0], [width, height], [0, height]])
    tilted = cv2.warpPerspective(frame, cv2.getPerspectiveTransform(source, target), (width, height), borderValue=(PAPER,) * 3)

    checks, _ = check_setup(tilted, board)

    assert checks[2].level in (WARN, BAD) and "top side" in checks[2].text and "to the bottom" in checks[2].text


def test_A6_paper_turned_by_90_degrees_counts_as_straight():
    board = load_settings().board
    frame = np.ascontiguousarray(np.rot90(_draw_board(board)))

    checks, _ = check_setup(frame, board)

    assert checks[2].level == OK


def test_A6_slightly_turned_paper_gets_a_turn_hint():
    board = load_settings().board
    frame = _draw_board(board)
    center = (frame.shape[1] / 2, frame.shape[0] / 2)
    # A positive angle turns the image anticlockwise on the screen, so the hint must say clockwise.
    turned = cv2.warpAffine(frame, cv2.getRotationMatrix2D(center, 5, 1.0), frame.shape[1::-1], borderValue=(PAPER,) * 3)

    checks, _ = check_setup(turned, board)

    assert checks[2].level == WARN and "5 deg" in checks[2].text and "turn paper clockwise" in checks[2].text


def test_A7_dark_paper_is_too_dark():
    field = np.full((640, 640, 3), 70, dtype=np.uint8)

    check = check_light(field)

    assert check.level == BAD and "too dark" in check.text


def test_A7_shadow_in_one_corner_is_uneven():
    field = np.full((640, 640, 3), 180, dtype=np.uint8)
    field[:213, :213] = 100

    check = check_light(field)

    assert check.level == BAD and "uneven" in check.text


def test_A7_glare_on_a_block_is_found():
    field = np.full((700, 700, 3), 180, dtype=np.uint8)
    cv2.rectangle(field, (300, 300), (400, 400), (200, 90, 30), -1)
    cv2.circle(field, (350, 350), 15, (255, 255, 255), -1)

    check = check_piece_glare(field)

    assert check.level == BAD and "glare on blocks" in check.text


def test_A7_empty_field_asks_for_blocks():
    field = np.full((700, 700, 3), 180, dtype=np.uint8)

    check = check_piece_glare(field)

    assert check.level == WARN and "put some blocks" in check.text


def test_A6_setup_view_fits_the_screen_and_has_a_start_button():
    board = load_settings().board
    frame = np.full((1080, 1920, 3), PAPER, dtype=np.uint8)
    checks, corners = check_setup(frame, board)

    view = draw_setup(frame, checks, corners)
    left, top, right, bottom = START_BUTTON

    assert view.shape[1] <= WINDOW_MAX_WIDTH
    assert is_start_click((left + right) // 2, (top + bottom) // 2)
    assert not is_start_click(right + 10, top)


def _draw_board(board):
    """Draws the printed page seen straight from above: grey paper and the four markers."""
    size_mm = board.field_mm + 2 * MARGIN_MM
    image = np.full((size_mm * PX_PER_MM, size_mm * PX_PER_MM, 3), PAPER, dtype=np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICTIONARY)
    marker_px = int(board.marker_mm * PX_PER_MM)
    for marker_id, (center_x, center_y) in board.marker_centers_mm().items():
        marker = cv2.aruco.generateImageMarker(dictionary, marker_id, marker_px)
        left = int((MARGIN_MM + center_x - board.marker_mm / 2) * PX_PER_MM)
        top = int((MARGIN_MM + center_y - board.marker_mm / 2) * PX_PER_MM)
        image[top : top + marker_px, left : left + marker_px] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
    return image
