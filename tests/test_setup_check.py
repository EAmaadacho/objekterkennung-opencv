"""Tests for the setup mode; the board is drawn with cv2, no camera is needed."""

import cv2
import numpy as np

from objekterkennung.capture.board import ARUCO_DICTIONARY
from objekterkennung.capture.setup_check import (
    BAD,
    GUIDES_BUTTON,
    OK,
    PANEL_WIDTH,
    START_BUTTON,
    WARN,
    WINDOW_MAX_WIDTH,
    camera_lean,
    check_blocks,
    check_centre,
    check_light,
    check_setup,
    check_size,
    check_straight,
    draw_setup,
    field_centre_lines,
    is_guides_click,
    is_start_click,
)
from objekterkennung.config import load_settings
from objekterkennung.output.models import DetectedObject

PX_PER_MM = 3
MARGIN_MM = 50
# Grey paper like under the lamp; pure white 255 would count as glare.
PAPER = 162


def test_A6_straight_centred_board_has_green_camera_and_light():
    settings = load_settings()
    board = settings.board
    frame = _draw_board(board)
    # One block in the field, like in a real setup.
    cv2.rectangle(frame, (400, 400), (480, 480), (40, 30, 210), -1)

    checks, corners = check_setup(frame, settings)

    assert [check.level for check in checks[:5]] == [OK, OK, OK, OK, OK]
    assert checks[5].name == "Blocks" and checks[5].level == BAD
    assert corners is not None


def test_A1_all_sixteen_cuboids_of_the_test_pattern_found_is_green():
    check = check_blocks(_test_pattern())

    assert check.level == OK and check.text == "16 of 16 found"


def test_A1_missing_red_cuboid_is_named():
    objects = _test_pattern()
    objects.remove(DetectedObject("cuboid", "red", 0.5, 0.5))

    check = check_blocks(objects)

    assert check.level == WARN and check.text == "15 of 16 found - missing: red cuboid"


def test_A1_cuboid_found_as_two_cubes_is_named():
    objects = _test_pattern()
    objects.remove(DetectedObject("cuboid", "green", 0.5, 0.5))
    objects += [DetectedObject("cube", "green", 0.4, 0.5), DetectedObject("cube", "green", 0.6, 0.5)]

    check = check_blocks(objects)

    assert check.level == BAD and check.text == "15 of 16 found - missing: green cuboid, extra: 2x green cube"


def test_A1_empty_field_is_red():
    check = check_blocks([])

    assert check.level == BAD and "0 of 16" in check.text


def _test_pattern():
    """The 16 cuboids of the setup: 4 of every colour."""
    return [DetectedObject("cuboid", color, 0.5, 0.5) for color in ("red", "yellow", "green", "blue") for _ in range(4)]


def test_A6_small_board_at_the_side_asks_to_move_the_camera():
    settings = load_settings()
    board = settings.board
    small = cv2.resize(_draw_board(board), (300, 300))
    frame = np.full((1080, 1920, 3), PAPER, dtype=np.uint8)
    frame[100:400, 100:400] = small

    checks, _ = check_setup(frame, settings)

    size, centre = checks[1], checks[3]
    assert size.level == BAD and "lower" in size.text
    assert centre.level == BAD and "left" in centre.text


def test_A6_covered_marker_is_named():
    settings = load_settings()
    board = settings.board
    frame = _draw_board(board)
    frame[700:, 600:] = PAPER

    checks, corners = check_setup(frame, settings)

    assert checks[0].level == BAD and "missing: 2" in checks[0].text
    assert corners is None


def test_A6_tilted_camera_is_found():
    settings = load_settings()
    board = settings.board
    frame = _draw_board(board)
    height, width = frame.shape[:2]
    source = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    target = np.float32([[90, 0], [width - 90, 0], [width, height], [0, height]])
    tilted = cv2.warpPerspective(frame, cv2.getPerspectiveTransform(source, target), (width, height), borderValue=(PAPER,) * 3)

    checks, _ = check_setup(tilted, settings)

    assert checks[2].level in (WARN, BAD) and "top side" in checks[2].text and "to the bottom" in checks[2].text


def test_A6_spirit_level_points_to_the_shorter_side():
    corners = np.float32([[110, 100], [190, 100], [200, 200], [100, 200]])

    lean_x, lean_y = camera_lean(corners)

    assert lean_x == 0
    assert lean_y < 0


def test_A6_paper_turned_by_90_degrees_counts_as_straight():
    settings = load_settings()
    board = settings.board
    frame = np.ascontiguousarray(np.rot90(_draw_board(board)))

    checks, _ = check_setup(frame, settings)

    assert checks[2].level == OK


def test_A6_slightly_turned_paper_gets_a_turn_hint():
    settings = load_settings()
    board = settings.board
    frame = _draw_board(board)
    center = (frame.shape[1] / 2, frame.shape[0] / 2)
    # A positive angle turns the image anticlockwise on the screen, so the hint must say clockwise.
    turned = cv2.warpAffine(frame, cv2.getRotationMatrix2D(center, 5, 1.0), frame.shape[1::-1], borderValue=(PAPER,) * 3)

    checks, _ = check_setup(turned, settings)

    assert checks[2].level == WARN and "5 deg" in checks[2].text and "turn paper clockwise" in checks[2].text


def test_A7_dark_paper_is_too_dark():
    field = np.full((640, 640, 3), 70, dtype=np.uint8)

    check = check_light(field, load_settings().light)

    assert check.level == BAD and "too dark" in check.text


def test_A7_paper_in_the_target_band_is_ok():
    field = np.full((640, 640, 3), 162, dtype=np.uint8)

    check = check_light(field, load_settings().light)

    assert check.level == OK and "paper 162" in check.text


def test_A7_slightly_too_bright_paper_asks_for_one_step_darker():
    field = np.full((640, 640, 3), 180, dtype=np.uint8)

    check = check_light(field, load_settings().light)

    assert check.level == WARN and "one step darker" in check.text


def test_A7_slightly_uneven_light_still_counts_as_good():
    # 70 % evenness: in the recordings such light caused no error.
    field = np.full((640, 640, 3), 165, dtype=np.uint8)
    field[:213, :213] = 116

    check = check_light(field, load_settings().light)

    assert check.level == OK


def test_A6_field_filling_36_percent_of_the_image_is_good():
    # The test setup reached 36-38 % of the image height and found 97 % of the blocks.
    corners = np.float32([[800, 346], [1189, 346], [1189, 735], [800, 735]])

    check = check_size(corners, (1080, 1920, 3))

    assert check.level == OK


def test_A7_shadow_in_one_corner_is_uneven():
    field = np.full((640, 640, 3), 180, dtype=np.uint8)
    field[:213, :213] = 100

    check = check_light(field, load_settings().light)

    assert check.level == BAD and "uneven" in check.text


def test_A6_setup_view_fits_the_screen_and_has_a_start_button():
    settings = load_settings()
    frame = np.full((1080, 1920, 3), PAPER, dtype=np.uint8)
    checks, corners = check_setup(frame, settings)

    view = draw_setup(frame, checks, corners)
    left, top, right, bottom = START_BUTTON

    assert view.shape[1] <= WINDOW_MAX_WIDTH
    assert is_start_click((left + right) // 2, (top + bottom) // 2)
    assert not is_start_click(right + 10, top)
    left, top, right, bottom = GUIDES_BUTTON
    assert bottom <= view.shape[0]
    assert is_guides_click((left + right) // 2, (top + bottom) // 2)
    assert not is_start_click((left + right) // 2, (top + bottom) // 2)


def test_A6_hidden_guides_show_the_plain_camera_image():
    settings = load_settings()
    frame = _draw_board(settings.board)
    checks, corners = check_setup(frame, settings)

    shown = draw_setup(frame, checks, corners)
    hidden = draw_setup(frame, checks, corners, guides_shown=False)

    camera = slice(PANEL_WIDTH, None)
    assert shown.shape == hidden.shape
    assert not np.array_equal(shown[:, camera], hidden[:, camera])
    # Without guides the middle of the image is plain paper, no cross and no spirit level.
    middle = hidden[hidden.shape[0] // 2, PANEL_WIDTH + (hidden.shape[1] - PANEL_WIDTH) // 2]
    assert abs(int(middle[0]) - PAPER) <= 2


def test_A6_centre_cross_runs_through_the_middle_of_the_field():
    square = np.float32([[0, 0], [100, 0], [100, 100], [0, 100]])
    # Seen at a slant: the far edge (top) looks shorter, so the middle lies closer to it.
    slanted = np.float32([[10, 0], [90, 0], [100, 100], [0, 100]])

    (down_start, down_end), (across_start, across_end) = field_centre_lines(square)
    _, (slanted_start, slanted_end) = field_centre_lines(slanted)

    assert np.allclose([down_start, down_end], [[50, 0], [50, 100]])
    assert np.allclose([across_start, across_end], [[0, 50], [100, 50]])
    assert slanted_start[1] == slanted_end[1] < 50


def test_A6_arrow_says_move_the_camera_left_when_it_hangs_too_far_right():
    # Real camera geometry: the camera hangs 60 mm right of the field centre.
    corners = _field_seen_by_camera(move_x_mm=60)

    check = check_centre(corners, (1080, 1920, 3))

    # The field then lies left of the image centre, so the arrow to it points left.
    assert corners.mean(axis=0)[0] < 960
    assert check.text.endswith("move camera left")


def test_A6_level_says_tilt_down_when_the_lens_looks_up():
    # The lens looks 9 degrees towards the top of the image; the camera is moved so the field stays in the middle.
    corners = _field_seen_by_camera(tilt_up_deg=9, move_y_mm=75)

    lean = camera_lean(corners)
    check = check_straight(corners)

    # The dot goes up, so the arrow from the dot to the middle points down.
    assert lean[1] < 0
    assert check.text.endswith("tilt lens a bit to the bottom")


def _field_seen_by_camera(tilt_up_deg=0, move_x_mm=0, move_y_mm=0, height_mm=470):
    """The four field corners in the image of a camera above the field, x right and y down as on the screen."""
    focal_px = 1300
    half = 90
    field = np.array([[-half, -half, 0], [half, -half, 0], [half, half, 0], [-half, half, 0]], dtype=float)
    rotation, _ = cv2.Rodrigues(np.radians([-tilt_up_deg, 0, 0]))
    seen = (rotation @ (field - [move_x_mm, move_y_mm, -height_mm]).T).T
    return np.float32(seen[:, :2] / seen[:, 2:] * focal_px + [960, 540])


def _draw_board(board):
    """Draws the printed page seen straight from above: grey paper and the four markers."""
    size_mm = board.field_size_mm + 2 * MARGIN_MM
    image = np.full((size_mm * PX_PER_MM, size_mm * PX_PER_MM, 3), PAPER, dtype=np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICTIONARY)
    marker_px = int(board.marker_size_mm * PX_PER_MM)
    for marker_id, (center_x, center_y) in board.marker_centers_mm().items():
        marker = cv2.aruco.generateImageMarker(dictionary, marker_id, marker_px)
        left = int((MARGIN_MM + center_x - board.marker_size_mm / 2) * PX_PER_MM)
        top = int((MARGIN_MM + center_y - board.marker_size_mm / 2) * PX_PER_MM)
        image[top : top + marker_px, left : left + marker_px] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
    return image
