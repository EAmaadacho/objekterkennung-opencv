"""Tests for the window content; the images are drawn with numpy, no camera is needed."""

import numpy as np

from objekterkennung.capture.board import unit_to_pixel
from objekterkennung.config import load_settings
from objekterkennung.output.display import (
    DOT_COLOR,
    MARKER_COLOR,
    MAX_WINDOW_WIDTH,
    PANEL_COLOR,
    STATUS_HEIGHT_PX,
    STATUS_OFF_COLOR,
    STATUS_ON_COLOR,
    STATUS_PAUSE_COLOR,
    add_status_bar,
    button_at,
    compose_window,
    draw_markers,
    draw_notice,
    draw_objects,
    pad_to_shape,
)
from objekterkennung.output.models import DetectedObject


def test_A6_window_shows_both_images_side_by_side():
    camera_image = np.zeros((480, 640, 3), dtype=np.uint8)
    field_image = np.zeros((600, 600, 3), dtype=np.uint8)

    window = compose_window(camera_image, field_image)

    assert window.shape == (600, 800 + 600, 3)


def test_A6_wide_camera_image_is_shrunk_so_the_window_fits_the_screen():
    camera_image = np.full((1080, 1920, 3), 255, dtype=np.uint8)
    field_image = np.zeros((640, 640, 3), dtype=np.uint8)

    window = compose_window(camera_image, field_image)

    assert window.shape == (640, MAX_WINDOW_WIDTH, 3)
    # Below the shrunk camera image the window shows the dark panel colour.
    assert window[0, 0].all() and tuple(window[-1, 0]) == PANEL_COLOR


def test_A6_without_field_only_camera_image_with_text():
    camera_image = np.zeros((480, 640, 3), dtype=np.uint8)

    window = compose_window(camera_image, None)

    assert window.shape == (480, 640, 3)
    assert window.any()


def test_A6_markers_are_drawn_without_changing_the_original():
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    markers = {0: np.array([[50, 50], [100, 50], [100, 100], [50, 100]], dtype=np.float32)}

    result = draw_markers(image, markers)

    assert (result == MARKER_COLOR).all(axis=2).any()
    assert not image.any()


def test_A3_detected_object_is_marked_at_its_position():
    board = load_settings().board
    field = np.full((700, 700, 3), 255, dtype=np.uint8)
    cube = DetectedObject("cube", "green", 0.5, 0.75)

    result = draw_objects(field, [cube], board)

    # 0.5 | 0.75 lies at pixel (350, 200): 50 px margin plus 300 and 150 px in the 600 px field.
    assert tuple(result[200, 350]) == DOT_COLOR
    assert (field == 255).all()


def test_A4_notice_is_written_without_changing_the_original():
    image = np.zeros((200, 400, 3), dtype=np.uint8)

    result = draw_notice(image, "Hand in field")

    assert result.any()
    assert not image.any()


def _has_color(image, color):
    return bool((image == np.array(color, dtype=np.uint8)).all(axis=2).any())


def test_A4_status_bar_shows_running_protocol_and_pause():
    window = np.zeros((600, 1400, 3), dtype=np.uint8)

    result = add_status_bar(window, protocol_running=True, paused=True)

    bar = result[:STATUS_HEIGHT_PX]
    assert result.shape == (600 + STATUS_HEIGHT_PX, 1400, 3)
    assert _has_color(bar, STATUS_ON_COLOR)
    assert _has_color(bar, STATUS_PAUSE_COLOR)


def test_A4_pause_puts_an_orange_frame_around_the_window():
    window = np.zeros((600, 1400, 3), dtype=np.uint8)

    result = add_status_bar(window, protocol_running=False, paused=True)

    field = result[STATUS_HEIGHT_PX:]
    assert tuple(field[0, 700]) == STATUS_PAUSE_COLOR
    assert tuple(field[300, 0]) == STATUS_PAUSE_COLOR
    assert not field[300, 700].any()
    assert not window.any()


def test_A4_status_bar_is_grey_when_protocol_is_off_and_no_pause():
    window = np.zeros((600, 1400, 3), dtype=np.uint8)

    result = add_status_bar(window, protocol_running=False, paused=False)

    bar = result[:STATUS_HEIGHT_PX]
    assert not _has_color(bar, STATUS_ON_COLOR)
    assert not _has_color(bar, STATUS_PAUSE_COLOR)
    assert _has_color(bar, STATUS_OFF_COLOR)


def test_A7_status_bar_shows_the_light_check():
    window = np.zeros((600, 1094, 3), dtype=np.uint8)
    light_color = (11, 222, 33)

    result = add_status_bar(window, protocol_running=True, paused=False, light=("Light: lamp one step darker", light_color))

    assert _has_color(result[:STATUS_HEIGHT_PX], light_color)


def test_A6_window_is_padded_to_the_shape_of_the_screen():
    image = np.full((500, 1000, 3), 7, dtype=np.uint8)

    padded, (left, top) = pad_to_shape(image, 1600, 900)

    assert padded.shape[:2] == (562, 1000)
    assert (left, top) == (0, 31)
    assert (padded[top : top + 500, left : left + 1000] == 7).all()
    assert tuple(padded[0, 0]) == PANEL_COLOR


def test_A6_window_of_unknown_size_is_not_padded():
    image = np.zeros((500, 1000, 3), dtype=np.uint8)

    padded, offset = pad_to_shape(image, 0, 0)

    assert padded is image and offset == (0, 0)


def test_A4_click_on_a_button_gives_its_name():
    pause_x = 400

    names = [button_at(pause_x, 20), button_at(pause_x, STATUS_HEIGHT_PX + 50), button_at(1300, 20)]

    assert names == ["pause", None, None]


def test_A3_labels_of_close_objects_do_not_cover_each_other():
    board = load_settings().board
    field = np.full((700, 700, 3), 255, dtype=np.uint8)
    cubes = [DetectedObject("cube", "red", 0.5, 0.5), DetectedObject("cube", "blue", 0.5, 0.45)]

    result = draw_objects(field, cubes, board)

    # Both dots stay visible at their own position; the second label moved away to make room.
    first, second = (unit_to_pixel(item.x, item.y, board) for item in cubes)
    assert tuple(result[first[1], first[0]]) == DOT_COLOR
    assert tuple(result[second[1], second[0]]) == DOT_COLOR
