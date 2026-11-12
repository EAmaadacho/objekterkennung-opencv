"""Tests for the window content; the images are drawn with numpy, no camera is needed."""

import numpy as np

from objekterkennung.config import load_settings
from objekterkennung.display import (
    DOT_COLOR,
    MAX_WINDOW_WIDTH,
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
)
from objekterkennung.models import DetectedObject

GREEN = (0, 255, 0)


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
    assert window[0, 0].all() and not window[-1, 0].any()


def test_A6_without_field_only_camera_image_with_text():
    camera_image = np.zeros((480, 640, 3), dtype=np.uint8)

    window = compose_window(camera_image, None)

    assert window.shape == (480, 640, 3)
    assert window.any()


def test_A6_markers_are_drawn_without_changing_the_original():
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    markers = {0: np.array([[50, 50], [100, 50], [100, 100], [50, 100]], dtype=np.float32)}

    result = draw_markers(image, markers)

    assert (result == GREEN).all(axis=2).any()
    assert not image.any()


def test_A3_detected_object_is_marked_at_its_position():
    board = load_settings().board
    field = np.full((640, 640, 3), 255, dtype=np.uint8)
    cube = DetectedObject("cube", "green", 0.5, 0.75)

    result = draw_objects(field, [cube], board)

    # 0.5 | 0.75 lies at pixel (320, 170): 20 px margin plus 300 and 150 px in the 600 px field.
    assert tuple(result[170, 320]) == DOT_COLOR
    assert (field == 255).all()


def test_A4_notice_is_written_without_changing_the_original():
    image = np.zeros((200, 400, 3), dtype=np.uint8)

    result = draw_notice(image, "Hand im Feld")

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


def test_A4_status_bar_is_grey_when_protocol_is_off_and_no_pause():
    window = np.zeros((600, 1400, 3), dtype=np.uint8)

    result = add_status_bar(window, protocol_running=False, paused=False)

    bar = result[:STATUS_HEIGHT_PX]
    assert not _has_color(bar, STATUS_ON_COLOR)
    assert not _has_color(bar, STATUS_PAUSE_COLOR)
    assert _has_color(bar, STATUS_OFF_COLOR)


def test_A4_click_on_a_button_gives_its_name():
    pause_x = 300

    names = [button_at(pause_x, 20), button_at(pause_x, STATUS_HEIGHT_PX + 50), button_at(1300, 20)]

    assert names == ["pause", None, None]
