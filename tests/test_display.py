"""Tests for the window content; the images are drawn with numpy, no camera is needed."""

import numpy as np

from objekterkennung.config import load_settings
from objekterkennung.display import DOT_COLOR, compose_window, draw_markers, draw_objects
from objekterkennung.models import DetectedObject

GREEN = (0, 255, 0)


def test_A6_window_shows_both_images_side_by_side():
    camera_image = np.zeros((480, 640, 3), dtype=np.uint8)
    field_image = np.zeros((600, 600, 3), dtype=np.uint8)

    window = compose_window(camera_image, field_image)

    assert window.shape == (600, 800 + 600, 3)


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
    field = np.full((600, 600, 3), 255, dtype=np.uint8)
    cube = DetectedObject("cube", "green", 0.5, 0.75)

    result = draw_objects(field, [cube], board)

    assert tuple(result[150, 300]) == DOT_COLOR
    assert (field == 255).all()
