"""Tests for the camera module; they never need a real camera."""

import numpy as np
import pytest

from objekterkennung.camera import CameraError, open_camera, read_frame

MISSING_CAMERA_INDEX = 99


class FakeCapture:
    """Stands in for cv2.VideoCapture and returns a prepared result."""

    def __init__(self, ok, frame):
        self._ok = ok
        self._frame = frame

    def read(self):
        return self._ok, self._frame


def test_A6_missing_camera_gives_clear_error():
    index = MISSING_CAMERA_INDEX

    with pytest.raises(CameraError, match="camera_index"):
        open_camera(index)


def test_A6_read_frame_returns_the_image():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    capture = FakeCapture(ok=True, frame=image)

    frame = read_frame(capture)

    assert frame.shape == (480, 640, 3)


def test_A6_lost_camera_gives_clear_error():
    capture = FakeCapture(ok=False, frame=None)

    with pytest.raises(CameraError, match="stopped"):
        read_frame(capture)
