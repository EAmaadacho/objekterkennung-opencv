"""Tests for the camera module; they never need a real camera."""

import cv2
import numpy as np
import pytest

from objekterkennung import camera
from objekterkennung.camera import FRAME_HEIGHT, FRAME_WIDTH, CameraError, open_camera, read_frame

MISSING_CAMERA_INDEX = 99


class FakeCapture:
    """Stands in for cv2.VideoCapture and returns a prepared result."""

    def __init__(self, ok, frame):
        self._ok = ok
        self._frame = frame

    def read(self):
        return self._ok, self._frame


class FlickeringCapture:
    """Fails a few times first, then delivers the frame, like a camera that hiccups briefly."""

    def __init__(self, failures, frame):
        self._failures = failures
        self._frame = frame

    def read(self):
        if self._failures > 0:
            self._failures -= 1
            return False, None
        return True, self._frame


class FakeOpenCamera:
    """Stands in for an opened cv2.VideoCapture and remembers the requested settings."""

    def __init__(self, index, backend=None):
        self.requested = {}

    def isOpened(self):
        return True

    def set(self, setting, value):
        self.requested[setting] = value


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

    with pytest.raises(CameraError, match="keine Bilder mehr"):
        read_frame(capture)


def test_A6_camera_is_asked_for_full_hd(monkeypatch):
    monkeypatch.setattr(camera.cv2, "VideoCapture", FakeOpenCamera)

    capture = open_camera(1)

    assert capture.requested[cv2.CAP_PROP_FRAME_WIDTH] == FRAME_WIDTH
    assert capture.requested[cv2.CAP_PROP_FRAME_HEIGHT] == FRAME_HEIGHT


def test_A6_single_failed_read_is_retried():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    capture = FlickeringCapture(failures=2, frame=image)

    frame = read_frame(capture)

    assert frame is image
