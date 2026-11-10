"""Opens the USB webcam and reads single frames from it (A6)."""

import cv2

# Without this request many webcams send 640 x 480, which is a cut-out from the
# middle of the sensor: the picture looks zoomed in and the board does not fit.
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080


class CameraError(RuntimeError):
    """Raised when the webcam cannot be opened or stops delivering frames."""


def open_camera(index):
    """Opens the webcam with the given index, e.g. 0 for the first camera.

    Index 0 is usually the built-in laptop camera, so the USB webcam is often
    index 1. The index is set in config.json ("camera_index").
    """
    capture = cv2.VideoCapture(index)
    if not capture.isOpened():
        capture.release()
        raise CameraError(
            f"No camera found at index {index}. "
            'Check the USB cable or change "camera_index" in config.json.'
        )
    # The camera picks the nearest size it supports, e.g. 1280 x 720.
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    return capture


def read_frame(capture):
    """Reads the next image from the camera as a BGR array of shape (height, width, 3)."""
    ok, frame = capture.read()
    if not ok:
        # read() fails when the cable is pulled while the program runs.
        raise CameraError("The camera stopped delivering images.")
    return frame
