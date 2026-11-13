"""Opens the USB webcam and reads single frames from it (A6)."""

import sys

import cv2

# A single failed read can happen, e.g. when Windows briefly hands the camera to
# another app. Only several failures in a row mean the camera is really gone.
READ_ATTEMPTS = 5


class CameraError(RuntimeError):
    """Raised when the webcam cannot be opened or stops delivering frames."""


def open_camera(camera):
    """Opens the webcam described in config.json (section "camera"), e.g. number 0 for the first camera.

    Number 0 is usually the built-in laptop camera, so the USB webcam is often number 1.
    """
    index = camera.number
    # On Windows DirectShow handles the Full HD request more reliably and can show the
    # driver settings. Not every OpenCV build supports it, so the default driver is the fallback.
    capture = None
    if sys.platform == "win32":
        capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not capture.isOpened():
            capture.release()
            capture = None
    if capture is None:
        capture = cv2.VideoCapture(index)
    if not capture.isOpened():
        capture.release()
        raise CameraError(
            f"No camera found with index {index}. "
            'Check the USB cable or change "camera": "number" in config.json.'
        )
    # MJPG is the compressed format in which USB webcams deliver Full HD fast enough.
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    # Without this request many webcams send 640 x 480, which is a cut-out from the middle
    # of the sensor: the picture looks zoomed in and the board does not fit.
    # The camera picks the nearest size it supports, e.g. 1280 x 720.
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, camera.width_px)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, camera.height_px)
    return capture


def read_frame(capture):
    """Reads the next image from the camera as a BGR array of shape (height, width, 3)."""
    for _ in range(READ_ATTEMPTS):
        ok, frame = capture.read()
        if ok:
            return frame
    # read() keeps failing when the cable is pulled or another app took the camera.
    raise CameraError(
        "The camera stopped sending images. Check the cable and close other programs "
        "that use the camera, e.g. Teams or the Camera app."
    )
