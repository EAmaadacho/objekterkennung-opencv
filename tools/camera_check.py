"""Shows what the webcam really delivers and lets you reset its zoom (A6).

Start from the project folder: python tools/camera_check.py
"""

import sys
from pathlib import Path

PROJECT_FOLDER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_FOLDER))

import cv2  # noqa: E402

from objekterkennung.camera import open_camera, read_frame  # noqa: E402
from objekterkennung.config import load_settings  # noqa: E402

WINDOW_NAME = "Kamera-Check"
QUIT_KEY = "q"
SETTINGS_KEY = "e"
ZOOM_OUT_KEY = "-"
ZOOM_IN_KEY = "+"
ZOOM_STEP = 10


def main():
    settings = load_settings()
    capture = open_camera(settings.camera_index)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Kamera {settings.camera_index} ({capture.getBackendName()}): {width} x {height} Pixel")
    _print_zoom(capture)
    print("Tasten: - = rauszoomen, + = reinzoomen, e = Treibereinstellungen, q = beenden")
    try:
        while True:
            frame = read_frame(capture)
            # Shown at half size, so a Full HD picture fits on the laptop screen.
            small = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
            cv2.imshow(WINDOW_NAME, small)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(QUIT_KEY):
                break
            if key == ord(SETTINGS_KEY):
                # Opens the driver dialog of the camera; only DirectShow supports it.
                if not capture.set(cv2.CAP_PROP_SETTINGS, 1):
                    print("Treibereinstellungen werden von diesem Treiber nicht unterstuetzt.")
            if key in (ord(ZOOM_OUT_KEY), ord(ZOOM_IN_KEY)):
                step = -ZOOM_STEP if key == ord(ZOOM_OUT_KEY) else ZOOM_STEP
                capture.set(cv2.CAP_PROP_ZOOM, capture.get(cv2.CAP_PROP_ZOOM) + step)
                _print_zoom(capture)
    finally:
        capture.release()
        cv2.destroyAllWindows()


def _print_zoom(capture):
    """Prints the current zoom; -1 or 0 means the driver does not report it."""
    print(f"Zoom-Wert: {capture.get(cv2.CAP_PROP_ZOOM)}")


if __name__ == "__main__":
    main()
