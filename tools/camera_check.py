"""Shows what the webcam really delivers and opens its driver settings, e.g. to reset a digital zoom (A6).

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


def main():
    settings = load_settings()
    capture = open_camera(settings.camera_index)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Kamera {settings.camera_index}: {width} x {height} Pixel, Zoom-Wert {capture.get(cv2.CAP_PROP_ZOOM)}")
    print("Taste e = Kameraeinstellungen öffnen (dort Zoom ganz zurück, Autofokus an), q = beenden")
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
                # Opens the Windows driver dialog of the camera; works with DirectShow.
                capture.set(cv2.CAP_PROP_SETTINGS, 1)
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
