"""Starts the program: reads the webcam, finds the field and shows both images in one window (A6)."""

import cv2

from objekterkennung.board import find_homography, find_markers, straighten_field
from objekterkennung.camera import open_camera, read_frame
from objekterkennung.config import load_settings
from objekterkennung.display import compose_window, draw_markers

WINDOW_NAME = "Objekterkennung"
QUIT_KEY = "q"


def main():
    """Runs the camera loop until the q key is pressed."""
    settings = load_settings()
    capture = open_camera(settings.camera_index)
    homography = None
    try:
        while True:
            frame = read_frame(capture)
            homography = find_homography(frame, settings.board, last_homography=homography)
            field = None if homography is None else straighten_field(frame, homography, settings.board)
            camera_view = draw_markers(frame, find_markers(frame))
            cv2.imshow(WINDOW_NAME, compose_window(camera_view, field))
            # waitKey(1) waits 1 ms for a key and lets the window redraw itself.
            if cv2.waitKey(1) & 0xFF == ord(QUIT_KEY):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
