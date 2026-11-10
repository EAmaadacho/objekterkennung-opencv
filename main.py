"""Starts the program: reads the webcam, finds the field, detects the objects and reports changes (A4, A6)."""

import argparse
from datetime import datetime
from pathlib import Path

import cv2

from objekterkennung.board import find_homography, find_markers, straighten_field
from objekterkennung.calibration import run_calibration
from objekterkennung.camera import open_camera, read_frame
from objekterkennung.config import load_settings
from objekterkennung.detector import detect_objects
from objekterkennung.display import compose_window, draw_markers, draw_objects
from objekterkennung.recording import save_recording
from objekterkennung.smoothing import FrameSmoother
from objekterkennung.tracker import ChangeTracker

WINDOW_NAME = "Objekterkennung"
QUIT_KEY = "q"
LIST_KEY = " "
SCREENSHOT_KEY = "s"
CALIBRATION_KEY = "c"
RECORDING_KEY = "a"
SCREENSHOT_FOLDER = Path("screenshots")


def main():
    """Runs the camera loop until the q key is pressed."""
    arguments = _parse_arguments()
    settings = load_settings()
    capture = open_camera(settings.camera_index)
    if arguments.calibrate:
        run_calibration(capture)
        settings = load_settings()
    tracker = ChangeTracker(settings.stable_frames, settings.move_threshold)
    smoother = FrameSmoother(settings.stable_frames)
    field = None
    objects = []
    homography = None
    print("Start. Tasten: q = beenden, Leertaste = Objektliste, s = Bildschirmfoto, a = Aufnahme, c = Kalibrierung")
    try:
        while True:
            frame = read_frame(capture)
            homography = find_homography(frame, settings.board, last_homography=homography)
            field_view = None
            if homography is not None:
                field = straighten_field(frame, homography, settings.board)
                # The smoothed objects keep their name when a single frame is unclear.
                objects = smoother.update(detect_objects(field, settings))
                for change in tracker.update(objects):
                    print(change.text())
                field_view = draw_objects(field, objects, settings.board)
            window = compose_window(draw_markers(frame, find_markers(frame)), field_view)
            cv2.imshow(WINDOW_NAME, window)
            # waitKey(1) waits 1 ms for a key and lets the window redraw itself.
            key = cv2.waitKey(1) & 0xFF
            if key == ord(QUIT_KEY):
                break
            if key == ord(LIST_KEY):
                _print_object_list(tracker.current_objects())
            if key == ord(SCREENSHOT_KEY):
                _save_screenshot(window)
            if key == ord(RECORDING_KEY) and field is not None:
                print(f"Aufnahme gespeichert: {save_recording(field, objects)}")
            if key == ord(CALIBRATION_KEY):
                run_calibration(capture)
                # The new colour ranges are in config.json now, so they are read again.
                settings = load_settings()
    finally:
        capture.release()
        cv2.destroyAllWindows()


def _parse_arguments():
    """Reads the command line: "python main.py --calibrate" starts with the calibration mode."""
    parser = argparse.ArgumentParser(description="Erkennt Form, Farbe und Position von Objekten auf dem Spielfeld.")
    parser.add_argument("--calibrate", action="store_true", help="erst den Kalibriermodus öffnen")
    return parser.parse_args()


def _print_object_list(objects):
    """Prints all objects of the last stable scene, one per line."""
    print(f"--- {len(objects)} Objekte ---")
    for item in objects:
        print(item.label())


def _save_screenshot(window):
    """Saves the window content, e.g. screenshots/2026-11-26_10-15-02.png."""
    SCREENSHOT_FOLDER.mkdir(exist_ok=True)
    path = SCREENSHOT_FOLDER / f"{datetime.now():%Y-%m-%d_%H-%M-%S}.png"
    cv2.imwrite(str(path), window)
    print(f"Bildschirmfoto gespeichert: {path}")


if __name__ == "__main__":
    main()
