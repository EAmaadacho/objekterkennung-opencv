"""Starts the program: reads the webcam, finds the field, detects the objects and reports changes (A4, A6)."""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import cv2

from objekterkennung.board import find_homography, find_markers, straighten_field
from objekterkennung.calibration import run_calibration
from objekterkennung.camera import CameraError, open_camera, read_frame
from objekterkennung.config import load_settings
from objekterkennung.detector import detect_objects
from objekterkennung.display import compose_window, draw_markers, draw_notice, draw_objects
from objekterkennung.hand import hand_in_field
from objekterkennung.protocol import LabelProtocol
from objekterkennung.smoothing import FrameSmoother
from objekterkennung.tracker import ChangeTracker

WINDOW_NAME = "Objekterkennung"
HAND_TEXT = "Hand im Feld - Meldungen pausiert"
QUIT_KEY = "q"
LIST_KEY = " "
SCREENSHOT_KEY = "s"
CALIBRATION_KEY = "c"
PAUSE_KEY = "u"
PAUSE_TEXT = "Pause zum Umbauen - u = weiter"
PROTOCOL_KEY = "p"
SCREENSHOT_FOLDER = Path("screenshots")


def main():
    """Starts the program and ends it with a short message instead of a traceback if the camera fails."""
    arguments = _parse_arguments()
    try:
        _run(arguments)
    except CameraError as error:
        print(f"Fehler: {error}")
        sys.exit(1)
    except KeyboardInterrupt:
        # Ctrl+C in the terminal: the protocol is already saved by the finally block in _run.
        print("Beendet mit Strg+C.")


def _run(arguments):
    """Runs the camera loop until the q key is pressed."""
    settings = load_settings()
    capture = open_camera(settings.camera_index)
    if arguments.calibrate:
        run_calibration(capture)
        settings = load_settings()
    tracker = ChangeTracker(settings.stable_frames, settings.move_threshold)
    smoother = FrameSmoother(settings.stable_frames)
    protocol = None
    homography = None
    objects = []
    paused = False
    print("Start. Tasten: q = beenden, Leertaste = Objektliste, s = Bildschirmfoto, p = Protokoll an/aus, u = Pause zum Umbauen, c = Kalibrierung")
    try:
        while True:
            frame = read_frame(capture)
            homography = find_homography(frame, settings.board, last_homography=homography)
            field_view = None
            if paused and homography is not None:
                # Nothing is detected, reported or recorded, but the field stays visible
                # so it can be rebuilt. Changes are reported once the pause ends.
                field_view = draw_notice(straighten_field(frame, homography, settings.board), PAUSE_TEXT)
            elif homography is not None:
                field = straighten_field(frame, homography, settings.board)
                hand = hand_in_field(field, settings)
                raw_objects = []
                # While a hand is in the field nothing is counted, so a covered block is not reported as removed.
                if not hand:
                    raw_objects = detect_objects(field, settings)
                    # The smoothed objects keep their name when a single frame is unclear.
                    objects = smoother.update(raw_objects)
                    for change in tracker.update(objects):
                        print(change.text())
                field_view = draw_objects(field, objects, settings.board)
                if hand:
                    field_view = draw_notice(field_view, HAND_TEXT)
            window = compose_window(draw_markers(frame, find_markers(frame)), field_view)
            cv2.imshow(WINDOW_NAME, window)
            if protocol is not None and field_view is not None and not paused:
                protocol.add(raw_objects, objects, field, window)
            # waitKey(1) waits 1 ms for a key and lets the window redraw itself.
            key = cv2.waitKey(1) & 0xFF
            if key == ord(QUIT_KEY):
                break
            if key == ord(LIST_KEY):
                _print_object_list(tracker.current_objects())
            if key == ord(SCREENSHOT_KEY):
                _save_screenshot(window)
            if key == ord(PROTOCOL_KEY):
                protocol = _toggle_protocol(protocol)
            if key == ord(PAUSE_KEY):
                paused = not paused
                print("Pause zum Umbauen (u = weiter)" if paused else "Weiter")
            if key == ord(CALIBRATION_KEY):
                run_calibration(capture)
                # The new colour ranges are in config.json now, so they are read again.
                settings = load_settings()
    finally:
        if protocol is not None:
            _toggle_protocol(protocol)
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


def _toggle_protocol(protocol):
    """Starts a new label protocol, or stops the running one and prints its summary."""
    if protocol is None:
        protocol = LabelProtocol()
        print(f"Protokoll läuft: {protocol.folder} (p = stoppen)")
        return protocol
    print(protocol.close())
    print(f"Protokoll gespeichert: {protocol.folder}")
    return None


def _save_screenshot(window):
    """Saves the window content, e.g. screenshots/2026-11-26_10-15-02.png."""
    SCREENSHOT_FOLDER.mkdir(exist_ok=True)
    path = SCREENSHOT_FOLDER / f"{datetime.now():%Y-%m-%d_%H-%M-%S}.png"
    cv2.imwrite(str(path), window)
    print(f"Bildschirmfoto gespeichert: {path}")


if __name__ == "__main__":
    main()
