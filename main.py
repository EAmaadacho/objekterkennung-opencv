"""Starts the program: reads the webcam, finds the field, detects the objects and reports changes (A4, A6)."""

import argparse
import sys

import cv2

from objekterkennung.board import find_homography, find_markers, straighten_field
from objekterkennung.calibration import run_calibration
from objekterkennung.camera import CameraError, open_camera, read_frame
from objekterkennung.config import load_settings
from objekterkennung.detector import detect_objects
from objekterkennung.display import add_status_bar, button_at, compose_window, draw_markers, draw_notice, draw_objects
from objekterkennung.hand import hand_in_field
from objekterkennung.protocol import LabelProtocol
from objekterkennung.setup_check import run_setup
from objekterkennung.smoothing import FrameSmoother
from objekterkennung.tracker import ChangeTracker

WINDOW_NAME = "Object detection"
HAND_TEXT = "Hand in field - reports paused"
QUIT_KEY = "q"
CALIBRATION_KEY = "c"
PAUSE_KEY = "u"
PAUSE_TEXT = "PAUSED - move the blocks, then press u"
PROTOCOL_KEY = "p"
BUTTON_KEYS = {"protocol": PROTOCOL_KEY, "pause": PAUSE_KEY, "quit": QUIT_KEY}


def main():
    """Starts the program and ends it with a short message instead of a traceback if the camera fails."""
    arguments = _parse_arguments()
    try:
        _run(arguments)
    except CameraError as error:
        print(f"Error: {error}")
        sys.exit(1)
    except KeyboardInterrupt:
        # Ctrl+C in the terminal: the protocol is already saved by the finally block in _run.
        print("Stopped with Ctrl+C.")


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
    clicks = []
    try:
        # First the setup mode, so camera and light are checked before anything is detected.
        if not arguments.skip_setup and not run_setup(capture, settings.board, WINDOW_NAME):
            return
        cv2.namedWindow(WINDOW_NAME)
        # OpenCV calls this function for every mouse event in the window; a left click is kept for the loop.
        cv2.setMouseCallback(WINDOW_NAME, lambda event, x, y, flags, param: _remember_click(event, x, y, clicks))
        print("Started. Keys: p = protocol on/off, u = pause on/off, c = calibration, q = quit")
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
            window = add_status_bar(window, protocol is not None, paused)
            cv2.imshow(WINDOW_NAME, window)
            if protocol is not None and field_view is not None and not paused:
                protocol.add(raw_objects, objects, field, window)
            # waitKey(1) waits 1 ms for a key and lets the window redraw itself.
            key = cv2.waitKey(1) & 0xFF
            # A click on a button counts like its key.
            if clicks:
                name = button_at(*clicks.pop(0))
                if name is not None:
                    key = ord(BUTTON_KEYS[name])
            if key == ord(QUIT_KEY):
                break
            if key == ord(PROTOCOL_KEY):
                protocol = _toggle_protocol(protocol)
            if key == ord(PAUSE_KEY):
                paused = not paused
                print("Paused (u = resume)" if paused else "Resumed")
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
    parser = argparse.ArgumentParser(description="Detects shape, colour and position of the objects on the field.")
    parser.add_argument("--calibrate", action="store_true", help="open the calibration mode first")
    parser.add_argument("--skip-setup", action="store_true", help="start the detection without the setup mode")
    return parser.parse_args()


def _remember_click(event, x, y, clicks):
    """Keeps the position of a left mouse click, e.g. on the pause button."""
    if event == cv2.EVENT_LBUTTONDOWN:
        clicks.append((x, y))


def _toggle_protocol(protocol):
    """Starts a new label protocol, or stops the running one and prints its summary."""
    if protocol is None:
        protocol = LabelProtocol()
        print(f"Protocol recording: {protocol.folder} (p = stop)")
        return protocol
    print(protocol.close())
    print(f"Protocol saved: {protocol.folder}")
    return None


if __name__ == "__main__":
    main()
