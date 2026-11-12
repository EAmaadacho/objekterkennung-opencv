"""Builds the one window: camera image and straight field with the detected objects side by side (A6)."""

import cv2
import numpy as np

from objekterkennung.board import unit_to_pixel

NO_FIELD_TEXT = "Spielfeld nicht gefunden"
TEXT_COLOR = (0, 0, 255)
MARKER_COLOR = (0, 255, 0)
LABEL_COLOR = (0, 0, 0)
DOT_COLOR = (255, 0, 255)
# Wider windows do not fit on a laptop screen; the camera image is shrunk instead of the field.
MAX_WINDOW_WIDTH = 1400
# Status bar above the window with buttons for the mouse; each button also has a key.
STATUS_HEIGHT_PX = 40
STATUS_BACKGROUND = (40, 40, 40)
STATUS_TEXT_COLOR = (255, 255, 255)
STATUS_ON_COLOR = (0, 0, 220)
STATUS_PAUSE_COLOR = (0, 160, 255)
STATUS_OFF_COLOR = (90, 90, 90)
# (name, left edge, right edge) of every button in pixels.
BUTTONS = [("protocol", 6, 256), ("pause", 264, 474), ("photo", 482, 602), ("list", 610, 770), ("quit", 778, 918)]
KEY_HELP = "c = Kalibrieren"
# cv2.putText only knows ASCII letters, so umlauts are written out.
ASCII_REPLACEMENTS = {"ä": "ae", "ö": "oe", "ü": "ue", "Ä": "Ae", "Ö": "Oe", "Ü": "Ue", "ß": "ss"}


def draw_markers(image, markers):
    """Draws the outline and the ID of every found marker into a copy of the camera image."""
    result = image.copy()
    for marker_id, corners in markers.items():
        points = corners.astype(np.int32)
        cv2.drawContours(result, [points], -1, MARKER_COLOR, 2)
        top_left = (int(points[0][0]), int(points[0][1]) - 8)
        cv2.putText(result, str(marker_id), top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.8, MARKER_COLOR, 2)
    return result


def compose_window(camera_image, field_image):
    """Puts both images next to each other: the camera image on the left, the straight field with its margin on the right.

    The field keeps its full size so the labels stay readable. The camera image
    gets the field's height, but at most the width that is left up to
    MAX_WINDOW_WIDTH; if it is lower than the field, black is added below.
    Without a field (no homography yet) only the camera image is shown,
    with the text "Spielfeld nicht gefunden" in the top-left corner.
    """
    if field_image is None:
        result = _fit(camera_image, camera_image.shape[0], MAX_WINDOW_WIDTH)
        cv2.putText(result, NO_FIELD_TEXT, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, TEXT_COLOR, 2)
        return result
    height = field_image.shape[0]
    camera_scaled = _fit(camera_image, height, MAX_WINDOW_WIDTH - field_image.shape[1])
    # np.hstack needs two images of the same height.
    padded = np.zeros((height, camera_scaled.shape[1], 3), dtype=np.uint8)
    padded[: camera_scaled.shape[0]] = camera_scaled
    return np.hstack([padded, field_image])


def _fit(image, max_height, max_width):
    """Scales an image so that it is at most max_height high and max_width wide, keeping its aspect ratio."""
    scale = min(max_height / image.shape[0], max_width / image.shape[1])
    size = (round(image.shape[1] * scale), round(image.shape[0] * scale))
    return cv2.resize(image, size)


def draw_objects(field_image, objects, board):
    """Marks every detected object in a copy of the straight field: a dot at its centre and its label."""
    result = field_image.copy()
    for item in objects:
        center = unit_to_pixel(item.x, item.y, board)
        cv2.circle(result, center, 5, DOT_COLOR, -1)
        lines = [to_ascii(item.name()), item.position()]
        for line_number, line in enumerate(lines):
            origin = (center[0] - 40, center[1] + 22 + 18 * line_number)
            cv2.putText(result, line, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, LABEL_COLOR, 1, cv2.LINE_AA)
    return result


def draw_notice(image, text):
    """Writes a red notice into the top-left corner of a copy, e.g. "Hand im Feld - Meldungen pausiert"."""
    result = image.copy()
    cv2.putText(result, to_ascii(text), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR, 2)
    return result


def to_ascii(text):
    """Replaces umlauts, e.g. "grün Würfel" becomes "gruen Wuerfel"."""
    return "".join(ASCII_REPLACEMENTS.get(letter, letter) for letter in text)


def add_status_bar(window, protocol_running, paused):
    """Puts a dark bar with buttons above the window; it shows whether the protocol runs and whether there is a pause.

    The protocol button is red while a protocol runs, the pause button orange during
    a pause. The buttons are plain filled rectangles (numpy slices) with text.
    """
    bar = np.zeros((STATUS_HEIGHT_PX, window.shape[1], 3), dtype=np.uint8)
    bar[:] = STATUS_BACKGROUND
    labels = {
        "protocol": ("PROTOKOLL STOPPEN (p)", STATUS_ON_COLOR) if protocol_running else ("Protokoll starten (p)", STATUS_OFF_COLOR),
        "pause": ("PAUSE - weiter (u)", STATUS_PAUSE_COLOR) if paused else ("Pause (u)", STATUS_OFF_COLOR),
        "photo": ("Foto (s)", STATUS_OFF_COLOR),
        "list": ("Liste (Leer)", STATUS_OFF_COLOR),
        "quit": ("Beenden (q)", STATUS_OFF_COLOR),
    }
    for name, left, right in BUTTONS:
        text, color = labels[name]
        bar[5 : STATUS_HEIGHT_PX - 5, left:right] = color
        cv2.putText(bar, text, (left + 8, STATUS_HEIGHT_PX // 2 + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, STATUS_TEXT_COLOR, 1)
    cv2.putText(bar, KEY_HELP, (BUTTONS[-1][2] + 16, STATUS_HEIGHT_PX // 2 + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, STATUS_TEXT_COLOR, 1)
    return np.vstack([bar, window])


def button_at(x, y):
    """Name of the button at a mouse position in the window, e.g. "pause", or None."""
    if not 0 <= y < STATUS_HEIGHT_PX:
        return None
    for name, left, right in BUTTONS:
        if left <= x < right:
            return name
    return None
