"""Builds the one window: camera image and straight field with the detected objects side by side (A6)."""

import cv2
import numpy as np

from objekterkennung.board import unit_to_pixel

NO_FIELD_TEXT = "Spielfeld nicht gefunden"
TEXT_COLOR = (0, 0, 255)
MARKER_COLOR = (0, 255, 0)
LABEL_COLOR = (0, 0, 0)
DOT_COLOR = (255, 0, 255)
# cv2.putText only knows ASCII letters, so umlauts are written out.
ASCII_REPLACEMENTS = {"ä": "ae", "ö": "oe", "ü": "ue", "Ä": "Ae", "Ö": "Oe", "Ü": "Ue", "ß": "ss"}


def draw_markers(image, markers):
    """Draws the outline and the ID of every found marker into a copy of the camera image."""
    result = image.copy()
    for marker_id, corners in markers.items():
        points = corners.astype(np.int32)
        cv2.polylines(result, [points], isClosed=True, color=MARKER_COLOR, thickness=2)
        top_left = (int(points[0][0]), int(points[0][1]) - 8)
        cv2.putText(result, str(marker_id), top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.8, MARKER_COLOR, 2)
    return result


def compose_window(camera_image, field_image):
    """Puts both images next to each other, e.g. a 640 px high camera image and the 640 x 640 field with its margin.

    Without a field (no homography yet) only the camera image is shown,
    with the text "Spielfeld nicht gefunden" in the top-left corner.
    """
    if field_image is None:
        result = camera_image.copy()
        cv2.putText(result, NO_FIELD_TEXT, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, TEXT_COLOR, 2)
        return result
    height = field_image.shape[0]
    scale = height / camera_image.shape[0]
    width = round(camera_image.shape[1] * scale)
    # np.hstack needs two images of the same height.
    camera_scaled = cv2.resize(camera_image, (width, height))
    return np.hstack([camera_scaled, field_image])


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
