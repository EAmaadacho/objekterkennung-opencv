"""Builds the one window: camera image and straight field with the detected objects side by side (A6)."""

import cv2
import numpy as np

from objekterkennung.board import unit_to_pixel
from objekterkennung.models import SHAPE_LABELS

NO_FIELD_TEXT = "Field not found"
TEXT_COLOR = (0, 0, 255)
MARKER_COLOR = (0, 255, 0)
LABEL_COLOR = (255, 255, 255)
# Label on the object: white text on a dark, half transparent box is readable on every colour.
LABEL_BOX_COLOR = (30, 30, 30)
LABEL_BOX_ALPHA = 0.65
LABEL_SCALE = 0.42
LABEL_LINE_PX = 15
LABEL_PADDING_PX = 4
DOT_GAP_PX = 10
# Smaller labels for objects lying close together, and how far a label may move to make room.
LABEL_SMALL_SCALE = 0.34
LABEL_SMALL_LINE_PX = 12
LABEL_SHIFT_STEP_PX = 6
LABEL_SHIFT_MAX_PX = 48
DOT_COLOR = (255, 0, 255)
# Wider windows do not fit on a laptop screen; the camera image is shrunk instead of the field.
MAX_WINDOW_WIDTH = 1400
# Status bar above the window with buttons for the mouse; each button also has a key.
STATUS_HEIGHT_PX = 50
STATUS_BACKGROUND = (40, 40, 40)
STATUS_TEXT_COLOR = (255, 255, 255)
STATUS_ON_COLOR = (0, 0, 220)
STATUS_PAUSE_COLOR = (0, 160, 255)
STATUS_OFF_COLOR = (90, 90, 90)
STATUS_LIVE_COLOR = (0, 200, 0)
# During a pause the whole window gets an orange frame of this width, so the pause is seen at once.
PAUSE_FRAME_PX = 8
# (name, left edge, right edge) of every button in pixels.
BUTTONS = [("protocol", 6, 306), ("pause", 314, 574), ("quit", 582, 702)]
KEY_HELP = "c = calibrate"


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
    with the text "Field not found" in the top-left corner.
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
    """Labels every detected object in a copy of the straight field, centred on the object itself.

    The label is a small dark box with white text, so it can be read on every colour:
    colour and shape above the centre, a dot at the measured centre, the position below.
    The box is half transparent, so the colour of the object still shows through.
    Labels of objects lying close together get a smaller font; if two boxes still
    overlap, the later one moves up or down and a thin line joins it to its dot.
    """
    result = field_image.copy()
    centers = [unit_to_pixel(item.x, item.y, board) for item in objects]
    full = [_label_box(item, center, LABEL_SCALE, LABEL_LINE_PX) for item, center in zip(objects, centers)]
    placed = []
    for number, (item, center) in enumerate(zip(objects, centers)):
        crowded = any(_overlap(full[number], other) for index, other in enumerate(full) if index != number)
        scale, line_px = (LABEL_SMALL_SCALE, LABEL_SMALL_LINE_PX) if crowded else (LABEL_SCALE, LABEL_LINE_PX)
        shift = 0
        for shift in [0] + [sign * step for step in range(LABEL_SHIFT_STEP_PX, LABEL_SHIFT_MAX_PX + 1, LABEL_SHIFT_STEP_PX) for sign in (-1, 1)]:
            box = _label_box(item, (center[0], center[1] + shift), scale, line_px)
            if not any(_overlap(box, other) for other in placed):
                break
        placed.append(box)
        if shift:
            cv2.line(result, center, (center[0], center[1] + shift), LABEL_COLOR, 1, cv2.LINE_AA)
        _draw_label(result, item, (center[0], center[1] + shift), box, scale, line_px)
    for center in centers:
        cv2.circle(result, center, 4, DOT_COLOR, -1)
    return result


def _label_lines(item):
    """The text above and below the dot, e.g. (["red", "cube"], ["(0.42|0.71)"])."""
    return [item.color, SHAPE_LABELS.get(item.shape, item.shape)], [item.compact_position()]


def _label_box(item, center, scale, line_px):
    """The rectangle (left, top, right, bottom) that the label of one object covers."""
    above, below = _label_lines(item)
    width = max(cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0] for line in above + below)
    half_width = width // 2 + LABEL_PADDING_PX
    top = center[1] - len(above) * line_px - DOT_GAP_PX // 2 - LABEL_PADDING_PX
    bottom = center[1] + len(below) * line_px + DOT_GAP_PX // 2 + LABEL_PADDING_PX
    return center[0] - half_width, top, center[0] + half_width, bottom


def _overlap(first, second):
    """True if two rectangles (left, top, right, bottom) cover a common pixel."""
    return first[0] < second[2] and second[0] < first[2] and first[1] < second[3] and second[1] < first[3]


def _draw_label(image, item, center, box, scale, line_px):
    """Darkens the box (numpy slices) and writes the lines centred into it."""
    left, top, right, bottom = (max(value, 0) for value in box)
    area = image[top:bottom, left:right]
    image[top:bottom, left:right] = (area * (1 - LABEL_BOX_ALPHA) + np.array(LABEL_BOX_COLOR) * LABEL_BOX_ALPHA).astype(np.uint8)
    above, below = _label_lines(item)
    for number, line in enumerate(above):
        _put_centred(image, line, center[0], box[1] + LABEL_PADDING_PX + (number + 1) * line_px - 3, scale)
    for number, line in enumerate(below):
        _put_centred(image, line, center[0], center[1] + DOT_GAP_PX // 2 + (number + 1) * line_px - 3, scale)


def _put_centred(image, text, center_x, baseline_y, scale):
    """Writes white text so that its middle lies at center_x."""
    (width, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    cv2.putText(image, text, (center_x - width // 2, baseline_y), cv2.FONT_HERSHEY_SIMPLEX, scale, LABEL_COLOR, 1, cv2.LINE_AA)


def draw_notice(image, text):
    """Writes a red notice into the top-left corner of a copy, e.g. "Hand in field - reports paused"."""
    result = image.copy()
    cv2.putText(result, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR, 2)
    return result


def add_status_bar(window, protocol_running, paused):
    """Puts a dark bar with buttons above the window; it shows whether the protocol runs and whether there is a pause.

    The protocol button is red while a protocol runs, the pause button orange during
    a pause, and a pause also gets an orange frame around the whole window. Next to the
    buttons one line says in words what the program is doing right now.
    The buttons are plain filled rectangles (numpy slices) with text.
    """
    result = window.copy()
    if paused:
        result[:PAUSE_FRAME_PX] = STATUS_PAUSE_COLOR
        result[-PAUSE_FRAME_PX:] = STATUS_PAUSE_COLOR
        result[:, :PAUSE_FRAME_PX] = STATUS_PAUSE_COLOR
        result[:, -PAUSE_FRAME_PX:] = STATUS_PAUSE_COLOR
    bar = np.zeros((STATUS_HEIGHT_PX, window.shape[1], 3), dtype=np.uint8)
    bar[:] = STATUS_BACKGROUND
    labels = {
        "protocol": ("REC - stop protocol (p)", STATUS_ON_COLOR) if protocol_running else ("Start protocol (p)", STATUS_OFF_COLOR),
        "pause": ("PAUSED - resume (u)", STATUS_PAUSE_COLOR) if paused else ("Pause (u)", STATUS_OFF_COLOR),
        "quit": ("Quit (q)", STATUS_OFF_COLOR),
    }
    text_y = STATUS_HEIGHT_PX // 2 + 7
    for name, left, right in BUTTONS:
        text, color = labels[name]
        bar[6 : STATUS_HEIGHT_PX - 6, left:right] = color
        cv2.putText(bar, text, (left + 10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, STATUS_TEXT_COLOR, 2 if color != STATUS_OFF_COLOR else 1)
    state, state_color = ("PAUSED: nothing is detected or recorded", STATUS_PAUSE_COLOR) if paused else ("LIVE: detecting objects", STATUS_LIVE_COLOR)
    if protocol_running and not paused:
        state += ", protocol is recording"
    cv2.putText(bar, state, (BUTTONS[-1][2] + 16, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 2)
    (help_width, _), _ = cv2.getTextSize(KEY_HELP, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(bar, KEY_HELP, (bar.shape[1] - help_width - 10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, STATUS_TEXT_COLOR, 1)
    return np.vstack([bar, result])


def button_at(x, y):
    """Name of the button at a mouse position in the window, e.g. "pause", or None."""
    if not 0 <= y < STATUS_HEIGHT_PX:
        return None
    for name, left, right in BUTTONS:
        if left <= x < right:
            return name
    return None
