"""Builds the one window: camera image and straight field with the detected objects side by side (A6).

Drawing and showing as in script ch. 2.3 and 3.3.6 (cv2.imshow, np.hstack); the see-through
labels use cv2.addWeighted (ch. 4.1.7).
"""

import cv2
import numpy as np

from objekterkennung.capture.board import unit_to_pixel
from objekterkennung.output.models import SHAPE_LABELS

NO_FIELD_TEXT = "Field not found"
# One look for every window: soft colours on dark grey, round corners, smooth text (cv2.LINE_AA).
# Colours are BGR.
FONT = cv2.FONT_HERSHEY_SIMPLEX
PANEL_COLOR = (32, 30, 28)
CARD_COLOR = (58, 54, 50)
WHITE = (245, 245, 245)
MUTED_COLOR = (160, 155, 150)
GREEN = (110, 200, 70)
AMBER = (50, 185, 250)
RED = (90, 90, 245)
MARKER_COLOR = GREEN
LABEL_COLOR = WHITE
# Label on the object: white text on a dark, half transparent box is readable on every colour.
LABEL_BOX_COLOR = (30, 28, 26)
LABEL_BOX_ALPHA = 0.7
LABEL_RADIUS_PX = 6
LABEL_SCALE = 0.42
LABEL_LINE_PX = 15
LABEL_PADDING_PX = 5
DOT_GAP_PX = 10
# Smaller labels for objects lying close together, and how far a label may move to make room.
LABEL_SMALL_SCALE = 0.34
LABEL_SMALL_LINE_PX = 12
LABEL_SHIFT_STEP_PX = 6
LABEL_SHIFT_MAX_PX = 48
DOT_COLOR = WHITE
# Wider windows do not fit on a laptop screen; the camera image is shrunk instead of the field.
MAX_WINDOW_WIDTH = 1400
# Status bar above the window with buttons for the mouse; each button also has a key.
STATUS_HEIGHT_PX = 50
STATUS_BACKGROUND = PANEL_COLOR
STATUS_TEXT_COLOR = WHITE
STATUS_ON_COLOR = RED
STATUS_PAUSE_COLOR = AMBER
STATUS_OFF_COLOR = CARD_COLOR
STATUS_LIVE_COLOR = GREEN
# During a pause the whole window gets an amber frame of this width, so the pause is seen at once.
PAUSE_FRAME_PX = 6
# (name, left edge, right edge) of every button in pixels.
BUTTONS = [("protocol", 8, 300), ("pause", 310, 560), ("quit", 570, 690)]
KEY_HELP = "c = calibrate"


def draw_markers(image, markers):
    """Draws the outline and the ID of every found marker into a copy of the camera image."""
    result = image.copy()
    for marker_id, corners in markers.items():
        points = corners.astype(np.int32)
        cv2.polylines(result, [points], True, MARKER_COLOR, 2, cv2.LINE_AA)
        put_text(result, str(marker_id), (int(points[0][0]), int(points[0][1]) - 8), 0.7, MARKER_COLOR, 2)
    return result


def compose_window(camera_image, field_image):
    """Puts both images next to each other: the camera image on the left, the straight field with its margin on the right.

    The field keeps its full size so the labels stay readable. The camera image
    gets the field's height, but at most the width that is left up to
    MAX_WINDOW_WIDTH; if it is lower than the field, the rest below is filled with the panel colour.
    Without a field (no homography yet) only the camera image is shown,
    with the text "Field not found" in the top-left corner.
    """
    if field_image is None:
        result = _fit(camera_image, camera_image.shape[0], MAX_WINDOW_WIDTH)
        return draw_notice(result, NO_FIELD_TEXT)
    height = field_image.shape[0]
    camera_scaled = _fit(camera_image, height, MAX_WINDOW_WIDTH - field_image.shape[1])
    # np.hstack needs two images of the same height.
    padded = np.full((height, camera_scaled.shape[1], 3), PANEL_COLOR, dtype=np.uint8)
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
        cv2.circle(result, center, 4, DOT_COLOR, -1, cv2.LINE_AA)
    return result


def _label_lines(item):
    """The text above and below the dot, e.g. (["red", "cube"], ["(0.42|0.71)"])."""
    return [item.color, SHAPE_LABELS.get(item.shape, item.shape)], [item.compact_position()]


def _label_box(item, center, scale, line_px):
    """The rectangle (left, top, right, bottom) that the label of one object covers."""
    above, below = _label_lines(item)
    width = max(cv2.getTextSize(line, FONT, scale, 1)[0][0] for line in above + below)
    half_width = width // 2 + LABEL_PADDING_PX
    top = center[1] - len(above) * line_px - DOT_GAP_PX // 2 - LABEL_PADDING_PX
    bottom = center[1] + len(below) * line_px + DOT_GAP_PX // 2 + LABEL_PADDING_PX
    return center[0] - half_width, top, center[0] + half_width, bottom


def _overlap(first, second):
    """True if two rectangles (left, top, right, bottom) cover a common pixel."""
    return first[0] < second[2] and second[0] < first[2] and first[1] < second[3] and second[1] < first[3]


def _draw_label(image, item, center, box, scale, line_px):
    """Draws the half transparent box with round corners and writes the lines centred into it."""
    left, top, right, bottom = (max(value, 0) for value in box)
    area = image[top:bottom, left:right]
    shaded = area.copy()
    rounded_box(shaded, (0, 0, right - left - 1, bottom - top - 1), LABEL_BOX_COLOR, LABEL_RADIUS_PX)
    # cv2.addWeighted mixes the box with the picture below it (script ch. 4.1.7).
    cv2.addWeighted(shaded, LABEL_BOX_ALPHA, area, 1 - LABEL_BOX_ALPHA, 0, dst=area)
    above, below = _label_lines(item)
    for number, line in enumerate(above):
        _put_centred(image, line, center[0], box[1] + LABEL_PADDING_PX + (number + 1) * line_px - 3, scale)
    for number, line in enumerate(below):
        _put_centred(image, line, center[0], center[1] + DOT_GAP_PX // 2 + (number + 1) * line_px - 3, scale)


def _put_centred(image, text, center_x, baseline_y, scale):
    """Writes white text so that its middle lies at center_x."""
    (width, _), _ = cv2.getTextSize(text, FONT, scale, 1)
    put_text(image, text, (center_x - width // 2, baseline_y), scale, LABEL_COLOR)


def draw_notice(image, text):
    """Shows a notice as a red pill in the top-left corner of a copy, e.g. "Hand in field - reports paused"."""
    result = image.copy()
    (width, _), _ = cv2.getTextSize(text, FONT, 0.6, 2)
    rounded_box(result, (10, 10, 10 + width + 28, 44), RED, 17)
    put_text(result, text, (24, 34), 0.6, WHITE, 2)
    return result


def add_status_bar(window, protocol_running, paused):
    """Puts a dark bar with buttons above the window; it shows whether the protocol runs and whether there is a pause.

    The protocol button is red while a protocol runs, the pause button amber during
    a pause, and a pause also gets an amber frame around the whole window. Next to the
    buttons a dot and one line say in words what the program is doing right now.
    """
    result = window.copy()
    if paused:
        result[:PAUSE_FRAME_PX] = STATUS_PAUSE_COLOR
        result[-PAUSE_FRAME_PX:] = STATUS_PAUSE_COLOR
        result[:, :PAUSE_FRAME_PX] = STATUS_PAUSE_COLOR
        result[:, -PAUSE_FRAME_PX:] = STATUS_PAUSE_COLOR
    bar = np.full((STATUS_HEIGHT_PX, window.shape[1], 3), STATUS_BACKGROUND, dtype=np.uint8)
    labels = {
        "protocol": ("REC  stop protocol (p)", STATUS_ON_COLOR) if protocol_running else ("Start protocol (p)", STATUS_OFF_COLOR),
        "pause": ("PAUSED  resume (u)", STATUS_PAUSE_COLOR) if paused else ("Pause (u)", STATUS_OFF_COLOR),
        "quit": ("Quit (q)", STATUS_OFF_COLOR),
    }
    text_y = STATUS_HEIGHT_PX // 2 + 6
    for name, left, right in BUTTONS:
        text, color = labels[name]
        rounded_box(bar, (left, 8, right, STATUS_HEIGHT_PX - 8), color, (STATUS_HEIGHT_PX - 16) // 2)
        (width, _), _ = cv2.getTextSize(text, FONT, 0.55, 1)
        put_text(bar, text, ((left + right - width) // 2, text_y), 0.55, STATUS_TEXT_COLOR, 2 if color != STATUS_OFF_COLOR else 1)
    state, state_color = ("PAUSED  nothing is detected or recorded", STATUS_PAUSE_COLOR) if paused else ("LIVE  detecting objects", STATUS_LIVE_COLOR)
    if protocol_running and not paused:
        state += ", protocol is recording"
    state_x = BUTTONS[-1][2] + 22
    cv2.circle(bar, (state_x, STATUS_HEIGHT_PX // 2), 6, state_color, -1, cv2.LINE_AA)
    put_text(bar, state, (state_x + 16, text_y), 0.55, WHITE, 1)
    (help_width, _), _ = cv2.getTextSize(KEY_HELP, FONT, 0.5, 1)
    put_text(bar, KEY_HELP, (bar.shape[1] - help_width - 14, text_y), 0.5, MUTED_COLOR, 1)
    return np.vstack([bar, result])


def button_at(x, y):
    """Name of the button at a mouse position in the window, e.g. "pause", or None."""
    if not 0 <= y < STATUS_HEIGHT_PX:
        return None
    for name, left, right in BUTTONS:
        if left <= x < right:
            return name
    return None


def rounded_box(image, box, color, radius):
    """A filled rectangle (left, top, right, bottom) with round corners: two rectangles and four circles."""
    left, top, right, bottom = box
    radius = min(radius, (right - left) // 2, (bottom - top) // 2)
    cv2.rectangle(image, (left + radius, top), (right - radius, bottom), color, -1)
    cv2.rectangle(image, (left, top + radius), (right, bottom - radius), color, -1)
    for x in (left + radius, right - radius):
        for y in (top + radius, bottom - radius):
            cv2.circle(image, (x, y), radius, color, -1, cv2.LINE_AA)


def put_text(image, text, origin, size, color, thickness=1):
    """Text with smooth edges (cv2.LINE_AA)."""
    cv2.putText(image, text, origin, FONT, size, color, thickness, cv2.LINE_AA)
