"""Setup mode before the start: checks markers, size, straightness, position, light and glare on the blocks live (A6, A7)."""

from dataclasses import dataclass

import cv2
import numpy as np

from objekterkennung.board import MARKER_IDS, find_homography, find_markers, margin_px, straighten_field
from objekterkennung.camera import read_frame
from objekterkennung.display import draw_markers

OK = "ok"
WARN = "warn"
BAD = "bad"
LEVEL_COLORS = {OK: (0, 200, 0), WARN: (0, 200, 255), BAD: (70, 70, 255)}
# Size: the field should fill this share of the image height. With the markers above and
# below it, the field can fill at most about 75 % before a marker leaves the image.
SIZE_GOOD = 0.5
SIZE_MIN = 0.35
SIZE_MAX = 0.78
# Straight: opposite field edges may differ by this share in length, the field may be turned this much.
TILT_GOOD = 0.03
TILT_MAX = 0.08
TURN_GOOD_DEG = 2
TURN_MAX_DEG = 6
# Centre: distance of the field centre from the image centre, as a share of the image size.
# Strict, because a camera beside the middle shifts the tops of high blocks outwards.
OFFSET_GOOD = 0.05
OFFSET_MAX = 0.15
# Light, measured on the white paper (low saturation) in the straight field.
PAPER_SATURATION_MAX = 40
BRIGHTNESS_GOOD = 130
BRIGHTNESS_MIN = 90
BRIGHTNESS_TOO_HIGH = 235
# The darkest of 3 x 3 field parts compared with the brightest one.
EVENNESS_GOOD = 0.75
EVENNESS_MIN = 0.6
# Share of overexposed pixels (brightness 250 or more): there a block loses its colour.
GLARE_VALUE = 250
GLARE_GOOD = 0.01
GLARE_MAX = 0.05
GRID_PARTS = 3
# Glare on the blocks: a block is every coloured pixel; around it this many pixels are counted
# as the block too, because a reflection has almost no colour and leaves a hole or a bright rim.
PIECE_SATURATION_MIN = 60
PIECE_GROW_PX = 9
PIECE_PIXELS_MIN = 2000
PIECE_GLARE_GOOD = 0.01
PIECE_GLARE_MAX = 0.03

WINDOW_MAX_WIDTH = 1400
WINDOW_MAX_HEIGHT = 760
PANEL_WIDTH = 440
LINE_HEIGHT = 30
START_BUTTON = (20, 430, 300, 470)  # left, top, right, bottom in the window
START_KEYS = (13, 10)  # Enter on Windows and Linux
QUIT_KEY = "q"


@dataclass(frozen=True)
class Check:
    """One traffic light of the setup mode, e.g. Check("Size", "bad", "fills 28% of image height - move camera lower")."""

    name: str
    level: str
    text: str


def check_setup(frame, board):
    """Runs all six checks on one camera image; returns the checks and the field corners in the image (or None)."""
    markers = find_markers(frame)
    found = [marker_id for marker_id in MARKER_IDS if marker_id in markers]
    if len(found) < len(MARKER_IDS):
        missing = ", ".join(str(marker_id) for marker_id in MARKER_IDS if marker_id not in found)
        checks = [Check("Markers", BAD, f"{len(found)} of 4 found - missing: {missing}")]
        checks += [Check(name, BAD, "needs all 4 markers") for name in ("Size", "Straight", "Centre", "Light", "Glare")]
        return checks, None
    homography = find_homography(frame, board)
    corners = field_corners(homography, board)
    checks = [
        Check("Markers", OK, "4 of 4 found"),
        check_size(corners, frame.shape),
        check_straight(corners),
        check_centre(corners, frame.shape),
    ]
    field_image = straighten_field(frame, homography, board)
    checks += [check_light(field_image), check_piece_glare(field_image)]
    return checks, corners


def field_corners(homography, board):
    """The four field corners in the camera image: top-left, top-right, bottom-right, bottom-left.

    The homography maps the camera image to the straight field, so its inverse maps back.
    """
    margin = margin_px(board)
    far = margin + board.warp_px
    field = np.float32([[margin, margin], [far, margin], [far, far], [margin, far]]).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(field, np.linalg.inv(homography)).reshape(4, 2)


def check_size(corners, shape):
    """How much of the image height the field fills."""
    height = (_length(corners[0], corners[3]) + _length(corners[1], corners[2])) / 2
    share = height / shape[0]
    text = f"fills {share:.0%} of image height"
    if share > SIZE_MAX:
        return Check("Size", WARN, f"{text} - move camera higher")
    if share >= SIZE_GOOD:
        return Check("Size", OK, text)
    return Check("Size", WARN if share >= SIZE_MIN else BAD, f"{text} - move camera lower")


def check_straight(corners):
    """Compares opposite field edges (camera tilted?) and the angle of the top edge (field turned?)."""
    top, bottom = _length(corners[0], corners[1]), _length(corners[3], corners[2])
    left, right = _length(corners[0], corners[3]), _length(corners[1], corners[2])
    vertical = 1 - min(top, bottom) / max(top, bottom)
    horizontal = 1 - min(left, right) / max(left, right)
    dx, dy = corners[1] - corners[0]
    turn = float(np.degrees(np.arctan2(dy, dx)))
    if max(vertical, horizontal) > TILT_GOOD:
        if vertical >= horizontal:
            side = "top" if top < bottom else "bottom"
        else:
            side = "left" if left < right else "right"
        level = WARN if max(vertical, horizontal) <= TILT_MAX else BAD
        return Check("Straight", level, f"{side} edge looks smaller - move camera towards the {side}")
    if abs(turn) > TURN_GOOD_DEG:
        level = WARN if abs(turn) <= TURN_MAX_DEG else BAD
        return Check("Straight", level, f"field is turned by {turn:.0f} deg - turn paper or camera")
    return Check("Straight", OK, "camera looks straight down")


def check_centre(corners, shape):
    """Distance of the field centre from the image centre."""
    centre_x, centre_y = corners.mean(axis=0)
    offset_x = centre_x / shape[1] - 0.5
    offset_y = centre_y / shape[0] - 0.5
    offset = max(abs(offset_x), abs(offset_y))
    if offset <= OFFSET_GOOD:
        return Check("Centre", OK, "in the middle of the image")
    if abs(offset_x) >= abs(offset_y):
        direction = "left" if offset_x < 0 else "right"
    else:
        direction = "up" if offset_y < 0 else "down"
    level = WARN if offset <= OFFSET_MAX else BAD
    return Check("Centre", level, f"field is too far {direction} - move camera {direction}")


def check_light(field_image):
    """Brightness of the paper, how even it is over the field and how much is overexposed."""
    hsv = cv2.cvtColor(field_image, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2]
    paper = hsv[:, :, 1] <= PAPER_SATURATION_MAX
    if not paper.any():
        return Check("Light", BAD, "no white paper found")
    brightness = float(np.median(value[paper]))
    parts = []
    step = field_image.shape[0] // GRID_PARTS
    for row in range(GRID_PARTS):
        for column in range(GRID_PARTS):
            window = (slice(row * step, (row + 1) * step), slice(column * step, (column + 1) * step))
            if paper[window].any():
                parts.append(float(np.median(value[window][paper[window]])))
    evenness = min(parts) / max(parts)
    glare = float(np.mean(value >= GLARE_VALUE))
    text = f"paper {brightness:.0f}, even {evenness:.0%}, glare {glare:.0%}"
    if brightness < BRIGHTNESS_MIN or evenness < EVENNESS_MIN or glare > GLARE_MAX:
        level = BAD
    elif brightness < BRIGHTNESS_GOOD or brightness > BRIGHTNESS_TOO_HIGH or evenness < EVENNESS_GOOD or glare > GLARE_GOOD:
        level = WARN
    else:
        return Check("Light", OK, text)
    if brightness < BRIGHTNESS_GOOD:
        hint = "lamp brighter (too dark)"
    elif glare > GLARE_GOOD or brightness > BRIGHTNESS_TOO_HIGH:
        hint = "lamp darker or higher (glare)"
    else:
        hint = "lamp more to the middle (uneven)"
    return Check("Light", level, f"{text} - {hint}")


def check_piece_glare(field_image):
    """Share of overexposed pixels on the blocks; some blocks must lie in the field for this check.

    Overexposed pixels have lost their colour, so the colour masks miss them and a
    block looks smaller or falls apart.
    """
    hsv = cv2.cvtColor(field_image, cv2.COLOR_BGR2HSV)
    coloured = np.where(hsv[:, :, 1] >= PIECE_SATURATION_MIN, 255, 0).astype(np.uint8)
    pieces = cv2.dilate(coloured, np.ones((PIECE_GROW_PX, PIECE_GROW_PX), np.uint8)) > 0
    if np.count_nonzero(pieces) < PIECE_PIXELS_MIN:
        return Check("Glare", WARN, "no blocks found - put some blocks in the field")
    glare = float(np.mean(hsv[:, :, 2][pieces] >= GLARE_VALUE))
    text = f"glare on blocks {glare:.1%}"
    if glare <= PIECE_GLARE_GOOD:
        return Check("Glare", OK, text)
    level = WARN if glare <= PIECE_GLARE_MAX else BAD
    return Check("Glare", level, f"{text} - lamp higher, darker or more to the side")


def draw_setup(frame, checks, corners):
    """A panel with the five traffic lights on the left, the camera image with guide lines on the right.

    The panel is placed next to the image (np.hstack), so it never hides the field.
    """
    scale = min((WINDOW_MAX_WIDTH - PANEL_WIDTH) / frame.shape[1], WINDOW_MAX_HEIGHT / frame.shape[0])
    image = cv2.resize(draw_markers(frame, find_markers(frame)), None, fx=scale, fy=scale)
    height, width = image.shape[:2]
    guide = (200, 200, 200)
    # Guide lines: a cross through the middle and the target square for the field.
    cv2.line(image, (width // 2, 0), (width // 2, height), guide, 1)
    cv2.line(image, (0, height // 2), (width, height // 2), guide, 1)
    half = round(height * (SIZE_GOOD + SIZE_MAX) / 4)
    cv2.rectangle(image, (width // 2 - half, height // 2 - half), (width // 2 + half, height // 2 + half), guide, 1)
    if corners is not None:
        worst = _worst(checks[1:4])
        cv2.polylines(image, [np.int32(corners * scale)], True, LEVEL_COLORS[worst], 3)
    panel = np.full((height, PANEL_WIDTH, 3), 40, dtype=np.uint8)
    cv2.putText(panel, "SETUP: camera and light", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    for number, check in enumerate(checks):
        y = 80 + number * 2 * LINE_HEIGHT
        cv2.circle(panel, (32, y - 6), 11, LEVEL_COLORS[check.level], -1)
        cv2.putText(panel, check.name, (54, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        # The hint goes in a second line, so it never runs out of the panel.
        status, _, hint = check.text.partition(" - ")
        cv2.putText(panel, status, (150, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        if hint:
            cv2.putText(panel, hint, (150, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, LEVEL_COLORS[check.level], 1)
    left, top, right, bottom = START_BUTTON
    ready = _worst(checks) == OK
    panel[top:bottom, left:right] = LEVEL_COLORS[OK] if ready else (90, 90, 90)
    cv2.putText(panel, "START (Enter)", (left + 60, bottom - 13), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    note = "all green - ready to start" if ready else "fix the red and yellow points first"
    cv2.putText(panel, note, (left, bottom + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(panel, "q = quit", (left, bottom + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return np.hstack([panel, image])


def is_start_click(x, y):
    """True if a mouse click in the setup window hit the START button."""
    left, top, right, bottom = START_BUTTON
    return left <= x < right and top <= y < bottom


def run_setup(capture, board, window_name):
    """Shows the setup mode until Enter or START is pressed (returns True) or q (returns False)."""
    clicks = []
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, lambda event, x, y, flags, param: _remember_start(event, x, y, clicks))
    while True:
        frame = read_frame(capture)
        checks, corners = check_setup(frame, board)
        cv2.imshow(window_name, draw_setup(frame, checks, corners))
        key = cv2.waitKey(1) & 0xFF
        if key in START_KEYS or clicks:
            return True
        if key == ord(QUIT_KEY):
            return False


def _remember_start(event, x, y, clicks):
    """Keeps a left click on the START button."""
    if event == cv2.EVENT_LBUTTONDOWN and is_start_click(x, y):
        clicks.append((x, y))


def _worst(checks):
    """The worst level of some checks: bad before warn before ok."""
    levels = [check.level for check in checks]
    return BAD if BAD in levels else WARN if WARN in levels else OK


def _length(first, second):
    return float(np.hypot(*(second - first)))
