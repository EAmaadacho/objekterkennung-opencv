"""Setup mode before the start: checks camera and light live and shows them as five traffic lights (A6, A7).

The five checks: all four markers seen, field large enough, camera straight above the
field, field in the middle of the image, and paper bright and evenly lit.
The file has three parts: the checks, the drawing of the setup window, and the loop that runs it.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from objekterkennung.capture.board import (
    MARKER_IDS,
    find_homography,
    find_markers,
    margin_px,
    straighten_field,
    turn_upright,
    upright_turns,
)
from objekterkennung.capture.camera import read_frame
from objekterkennung.output.display import (
    AMBER,
    CARD_COLOR,
    FONT,
    GREEN,
    MUTED_COLOR,
    PANEL_COLOR,
    RED,
    WHITE,
    draw_markers,
    put_text,
    rounded_box,
    shorten,
    show,
    to_image,
)

OK = "ok"
WARN = "warn"
BAD = "bad"
LEVEL_COLORS = {OK: GREEN, WARN: AMBER, BAD: RED}
TEXT_COLOR = WHITE
# Size: the field should fill this share of the image height. If the markers lie above and
# below the field in the image, it can fill at most 70 % (180 of 256 mm) before a marker
# leaves the image; with the paper turned by 90 degrees they lie left and right, and the
# field can fill more. Above SIZE_MAX a block at the edge of the field leaves the image.
# The setup used for all tests reached 36-38 % (camera arm at its lowest); with that and a
# green light, 97 % of the blocks were found (13.11.). Below 25 % a cube is less than 25 px
# wide in the camera image and the seam between two flush blocks disappears.
SIZE_GOOD = 0.35
SIZE_MIN = 0.25
SIZE_MAX = 0.78
# Straight: opposite field edges may differ by this share in length, the field may be turned this much.
# With the camera about 47 cm above the field, 5 % is a lean of about 7 degrees: the tops of
# the 32 mm blocks then shift by about 4 mm. The webcam lens alone bends the field edges by
# 1-2 %, and the best real setup reached 2.5-3 %, so a stricter limit could not be reached.
TILT_GOOD = 0.05
TILT_MAX = 0.1
TURN_GOOD_DEG = 2
TURN_MAX_DEG = 6
# Centre: distance of the field centre from the image centre, as a share of the image size.
# Strict, because a camera beside the middle shifts the tops of high blocks outwards.
OFFSET_GOOD = 0.05
OFFSET_MAX = 0.15
# Light, measured on the white paper (low saturation) in the straight field. The target band
# for a green light comes from config.json (section "light"): in the light-level test (13.11.)
# the paper went from 173 at the strongest to 145 at the weakest lamp level, about 3 grey levels
# per step. In the test of 15.11. a pale red cube was lost at 172-176 and the green cuboid in
# the shadow of the ball at 137-139; 155-162 (darkest lamp level) worked best. A single level
# cannot be told apart, because the camera adjusts its exposure. Below BRIGHTNESS_MIN the light
# is clearly too weak.
PAPER_SATURATION_MAX = 40
BRIGHTNESS_MIN = 90
BRIGHTNESS_TOO_HIGH = 235
# The darkest of 3 x 3 field parts compared with the brightest one. In the recordings even
# 61 % caused no error that came from the light (a pale block in a dark corner), so yellow
# starts only at 65 %.
EVENNESS_GOOD = 0.65
EVENNESS_MIN = 0.6
# Share of overexposed pixels (brightness 250 or more): there a block loses its colour.
GLARE_VALUE = 250
GLARE_GOOD = 0.01
GLARE_MAX = 0.05
GRID_PARTS = 3
# Radius of the spirit level around the image centre; its outer ring stands for TILT_MAX.
LEVEL_RADIUS_PX = 70
# Strength of the see-through layers (cv2.addWeighted, script ch. 4.1.7): field tint, guide lines, level.
FIELD_TINT = 0.18
GUIDE_ALPHA = 0.5
LEVEL_ALPHA = 0.35
WINDOW_MAX_WIDTH = 1400
WINDOW_MAX_HEIGHT = 760
PANEL_WIDTH = 440
CARD_TOP = 84
CARD_HEIGHT = 56
CARD_GAP = 8
START_BUTTON = (20, 424, 420, 468)  # left, top, right, bottom in the window
GUIDES_BUTTON = (20, 536, 420, 576)
START_KEYS = (13, 10)  # Enter on Windows and Linux
QUIT_KEY = "q"
# Hides all guides on the camera image, so the blocks can be seen as they look in the light.
# Both h and H count, so it also works with Caps Lock on; the button below START does the same.
GUIDES_KEYS = ("h", "H")


@dataclass(frozen=True)
class Check:
    """One traffic light of the setup mode, e.g. Check("Size", "bad", "fills 28% of image height - move camera lower")."""

    name: str
    level: str
    text: str


# Part 1: the five checks, each on one camera image.


def check_setup(frame, settings):
    """Runs all five checks on one camera image; returns the checks and the field corners in the image (or None)."""
    markers = find_markers(frame)
    found = [marker_id for marker_id in MARKER_IDS if marker_id in markers]
    if len(found) < len(MARKER_IDS):
        missing = ", ".join(str(marker_id) for marker_id in MARKER_IDS if marker_id not in found)
        checks = [Check("Markers", BAD, f"{len(found)} of 4 found - missing: {missing}")]
        checks += [Check(name, BAD, "needs all 4 markers") for name in ("Size", "Straight", "Centre", "Light")]
        return checks, None
    board = settings.board
    homography = find_homography(frame, board)
    corners = field_corners(homography, board)
    checks = [
        Check("Markers", OK, "4 of 4 found"),
        check_size(corners, frame.shape),
        check_straight(corners),
        check_centre(corners, frame.shape),
    ]
    field_image = straighten_field(frame, homography, board)
    checks.append(check_light(field_image, settings.light))
    return checks, corners


def field_corners(homography, board):
    """The four field corners in the camera image: top-left, top-right, bottom-right, bottom-left.

    The homography maps the camera image to the straight field, so its inverse maps back.
    """
    margin = margin_px(board)
    far = margin + board.field_size_in_image_px
    field = np.float32([[margin, margin], [far, margin], [far, far], [margin, far]]).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(field, np.linalg.inv(homography)).reshape(4, 2)


def field_centre_lines(corners):
    """The two lines through the middle of the field in the camera image, as (start, end) pairs.

    In a slanted view the middle of each edge is not where it seems, so the lines are found
    with homogeneous coordinates: np.cross of two points is the line through them, np.cross
    of two lines is their crossing point. The centre is where the diagonals cross. Each centre
    line also runs through the point where the two field edges beside it meet (far away, or
    at infinity when they are parallel), and it ends on the other two edges.
    """
    top_left, top_right, bottom_right, bottom_left = (np.append(point, 1.0) for point in corners)
    centre = np.cross(np.cross(top_left, bottom_right), np.cross(top_right, bottom_left))
    top, bottom = np.cross(top_left, top_right), np.cross(bottom_left, bottom_right)
    left, right = np.cross(top_left, bottom_left), np.cross(top_right, bottom_right)
    lines = []
    for side_a, side_b, end_a, end_b in ((left, right, top, bottom), (top, bottom, left, right)):
        centre_line = np.cross(centre, np.cross(side_a, side_b))
        start, end = np.cross(centre_line, end_a), np.cross(centre_line, end_b)
        lines.append((start[:2] / start[2], end[:2] / end[2]))
    return lines


def check_size(corners, shape):
    """How much of the image height the field fills."""
    height = (_length(corners[0], corners[3]) + _length(corners[1], corners[2])) / 2
    share = height / min(shape[:2])
    text = f"fills {share:.0%} of image height"
    if share > SIZE_MAX:
        return Check("Size", WARN, f"{text} - move camera higher")
    if share >= SIZE_GOOD:
        return Check("Size", OK, text)
    return Check("Size", WARN if share >= SIZE_MIN else BAD, f"{text} - move camera lower")


def check_straight(corners):
    """Is the camera tilted (one side of the field looks shorter) or the field turned in the image?

    Everything is said in the directions of the image on the screen, because the paper may
    lie turned under the camera. A field side that looks shorter is further away: the
    camera does not look straight down, but leans towards that side.
    """
    edges = _edges_in_image(corners)
    lean_x, lean_y = camera_lean(corners)
    horizontal, vertical = abs(lean_x), abs(lean_y)
    if max(vertical, horizontal) > TILT_GOOD:
        if vertical >= horizontal:
            far, near, share = ("top", "bottom", vertical) if lean_y < 0 else ("bottom", "top", vertical)
        else:
            far, near, share = ("left", "right", horizontal) if lean_x < 0 else ("right", "left", horizontal)
        level = WARN if share <= TILT_MAX else BAD
        return Check("Straight", level, f"{far} side {share:.0%} smaller - tilt lens a bit to the {near}")
    dx, dy = edges["top"][1] - edges["top"][0]
    # The paper may lie turned by 90 degrees under the camera, so only the rest up to 45 degrees counts.
    turn = (float(np.degrees(np.arctan2(dy, dx))) + 45) % 90 - 45
    if abs(turn) > TURN_GOOD_DEG:
        level = WARN if abs(turn) <= TURN_MAX_DEG else BAD
        direction = "anticlockwise" if turn > 0 else "clockwise"
        return Check("Straight", level, f"field turned by {abs(turn):.0f} deg - turn paper {direction}")
    return Check("Straight", OK, "camera looks straight down")


def camera_lean(corners):
    """In which direction and how much the camera leans, in image directions, e.g. (0.0, -0.03).

    The side the camera leans towards is further away and looks shorter. x > 0 means it
    leans to the right (the right side is shorter), y < 0 that it leans up (the top side
    is shorter). (0, 0) means the camera looks straight down.
    """
    edges = _edges_in_image(corners)
    top, bottom, left, right = (_length(*edges[side]) for side in ("top", "bottom", "left", "right"))
    return (left - right) / max(left, right), (top - bottom) / max(top, bottom)


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


def check_light(field_image, light):
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
    elif brightness < light.paper_brightness_min or brightness > light.paper_brightness_max or evenness < EVENNESS_GOOD or glare > GLARE_GOOD:
        level = WARN
    else:
        return Check("Light", OK, text)
    if brightness < light.paper_brightness_min:
        hint = f"too dark: lamp brighter (target {light.paper_brightness_min}-{light.paper_brightness_max})"
    elif glare > GLARE_GOOD or brightness > BRIGHTNESS_TOO_HIGH:
        hint = "lamp darker or higher (glare)"
    elif evenness < EVENNESS_GOOD:
        hint = "lamp more to the middle (uneven)"
    else:
        hint = f"lamp one step darker (target {light.paper_brightness_min}-{light.paper_brightness_max})"
    return Check("Light", level, f"{text} - {hint}")


# Part 2: drawing the setup window.


def draw_setup(frame, checks, corners, guides_shown=True):
    """A panel with the five traffic lights on the left, the camera image with guide lines on the right.

    The panel is placed next to the image (np.hstack), so it never hides the field.
    With guides_shown False the camera image stays clean: no crosses, no tint, no spirit
    level and no arrows, only the marker outlines. The traffic lights keep running.
    """
    # Only the square in the middle is shown, so a turned (upright) camera image is as large as a normal one.
    side = min(frame.shape[:2])
    top, left = (frame.shape[0] - side) // 2, (frame.shape[1] - side) // 2
    square = draw_markers(frame, find_markers(frame))[top : top + side, left : left + side]
    if corners is not None:
        corners = corners - [left, top]
    scale = min((WINDOW_MAX_WIDTH - PANEL_WIDTH) / side, WINDOW_MAX_HEIGHT / side)
    image = cv2.resize(square, None, fx=scale, fy=scale)
    height, width = image.shape[:2]
    if not guides_shown:
        return np.hstack([_draw_panel(checks, height, guides_shown), image])
    # Guide lines, drawn see-through: a cross through the middle and the corners of the target square.
    guides = image.copy()
    cv2.line(guides, (width // 2, 0), (width // 2, height), TEXT_COLOR, 1, cv2.LINE_AA)
    cv2.line(guides, (0, height // 2), (width, height // 2), TEXT_COLOR, 1, cv2.LINE_AA)
    half = round(height * (SIZE_GOOD + SIZE_MAX) / 4)
    for x_sign in (-1, 1):
        for y_sign in (-1, 1):
            corner = (width // 2 + x_sign * half, height // 2 + y_sign * half)
            cv2.line(guides, corner, (corner[0] - x_sign * 30, corner[1]), TEXT_COLOR, 2, cv2.LINE_AA)
            cv2.line(guides, corner, (corner[0], corner[1] - y_sign * 30), TEXT_COLOR, 2, cv2.LINE_AA)
    cv2.addWeighted(guides, GUIDE_ALPHA, image, 1 - GUIDE_ALPHA, 0, dst=image)
    if corners is not None:
        color = LEVEL_COLORS[_worst(checks[1:4])]
        outline = np.int32(np.round(corners * scale))
        tint = image.copy()
        cv2.fillPoly(tint, [outline], color)
        cv2.addWeighted(tint, FIELD_TINT, image, 1 - FIELD_TINT, 0, dst=image)
        cv2.polylines(image, [outline], True, color, 2, cv2.LINE_AA)
        # A cross through the middle of the field, in the same colour: when the camera is
        # centred, it lies on the light cross through the middle of the image.
        for start, end in field_centre_lines(corners * scale):
            cv2.line(image, np.int32(np.round(start)), np.int32(np.round(end)), color, 2, cv2.LINE_AA)
        _draw_level(image, camera_lean(corners))
        if checks[3].level != OK:
            _draw_move_arrow(image, corners.mean(axis=0) * scale, LEVEL_COLORS[checks[3].level])
    return np.hstack([_draw_panel(checks, height, guides_shown), image])


def _draw_move_arrow(image, field_centre, color):
    """An arrow from the image centre towards the field centre: the way the camera has to move.

    If the field lies left of the middle, the camera hangs too far right; moving it left
    moves the field to the right in the image, until both crosses lie on each other.
    """
    centre = np.array([image.shape[1] // 2, image.shape[0] // 2], dtype=float)
    start = centre
    end = tuple(np.int32(np.round(field_centre)))
    cv2.arrowedLine(image, tuple(np.int32(np.round(start))), end, PANEL_COLOR, 7, cv2.LINE_AA, tipLength=0.25)
    cv2.arrowedLine(image, tuple(np.int32(np.round(start))), end, color, 4, cv2.LINE_AA, tipLength=0.25)
    text = "move the camera this way"
    (width, _), _ = cv2.getTextSize(text, FONT, 0.45, 1)
    # The label sits beyond the tip of the arrow, so it does not cover the spirit level.
    direction = (field_centre - centre) / max(1.0, float(np.hypot(*(field_centre - centre))))
    middle = field_centre + direction * (width // 2 + 30)
    left = int(np.clip(middle[0] - width // 2 - 12, 0, image.shape[1] - width - 24))
    top = int(np.clip(middle[1] - 13, 0, image.shape[0] - 26))
    rounded_box(image, (left, top, left + width + 24, top + 26), PANEL_COLOR, 13)
    put_text(image, text, (left + 12, top + 18), 0.45, TEXT_COLOR)


def _draw_panel(checks, height, guides_shown=True):
    """The left panel: one card per traffic light, the START button and the keys."""
    panel = np.full((height, PANEL_WIDTH, 3), PANEL_COLOR, dtype=np.uint8)
    put_text(panel, "Setup", (20, 40), 0.95, TEXT_COLOR, 2)
    put_text(panel, "camera and light", (122, 40), 0.55, MUTED_COLOR)
    for number, check in enumerate(checks):
        top = CARD_TOP + number * (CARD_HEIGHT + CARD_GAP)
        rounded_box(panel, (20, top, PANEL_WIDTH - 20, top + CARD_HEIGHT), CARD_COLOR, 10)
        color = LEVEL_COLORS[check.level]
        cv2.circle(panel, (44, top + 28), 13, color, 2, cv2.LINE_AA)
        cv2.circle(panel, (44, top + 28), 7, color, -1, cv2.LINE_AA)
        put_text(panel, check.name, (70, top + 24), 0.6, TEXT_COLOR, 2)
        # The hint goes in a second line, so it never runs out of the panel.
        status, _, hint = check.text.partition(" - ")
        put_text(panel, status, (170, top + 24), 0.45, MUTED_COLOR)
        if hint:
            put_text(panel, shorten(hint, PANEL_WIDTH - 100, 0.45), (70, top + 46), 0.45, color)
    ready = _worst(checks) == OK
    left, top, right, bottom = START_BUTTON
    rounded_box(panel, START_BUTTON, LEVEL_COLORS[OK] if ready else CARD_COLOR, 22)
    put_text(panel, "START", (left + 160, bottom - 15), 0.7, TEXT_COLOR if ready else MUTED_COLOR, 2)
    note = "all green - ready to start" if ready else "fix the yellow and red points first"
    put_text(panel, note, (left, bottom + 26), 0.45, TEXT_COLOR)
    put_text(panel, "Enter = start    h = guides    q = quit", (left, bottom + 50), 0.45, MUTED_COLOR)
    left, top, right, bottom = GUIDES_BUTTON
    rounded_box(panel, GUIDES_BUTTON, CARD_COLOR, 20)
    label = "Hide guides (h)" if guides_shown else "Show guides (h)"
    (label_width, _), _ = cv2.getTextSize(label, FONT, 0.55, 1)
    put_text(panel, label, ((left + right - label_width) // 2, bottom - 14), 0.55, TEXT_COLOR)
    return panel


def _draw_level(image, lean):
    """A spirit level around the image centre: the dot shows where the camera leans.

    The dot moves the same way as the lens when the camera is tilted. Inside the green
    disc the camera looks straight down (TILT_GOOD), the outer ring is TILT_MAX. An arrow
    from the dot to the middle shows which way to tilt the lens.
    """
    centre = np.array([image.shape[1] // 2, image.shape[0] // 2])
    good = round(LEVEL_RADIUS_PX * TILT_GOOD / TILT_MAX)
    # The dot stays on the outer ring when the camera leans even more.
    shift = np.clip(np.array(lean) / TILT_MAX, -1, 1)
    dot = centre + LEVEL_RADIUS_PX * shift / max(1.0, float(np.hypot(*shift)))
    lean_share = max(abs(lean[0]), abs(lean[1]))
    level = OK if lean_share <= TILT_GOOD else WARN if lean_share <= TILT_MAX else BAD
    hud = image.copy()
    cv2.circle(hud, tuple(centre), LEVEL_RADIUS_PX + 18, PANEL_COLOR, -1, cv2.LINE_AA)
    cv2.circle(hud, tuple(centre), good, LEVEL_COLORS[OK], -1, cv2.LINE_AA)
    cv2.addWeighted(hud, LEVEL_ALPHA, image, 1 - LEVEL_ALPHA, 0, dst=image)
    cv2.circle(image, tuple(centre), LEVEL_RADIUS_PX, MUTED_COLOR, 1, cv2.LINE_AA)
    cv2.circle(image, tuple(centre), good, LEVEL_COLORS[OK], 2, cv2.LINE_AA)
    for direction in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        start = centre + np.multiply(direction, LEVEL_RADIUS_PX - 6)
        cv2.line(image, tuple(start), tuple(centre + np.multiply(direction, LEVEL_RADIUS_PX + 6)), MUTED_COLOR, 1, cv2.LINE_AA)
    dot_px = tuple(np.int32(np.round(dot)))
    if level != OK:
        cv2.arrowedLine(image, dot_px, tuple(np.int32(np.round(centre + (dot - centre) * 0.2))),
                        LEVEL_COLORS[level], 3, cv2.LINE_AA, tipLength=0.35)
    cv2.circle(image, dot_px, 10, LEVEL_COLORS[level], -1, cv2.LINE_AA)
    cv2.circle(image, dot_px, 10, TEXT_COLOR, 2, cv2.LINE_AA)
    if level == OK:
        return
    text = "tilt the lens the way the arrow points"
    (text_width, _), _ = cv2.getTextSize(text, FONT, 0.45, 1)
    top = centre[1] + LEVEL_RADIUS_PX + 26
    rounded_box(image, (centre[0] - text_width // 2 - 12, top, centre[0] + text_width // 2 + 12, top + 26), PANEL_COLOR, 13)
    put_text(image, text, (centre[0] - text_width // 2, top + 18), 0.45, TEXT_COLOR)


# Part 3: the loop of the setup mode.


def is_start_click(x, y):
    """True if a mouse click in the setup window hit the START button."""
    return _inside(START_BUTTON, x, y)


def is_guides_click(x, y):
    """True if a mouse click in the setup window hit the button that hides or shows the guides."""
    return _inside(GUIDES_BUTTON, x, y)


def run_setup(capture, settings, window_name):
    """Shows the setup mode until Enter or START is pressed (returns True) or q (returns False); h hides the guides."""
    clicks = []
    cv2.setMouseCallback(window_name, lambda event, x, y, flags, param: _remember_click(event, window_name, x, y, clicks))
    turns = 0
    guides_shown = True
    while True:
        frame = read_frame(capture)
        # The image is turned so the field is seen from the front; then "left" or "down" on the
        # screen is the same as at the table. Without all markers the last turn is kept.
        homography = find_homography(frame, settings.board)
        if homography is not None:
            turns = upright_turns(homography, settings.board)
        frame = turn_upright(frame, turns)
        checks, corners = check_setup(frame, settings)
        show(window_name, draw_setup(frame, checks, corners, guides_shown))
        # The keys only arrive while the setup window is the active window (click into it once).
        key = cv2.waitKey(1) & 0xFF
        clicked = clicks.pop(0) if clicks else None
        if key in START_KEYS or clicked == "start":
            return True
        if key == ord(QUIT_KEY):
            return False
        if key in (ord(name) for name in GUIDES_KEYS) or clicked == "guides":
            guides_shown = not guides_shown


def _remember_click(event, window_name, x, y, clicks):
    """Keeps a left click on the START button or on the guides button, as "start" or "guides"."""
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    x, y = to_image(window_name, x, y)
    if is_start_click(x, y):
        clicks.append("start")
    elif is_guides_click(x, y):
        clicks.append("guides")


# Small helpers used by the parts above.


def _worst(checks):
    """The worst level of some checks: bad before warn before ok."""
    levels = [check.level for check in checks]
    return BAD if BAD in levels else WARN if WARN in levels else OK


def _edges_in_image(corners):
    """The four field edges named by where they lie on the screen, e.g. {"top": (point, point), ...}.

    Each edge runs from left to right or from top to bottom.
    """
    edges = [tuple(sorted((corners[i], corners[(i + 1) % 4]), key=lambda point: (point[0], point[1]))) for i in range(4)]
    middles = [(first + second) / 2 for first, second in edges]
    by_y = sorted(range(4), key=lambda i: middles[i][1])
    by_x = sorted(range(4), key=lambda i: middles[i][0])
    return {"top": edges[by_y[0]], "bottom": edges[by_y[3]], "left": edges[by_x[0]], "right": edges[by_x[3]]}


def _inside(box, x, y):
    """True if the point (x, y) lies in the box (left, top, right, bottom)."""
    left, top, right, bottom = box
    return left <= x < right and top <= y < bottom


def _length(first, second):
    """Length of the line between two points in pixels."""
    return float(np.hypot(*(second - first)))
