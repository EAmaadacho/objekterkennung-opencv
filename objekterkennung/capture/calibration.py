"""Calibration mode: adjusts the colour ranges to the light with sliders and saves them in config.json (A7)."""

import json
import re
from dataclasses import replace

import cv2
import numpy as np

from objekterkennung.capture.board import find_homography, straighten_field
from objekterkennung.capture.camera import read_frame
from objekterkennung.config import COLOR_NAMES, DEFAULT_CONFIG_PATH, load_settings
from objekterkennung.detection.segmentation import HUE_MAX, color_mask

WINDOW_NAME = "Calibration"
COLOR_SLIDER = "Colour 0-3"
HUE_LOW_SLIDER = "Hue from"
HUE_HIGH_SLIDER = "Hue to"
SATURATION_SLIDER = "Saturation min"
VALUE_SLIDER = "Brightness min"
SAVE_KEY = "w"
BACK_KEY = "q"
HELP_TEXT = "w = save, q = back"


def run_calibration(capture, path=DEFAULT_CONFIG_PATH):
    """Shows the field and the mask of one colour; the sliders change the HSV range live.

    Hue and minimum saturation belong to the chosen colour, the minimum brightness to all colours.

    "Colour" selects red (0), yellow (1), green (2) or blue (3). The mask on the
    right should be white exactly where the blocks of that colour lie.
    """
    settings = load_settings(path)
    hue_ranges = dict(settings.hue_ranges)
    saturation_min = dict(settings.saturation_min)
    cv2.namedWindow(WINDOW_NAME)
    cv2.createTrackbar(COLOR_SLIDER, WINDOW_NAME, 0, len(COLOR_NAMES) - 1, _ignore)
    cv2.createTrackbar(HUE_LOW_SLIDER, WINDOW_NAME, 0, HUE_MAX, _ignore)
    cv2.createTrackbar(HUE_HIGH_SLIDER, WINDOW_NAME, 0, HUE_MAX, _ignore)
    cv2.createTrackbar(SATURATION_SLIDER, WINDOW_NAME, 0, 255, _ignore)
    cv2.createTrackbar(VALUE_SLIDER, WINDOW_NAME, settings.value_min, 255, _ignore)
    shown_color = None
    homography = None
    while True:
        color = COLOR_NAMES[cv2.getTrackbarPos(COLOR_SLIDER, WINDOW_NAME)]
        if color != shown_color:
            # A new colour was chosen: put its saved values on the sliders.
            low, high = bounds_from_hue_ranges(hue_ranges[color])
            cv2.setTrackbarPos(HUE_LOW_SLIDER, WINDOW_NAME, low)
            cv2.setTrackbarPos(HUE_HIGH_SLIDER, WINDOW_NAME, high)
            cv2.setTrackbarPos(SATURATION_SLIDER, WINDOW_NAME, saturation_min[color])
            shown_color = color
        low = cv2.getTrackbarPos(HUE_LOW_SLIDER, WINDOW_NAME)
        high = cv2.getTrackbarPos(HUE_HIGH_SLIDER, WINDOW_NAME)
        hue_ranges[color] = hue_ranges_from_bounds(low, high)
        saturation_min[color] = cv2.getTrackbarPos(SATURATION_SLIDER, WINDOW_NAME)
        value_min = cv2.getTrackbarPos(VALUE_SLIDER, WINDOW_NAME)
        current = replace(settings, hue_ranges=hue_ranges, saturation_min=saturation_min, value_min=value_min)

        frame = read_frame(capture)
        homography = find_homography(frame, settings.board, last_homography=homography)
        if homography is None:
            # Without a field the whole camera image is shown, made smaller so both halves fit on the screen.
            scale = settings.board.warp_px / frame.shape[0]
            image = cv2.resize(frame, (round(frame.shape[1] * scale), settings.board.warp_px))
        else:
            image = straighten_field(frame, homography, settings.board)
        mask = color_mask(image, color, current)
        view = np.hstack([image, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)])
        title = f"{color}: {HELP_TEXT}"
        cv2.putText(view, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow(WINDOW_NAME, view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(SAVE_KEY):
            save_colors(path, hue_ranges, saturation_min, value_min)
            print(f"Colour ranges saved in {path}.")
        if key == ord(BACK_KEY):
            break
    cv2.destroyWindow(WINDOW_NAME)


def hue_ranges_from_bounds(low, high):
    """Turns the two slider values into hue ranges.

    If "from" is larger than "to", the range wraps around the end of the
    scale, as red does: from 170, to 10 gives [(170, 180), (0, 10)].
    """
    if low <= high:
        return [(low, high)]
    return [(low, HUE_MAX), (0, high)]


def bounds_from_hue_ranges(hue_ranges):
    """The way back for the sliders: [(0, 10), (170, 180)] gives from 170, to 10."""
    if len(hue_ranges) == 2:
        upper_part = max(hue_ranges, key=lambda hue_range: hue_range[1])
        lower_part = min(hue_ranges, key=lambda hue_range: hue_range[0])
        return upper_part[0], lower_part[1]
    return hue_ranges[0]


def save_colors(path, hue_ranges, saturation_min, value_min):
    """Writes the colour settings into config.json and keeps all other settings."""
    with open(path, encoding="utf-8") as file:
        raw = json.load(file)
    raw["hue_ranges"] = {name: [list(hue_range) for hue_range in hue_ranges[name]] for name in COLOR_NAMES}
    raw["saturation_min"] = {name: saturation_min[name] for name in COLOR_NAMES}
    raw["value_min"] = value_min
    text = json.dumps(raw, indent=2)
    # Keeps the short form [[0, 10], [170, 180]] instead of one number per line.
    text = re.sub(r"\[\s+(\d+),\s+(\d+)\s+\]", r"[\1, \2]", text)
    text = re.sub(r"\[\s+(\[\d+, \d+\](?:,\s+\[\d+, \d+\])*)\s+\]", _one_line, text)
    with open(path, "w", encoding="utf-8") as file:
        file.write(text + "\n")


def _one_line(match):
    """Joins the inner lists found by the regular expression into one line."""
    return "[" + re.sub(r",\s+", ", ", match.group(1)) + "]"


def _ignore(_value):
    """createTrackbar needs a callback; the loop reads the sliders itself."""
