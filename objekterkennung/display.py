"""Builds the one window that shows the camera image and the straight field side by side (A6)."""

import cv2
import numpy as np

NO_FIELD_TEXT = "Spielfeld nicht gefunden"
TEXT_COLOR = (0, 0, 255)
MARKER_COLOR = (0, 255, 0)


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
    """Puts both images next to each other, e.g. a 600 px high camera image and the 600 x 600 field.

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
