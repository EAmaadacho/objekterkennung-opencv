"""Tests for saving real test images; the field is drawn with numpy, no camera is needed."""

import json
from datetime import datetime

import cv2
import numpy as np

from objekterkennung.models import DetectedObject
from objekterkennung.recording import save_recording

FIELD_PX = 600
RED_CUBE = DetectedObject("cube", "red", 0.4219, 0.71)
TIME = datetime(2026, 11, 10, 19, 45, 2)


def test_A3_recording_saves_clean_field_and_objects(tmp_path):
    field = np.full((FIELD_PX, FIELD_PX, 3), 255, dtype=np.uint8)

    image_path = save_recording(field, [RED_CUBE], folder=tmp_path, now=TIME)

    assert image_path.name == "2026-11-10_19-45-02_feld.png"
    assert (cv2.imread(str(image_path)) == field).all()
    rows = json.loads((tmp_path / "2026-11-10_19-45-02_objekte.json").read_text(encoding="utf-8"))
    assert rows == [{"shape": "cube", "color": "red", "x": 0.422, "y": 0.71}]
