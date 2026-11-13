"""Tests for the label protocol; objects are given directly, no camera is needed."""

import csv
from datetime import datetime

import numpy as np

from objekterkennung.output.models import DetectedObject
from objekterkennung.output.protocol import LabelProtocol

RED_CUBE = DetectedObject("cube", "red", 0.40, 0.70)
RED_CUBOID = DetectedObject("cuboid", "red", 0.40, 0.70)
TIME = datetime(2026, 11, 10, 20, 0, 0)


def test_A4_protocol_counts_label_changes(tmp_path):
    protocol = LabelProtocol(folder=tmp_path, now=TIME)
    raw_frames = [[RED_CUBE], [RED_CUBOID], [RED_CUBE]]

    for raw in raw_frames:
        protocol.add(raw, [RED_CUBE])
    summary = protocol.close()

    assert "Label changes, raw: 2" in summary
    assert "Label changes, shown: 0" in summary
    assert "red cube: 3" in summary


def test_A4_protocol_writes_one_row_per_object_and_saves_images(tmp_path):
    protocol = LabelProtocol(folder=tmp_path, now=TIME)
    field = np.zeros((60, 60, 3), dtype=np.uint8)

    protocol.add([RED_CUBE], [RED_CUBE], field_image=field, window_image=field)
    protocol.close()

    folder = tmp_path / "2026-11-10_20-00-00_protocol"
    with open(folder / "labels.csv", encoding="utf-8") as file:
        rows = list(csv.reader(file, delimiter=";"))
    assert rows[1:] == [["0", rows[1][1], "raw", "red", "cube", "0.400", "0.700"], ["0", rows[2][1], "shown", "red", "cube", "0.400", "0.700"]]
    assert (folder / "00000_field.png").exists()
    assert (folder / "00000_window.jpg").exists()
