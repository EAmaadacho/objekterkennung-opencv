"""Tests for the shared data types."""

import pytest

from objekterkennung.models import SHAPE_NAMES, DetectedObject


def test_A3_label_shows_rounded_coordinates():
    detected = DetectedObject(shape="cube", color="red", x=0.4249, y=0.7051)

    text = detected.label()

    assert text == "red cube (0.42 | 0.71)"


def test_A1_all_four_object_shapes_are_known():
    shapes = {"cube", "cuboid", "prism", "sphere"}

    known = set(SHAPE_NAMES)

    assert shapes <= known


def test_A3_detected_object_cannot_be_changed():
    # Frozen objects make sure no module moves an object by accident.
    detected = DetectedObject(shape="prism", color="blue", x=0.1, y=0.9)

    with pytest.raises(AttributeError):
        detected.x = 0.2
