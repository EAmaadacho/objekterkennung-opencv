"""Tests for smoothing the detections over several frames."""

from objekterkennung.output.models import DetectedObject
from objekterkennung.output.smoothing import FrameSmoother

FRAMES = 10
RED_PRISM = DetectedObject("prism", "red", 0.50, 0.50)
RED_CUBE_SAME_PLACE = DetectedObject("cube", "red", 0.51, 0.50)
BLUE_CUBE = DetectedObject("cube", "blue", 0.20, 0.80)


def test_A1_shape_found_most_often_wins():
    smoother = FrameSmoother(FRAMES)
    frames = [[RED_PRISM]] * 7 + [[RED_CUBE_SAME_PLACE]] * 3

    for objects in frames:
        result = smoother.update(objects)

    assert [item.shape for item in result] == ["prism"]


def test_A4_object_missing_in_one_frame_is_kept():
    smoother = FrameSmoother(FRAMES)
    frames = [[RED_PRISM, BLUE_CUBE]] * 9 + [[RED_PRISM]]

    for objects in frames:
        result = smoother.update(objects)

    assert BLUE_CUBE in result


def test_A4_removed_object_disappears_after_half_the_frames():
    smoother = FrameSmoother(FRAMES)
    frames = [[RED_PRISM, BLUE_CUBE]] * FRAMES + [[RED_PRISM]] * 6

    for objects in frames:
        result = smoother.update(objects)

    assert result == [RED_PRISM]


def test_A4_single_jittering_frame_does_not_move_the_position():
    smoother = FrameSmoother(FRAMES)
    jitter = DetectedObject("prism", "red", 0.52, 0.51)
    frames = [[RED_PRISM]] * 9 + [[jitter]]

    for objects in frames:
        result = smoother.update(objects)

    assert result == [RED_PRISM]


def test_A4_moved_object_shows_its_new_place_after_half_the_frames():
    smoother = FrameSmoother(FRAMES)
    moved = DetectedObject("prism", "red", 0.80, 0.20)
    frames = [[RED_PRISM]] * FRAMES + [[moved]] * 6

    for objects in frames:
        result = smoother.update(objects)

    assert result == [moved]
