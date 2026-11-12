"""Tests for reporting changes: new, removed and moved objects."""

from objekterkennung.models import DetectedObject
from objekterkennung.tracker import MOVED, NEW, REMOVED, ChangeTracker, compare_scenes

STABLE_FRAMES = 3
MOVE_THRESHOLD = 0.02
RED_CUBE = DetectedObject("cube", "red", 0.20, 0.20)
BLUE_SPHERE = DetectedObject("sphere", "blue", 0.50, 0.50)


def test_A4_new_object_is_reported_after_stable_frames():
    tracker = ChangeTracker(STABLE_FRAMES, MOVE_THRESHOLD)

    results = [tracker.update([RED_CUBE]) for _ in range(STABLE_FRAMES)]

    assert results[:-1] == [[], []]
    assert [change.kind for change in results[-1]] == [NEW]


def test_A4_change_is_reported_only_once():
    tracker = ChangeTracker(STABLE_FRAMES, MOVE_THRESHOLD)

    results = [tracker.update([RED_CUBE]) for _ in range(3 * STABLE_FRAMES)]

    assert sum(len(changes) for changes in results) == 1


def test_A4_hand_in_the_picture_delays_the_report():
    # A flickering scene (hand moves a block) never stays the same long enough.
    tracker = ChangeTracker(STABLE_FRAMES, MOVE_THRESHOLD)
    frames = [[RED_CUBE], [], [RED_CUBE], []] * 2

    results = [tracker.update(objects) for objects in frames]

    assert all(changes == [] for changes in results)


def test_A4_removed_object_is_reported():
    changes = compare_scenes([RED_CUBE, BLUE_SPHERE], [BLUE_SPHERE], MOVE_THRESHOLD)

    assert [(change.kind, change.item) for change in changes] == [(REMOVED, RED_CUBE)]


def test_A4_moved_object_is_reported_with_old_and_new_position():
    moved_cube = DetectedObject("cube", "red", 0.80, 0.70)

    changes = compare_scenes([RED_CUBE], [moved_cube], MOVE_THRESHOLD)

    assert changes[0].kind == MOVED
    assert changes[0].text() == "MOVED: red cube (0.20 | 0.20) -> (0.80 | 0.70)"


def test_A4_small_jitter_is_not_a_move():
    jittered_cube = DetectedObject("cube", "red", 0.205, 0.195)

    changes = compare_scenes([RED_CUBE], [jittered_cube], MOVE_THRESHOLD)

    assert changes == []


def test_A4_two_equal_cubes_are_matched_by_distance():
    second_cube = DetectedObject("cube", "red", 0.80, 0.80)
    second_moved = DetectedObject("cube", "red", 0.80, 0.60)

    changes = compare_scenes([RED_CUBE, second_cube], [RED_CUBE, second_moved], MOVE_THRESHOLD)

    assert [(change.kind, change.old_item) for change in changes] == [(MOVED, second_cube)]
