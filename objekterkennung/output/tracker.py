"""Compares the scene over time and reports new, removed and moved objects (A4)."""

import math
from dataclasses import dataclass

NEW = "NEW"
REMOVED = "REMOVED"
MOVED = "MOVED"


@dataclass(frozen=True)
class Change:
    """One change of the scene, e.g. a red cube that was moved from (0.20 | 0.20) to (0.80 | 0.70)."""

    kind: str
    item: object
    old_item: object = None

    def text(self):
        """Line for the console, e.g. "MOVED: red cube (0.20 | 0.20) -> (0.80 | 0.70)"."""
        if self.kind == MOVED:
            return f"{MOVED}: {self.item.name()} {self.old_item.position()} -> {self.item.position()}"
        return f"{self.kind}: {self.item.label()}"


class ChangeTracker:
    """Reports a change only when the new scene has stayed the same for frames_until_reported images.

    This waits until a hand has left the field: while the lecturer places a
    block, the scene changes in every frame, so nothing is reported yet.
    """

    def __init__(self, frames_until_reported, min_move_share_of_field):
        self._frames_until_reported = frames_until_reported
        self._min_move_share_of_field = min_move_share_of_field
        self._reported = []
        self._candidate = []
        self._candidate_frames = 0

    def update(self, objects):
        """Takes the objects of one frame and returns the changes, usually an empty list."""
        if _same_scene(objects, self._candidate, self._min_move_share_of_field):
            self._candidate_frames += 1
        else:
            self._candidate = objects
            self._candidate_frames = 1
        if self._candidate_frames != self._frames_until_reported:
            return []
        changes = compare_scenes(self._reported, objects, self._min_move_share_of_field)
        if changes:
            self._reported = objects
        return changes


def compare_scenes(old_objects, new_objects, min_move_share_of_field):
    """Lists what changed from the old to the new scene: NEW, REMOVED or MOVED.

    An object only matches an object of the same shape and colour. The two
    nearest of all possible pairs are matched first.
    """
    pairs, removed, added = _match(old_objects, new_objects)
    changes = [Change(REMOVED, item) for item in removed]
    changes += [Change(NEW, item) for item in added]
    for old, new in pairs:
        if _distance(old, new) > min_move_share_of_field:
            changes.append(Change(MOVED, new, old))
    return changes


def _same_scene(first, second, min_move_share_of_field):
    """True if both lists hold the same objects, each one moved less than min_move_share_of_field."""
    pairs, removed, added = _match(first, second)
    return not removed and not added and all(_distance(a, b) <= min_move_share_of_field for a, b in pairs)


def _match(old_objects, new_objects):
    """Pairs old and new objects of the same kind; returns pairs, unmatched old, unmatched new."""
    candidates = [
        (_distance(old, new), old_index, new_index)
        for old_index, old in enumerate(old_objects)
        for new_index, new in enumerate(new_objects)
        if (old.shape, old.color) == (new.shape, new.color)
    ]
    used_old, used_new, pairs = set(), set(), []
    for _, old_index, new_index in sorted(candidates):
        if old_index in used_old or new_index in used_new:
            continue
        used_old.add(old_index)
        used_new.add(new_index)
        pairs.append((old_objects[old_index], new_objects[new_index]))
    removed = [item for index, item in enumerate(old_objects) if index not in used_old]
    added = [item for index, item in enumerate(new_objects) if index not in used_new]
    return pairs, removed, added


def _distance(first, second):
    """Distance of two objects in the unit square, e.g. 0.1 is 18 mm on the field."""
    return math.hypot(first.x - second.x, first.y - second.y)
