"""Smooths the detections over the last frames, so a single unclear frame changes nothing (A1, A4)."""

import math
import statistics
from collections import Counter, deque
from dataclasses import replace

# 0.05 is 9 mm on the 180 mm field: a block that jitters by a few pixels stays the same object.
SAME_OBJECT_DISTANCE = 0.05
# An object must be seen in more than half of the remembered frames. "More than" matters:
# right after a block was moved, its old and new place are each seen in exactly half.
MIN_SHARE = 0.5


class FrameSmoother:
    """Remembers the objects of the last frames and returns the objects seen in most of them.

    Each object gets the shape that was found most often, e.g. "prism" in 8 of
    10 frames and "cube" in 2 gives "prism", and the median of its positions. The median
    ignores single outliers, so the shown position and label stay calm when the camera
    shakes slightly or the border of a block flickers by a pixel.
    """

    def __init__(self, frames):
        self._history = deque(maxlen=frames)

    def update(self, objects):
        """Takes the objects of one frame and returns the smoothed objects."""
        self._history.append(list(objects))
        groups = []
        # The newest frame comes first, so every group starts with the newest position.
        for frame_objects in reversed(self._history):
            for item in frame_objects:
                group = _nearest_group(groups, item)
                if group is None:
                    groups.append([item])
                else:
                    group.append(item)
        needed = MIN_SHARE * len(self._history)
        result = []
        for group in groups:
            if len(group) > needed:
                shape = Counter(item.shape for item in group).most_common(1)[0][0]
                x = statistics.median(item.x for item in group)
                y = statistics.median(item.y for item in group)
                result.append(replace(group[0], shape=shape, x=x, y=y))
        return result


def _nearest_group(groups, item):
    """The group of the same colour whose newest object is closest, or None if none is near enough."""
    best, best_distance = None, SAME_OBJECT_DISTANCE
    for group in groups:
        first = group[0]
        distance = math.hypot(first.x - item.x, first.y - item.y)
        if first.color == item.color and distance <= best_distance:
            best, best_distance = group, distance
    return best
