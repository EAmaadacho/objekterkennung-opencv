"""Data types shared by all modules: one detected object with shape, colour and position (A1, A2, A3)."""

from collections import Counter
from dataclasses import dataclass

SHAPE_NAMES = ("cube", "cuboid", "prism", "sphere", "unknown")
# Words for the output. "prism" alone would also fit a cube, so the prism is called triangular.
SHAPE_LABELS = {"cube": "cube", "cuboid": "cuboid", "prism": "triangular prism", "sphere": "sphere", "unknown": "unknown"}
# The 12 blocks of the project as (colour, shape) and how many of each (docs/anforderungen.md).
# The setup mode counts them, and tools/evaluate_tests.py compares every test layout with them.
EXPECTED_OBJECTS = Counter({
    ("red", "cube"): 2, ("blue", "cube"): 2, ("yellow", "cuboid"): 2, ("green", "cuboid"): 1,
    ("blue", "prism"): 2, ("red", "prism"): 1, ("red", "sphere"): 1, ("green", "sphere"): 1,
})


@dataclass(frozen=True)
class DetectedObject:
    """One object on the board, e.g. a red cube at x=0.42, y=0.71.

    x and y are coordinates in the unit square: (0, 0) is the bottom-left
    corner of the field, (1, 1) the top-right corner, and y points up.
    """

    shape: str
    color: str
    x: float
    y: float

    def name(self):
        """Name for the output, e.g. "red cube" or "blue triangular prism"."""
        return f"{self.color} {SHAPE_LABELS[self.shape]}"

    def position(self):
        """Rounded position for the output, e.g. "(0.42 | 0.71)"."""
        return f"({self.x:.2f} | {self.y:.2f})"

    def compact_position(self):
        """Short position for the label on the object, e.g. "(0.42|0.71)"."""
        return f"({self.x:.2f}|{self.y:.2f})"

    def label(self):
        """Name and position, e.g. "red cube (0.42 | 0.71)"."""
        return f"{self.name()} {self.position()}"
