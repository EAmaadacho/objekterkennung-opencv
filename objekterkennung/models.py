"""Data types shared by all modules: one detected object with shape, colour and position (A1, A2, A3)."""

from dataclasses import dataclass

SHAPE_NAMES = ("cube", "cuboid", "prism", "sphere", "unknown")
SHAPE_LABELS = {
    "cube": "Würfel",
    "cuboid": "Quader",
    "prism": "Prisma",
    "sphere": "Kugel",
    "unknown": "unbekannt",
}
COLOR_LABELS = {"red": "rot", "yellow": "gelb", "green": "grün", "blue": "blau"}


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
        """German name for the output, e.g. "rot Würfel"."""
        return f"{COLOR_LABELS[self.color]} {SHAPE_LABELS[self.shape]}"

    def position(self):
        """Rounded position for the output, e.g. "(0.42 | 0.71)"."""
        return f"({self.x:.2f} | {self.y:.2f})"

    def label(self):
        """Name and position, e.g. "rot Würfel (0.42 | 0.71)"."""
        return f"{self.name()} {self.position()}"
