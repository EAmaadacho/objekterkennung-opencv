"""Records every label the program shows, to count later how often it was right and how often it changed (A1, A2, A4)."""

import csv
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2

PROTOCOL_FOLDER = Path("recordings")
# Every 30th frame (about every 3-4 seconds at 7-10 frames per second) the images are saved as well.
IMAGE_EVERY_FRAMES = 30
RAW = "raw"
SHOWN = "shown"


class LabelProtocol:
    """Writes one CSV row per object and frame, both for the raw and for the shown (smoothed) result.

    "raw" is what one single frame gives, "shown" is what the window shows
    after the smoothing over several frames. Comparing both shows how much the
    smoothing helps.
    """

    def __init__(self, folder=PROTOCOL_FOLDER, now=None):
        self.folder = Path(folder) / f"{(now or datetime.now()):%Y-%m-%d_%H-%M-%S}_protocol"
        self.folder.mkdir(parents=True, exist_ok=True)
        self._file = open(self.folder / "labels.csv", "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file, delimiter=";")
        self._writer.writerow(["frame", "time_s", "kind", "color", "shape", "x", "y"])
        self._start = time.monotonic()
        self._frames = 0
        self._changes = {RAW: 0, SHOWN: 0}
        self._last = {RAW: None, SHOWN: None}
        self._shown_counts = Counter()

    def add(self, raw_objects, shown_objects, field_image=None, window_image=None):
        """Adds one frame; saves the clean field and the labelled window every IMAGE_EVERY_FRAMES frames."""
        seconds = round(time.monotonic() - self._start, 2)
        for kind, objects in ((RAW, raw_objects), (SHOWN, shown_objects)):
            for item in objects:
                self._writer.writerow([self._frames, seconds, kind, item.color, item.shape, f"{item.x:.3f}", f"{item.y:.3f}"])
            names = sorted(item.name() for item in objects)
            if self._last[kind] is not None and names != self._last[kind]:
                self._changes[kind] += 1
            self._last[kind] = names
        self._shown_counts.update(item.name() for item in shown_objects)
        if self._frames % IMAGE_EVERY_FRAMES == 0:
            if field_image is not None:
                cv2.imwrite(str(self.folder / f"{self._frames:05d}_field.png"), field_image)
            if window_image is not None:
                cv2.imwrite(str(self.folder / f"{self._frames:05d}_window.jpg"), window_image)
        self._frames += 1

    def close(self):
        """Closes the CSV file, writes summary.txt and returns its text."""
        self._file.close()
        lines = [
            f"Frames: {self._frames}",
            f"Label changes, raw: {self._changes[RAW]}",
            f"Label changes, shown: {self._changes[SHOWN]}",
            "Shown labels (in how many frames):",
        ]
        lines += [f"  {name}: {count}" for name, count in sorted(self._shown_counts.items())]
        text = "\n".join(lines)
        (self.folder / "summary.txt").write_text(text + "\n", encoding="utf-8")
        return text
