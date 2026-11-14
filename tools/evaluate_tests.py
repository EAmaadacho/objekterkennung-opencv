"""Splits a protocol of many test layouts into one part per layout and checks each against the 12 blocks.

Not part of the program: it only reads a finished protocol folder.
Build every layout during a pause (key u); nothing is recorded while paused, so the
time column jumps there, and each jump starts the next layout L1, L2, ...

Start from the project folder: python tools/evaluate_tests.py recordings/<date>_protocol
Options: --first 11 numbers the first part L11, --skip 4 7 leaves out layouts that were not built.
"""

import argparse
import csv
import shutil
import sys
from collections import Counter
from pathlib import Path

PROJECT_FOLDER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_FOLDER))

import cv2  # noqa: E402

from objekterkennung.capture.setup_check import check_light  # noqa: E402
from objekterkennung.config import load_settings  # noqa: E402

# The 12 blocks that lie in the field in every test layout.
EXPECTED = Counter({
    ("red", "cube"): 2, ("blue", "cube"): 2, ("yellow", "cuboid"): 2, ("green", "cuboid"): 1,
    ("blue", "prism"): 2, ("red", "prism"): 1, ("red", "sphere"): 1, ("green", "sphere"): 1,
})
LIGHT = load_settings().light
# A pause of the program shows up as a jump of more than this many seconds.
PAUSE_GAP_S = 2.0
# The first part of every layout is left out, because the smoothing still settles there.
SETTLE_SHARE = 0.25


def main():
    """Reads the protocol, cuts it at the pauses and writes one result line per layout."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("protocol", type=Path, help="protocol folder with labels.csv")
    parser.add_argument("--first", type=int, default=1, help="number of the first layout")
    parser.add_argument("--skip", type=int, nargs="*", default=[], help="layout numbers that were not built")
    arguments = parser.parse_args()
    frames = read_frames(arguments.protocol / "labels.csv")
    parts = split_at_pauses(frames)
    numbers = layout_numbers(len(parts), arguments.first, arguments.skip)
    output = arguments.protocol / "evaluation"
    output.mkdir(exist_ok=True)
    rows = [evaluate(number, part, arguments.protocol, output) for number, part in zip(numbers, parts)]
    write_results(rows, output)


def read_frames(path):
    """Returns a list of frames in time order: (frame, time_s, raw Counter, shown Counter)."""
    frames = {}
    with open(path, encoding="utf-8") as file:
        for row in csv.DictReader(file, delimiter=";"):
            frame = int(row["frame"])
            entry = frames.setdefault(frame, [frame, float(row["time_s"]), Counter(), Counter()])
            entry[2 if row["kind"] == "raw" else 3][(row["color"], row["shape"])] += 1
    return [tuple(entry) for _, entry in sorted(frames.items())]


def split_at_pauses(frames):
    """Cuts the frames wherever the time jumps by more than PAUSE_GAP_S."""
    parts = []
    for frame in frames:
        if not parts or frame[1] - parts[-1][-1][1] > PAUSE_GAP_S:
            parts.append([])
        parts[-1].append(frame)
    return parts


def layout_numbers(count, first, skip):
    """The layout number of each part, e.g. 1, 2, 3, 5 when layout 4 was skipped."""
    numbers = []
    number = first
    while len(numbers) < count:
        if number not in skip:
            numbers.append(number)
        number += 1
    return numbers


def evaluate(number, part, protocol, output):
    """Counts how well one layout was detected and copies its images into evaluation/L<number>."""
    settled = part[int(len(part) * SETTLE_SHARE):]
    shown_correct = [sum((shown & EXPECTED).values()) for _, _, _, shown in settled]
    raw_correct = [sum((raw & EXPECTED).values()) for _, _, raw, _ in settled]
    result, _ = Counter(frozenset(shown.items()) for _, _, _, shown in settled).most_common(1)[0]
    result = Counter(dict(result))
    changes = sum(1 for before, after in zip(settled, settled[1:]) if before[3] != after[3])
    folder = output / f"L{number:02d}"
    folder.mkdir(exist_ok=True)
    brightness = []
    for frame, _, _, _ in part:
        for suffix in ("_field.png", "_window.jpg"):
            image = protocol / f"{frame:05d}{suffix}"
            if image.exists():
                shutil.copy(image, folder / image.name)
                if suffix == "_field.png":
                    brightness.append(check_light(cv2.imread(str(image)), LIGHT).text.split(",")[0])
    return {
        "layout": f"L{number}",
        "seconds": round(part[-1][1] - part[0][1], 1),
        "result": f"{sum((result & EXPECTED).values())}/12",
        "missing": _names(EXPECTED - result),
        "extra": _names(result - EXPECTED),
        "shown_correct_mean": round(sum(shown_correct) / max(len(shown_correct), 1), 2),
        "raw_correct_mean": round(sum(raw_correct) / max(len(raw_correct), 1), 2),
        "label_changes": changes,
        "light": brightness[len(brightness) // 2] if brightness else "",
    }


def write_results(rows, output):
    """Writes results.csv and prints a short table."""
    with open(output / "results.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), delimiter=";")
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(f"{row['layout']:>4}  {row['result']:>5}  changes {row['label_changes']:>3}  {row['light']:<10}  "
              f"missing: {row['missing'] or '-'}  extra: {row['extra'] or '-'}")
    full = sum(1 for row in rows if row["result"] == "12/12")
    print(f"{full} of {len(rows)} layouts fully correct. Details: {output / 'results.csv'}")


def _names(counter):
    """Readable list of blocks, e.g. "2x red cube, 1x blue prism"."""
    return ", ".join(f"{count}x {color} {shape}" for (color, shape), count in sorted(counter.items()))


if __name__ == "__main__":
    main()
