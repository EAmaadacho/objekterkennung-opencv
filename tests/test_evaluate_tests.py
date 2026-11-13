"""Tests for the evaluation tool that splits a protocol of many test layouts."""

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from evaluate_tests import layout_numbers, split_at_pauses  # noqa: E402


def test_A4_pause_starts_the_next_layout():
    frames = [(0, 0.0, Counter(), Counter()), (1, 0.1, Counter(), Counter()), (2, 9.5, Counter(), Counter())]

    parts = split_at_pauses(frames)

    assert [len(part) for part in parts] == [2, 1]


def test_A4_skipped_layouts_keep_their_numbers():
    assert layout_numbers(3, 1, [2]) == [1, 3, 4]
