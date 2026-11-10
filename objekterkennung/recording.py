"""Saves real test images: the straight field without labels plus the detected objects (A1, A2, A3)."""

import json
from datetime import datetime
from pathlib import Path

import cv2

RECORDING_FOLDER = Path("aufnahmen")


def save_recording(field_image, objects, folder=RECORDING_FOLDER, now=None):
    """Writes e.g. aufnahmen/2026-11-10_19-45-02_feld.png and ..._objekte.json; returns the image path.

    The image is the clean straight field, so it can be checked again later with
    other colour ranges or rules. The JSON file holds what the program found,
    and a person can correct it into the expected result for a test.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"{(now or datetime.now()):%Y-%m-%d_%H-%M-%S}"
    image_path = folder / f"{stem}_feld.png"
    cv2.imwrite(str(image_path), field_image)
    rows = [{"shape": item.shape, "color": item.color, "x": round(item.x, 3), "y": round(item.y, 3)} for item in objects]
    text = json.dumps(rows, indent=2, ensure_ascii=False)
    (folder / f"{stem}_objekte.json").write_text(text + "\n", encoding="utf-8")
    return image_path
