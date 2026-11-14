"""Creates the printable A4 board: 18 cm x 18 cm field, four ArUco markers and a 0.1 grid (A3).

Print it at 100 % ("actual size", not "fit to page") and check the field frame with a ruler: 18 cm.
Start from the project folder: python tools/make_board.py
"""

import sys
from pathlib import Path

PROJECT_FOLDER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_FOLDER))

import cv2  # noqa: E402
from PIL import Image  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.units import mm  # noqa: E402
from reportlab.lib.utils import ImageReader  # noqa: E402
from reportlab.pdfgen import canvas  # noqa: E402

from objekterkennung.config import load_settings  # noqa: E402

PAGE_WIDTH_MM, PAGE_HEIGHT_MM = 210.0, 297.0
OUTPUT_PATH = PROJECT_FOLDER / "assets" / "Spielfeld_A4.pdf"
ARUCO_DICTIONARY = cv2.aruco.DICT_4X4_50


def main():
    """Draws field, frame, grid and markers with the sizes from config.json and saves the PDF."""
    board = load_settings().board
    field_left = (PAGE_WIDTH_MM - board.field_size_mm) / 2
    field_top = (PAGE_HEIGHT_MM - board.field_size_mm) / 2
    pdf = canvas.Canvas(str(OUTPUT_PATH), pagesize=A4)

    _draw_markers(pdf, board, field_left, field_top)
    _draw_grid(pdf, board, field_left, field_top)
    _draw_axis_labels(pdf, board, field_left, field_top)

    pdf.showPage()
    pdf.save()
    print(f"Board saved: {OUTPUT_PATH}")


def _y(top_mm):
    """Converts a distance from the top of the page into reportlab's y, which starts at the bottom."""
    return (PAGE_HEIGHT_MM - top_mm) * mm


def _draw_markers(pdf, board, field_left, field_top):
    """Draws marker 0 top-left, 1 top-right, 2 bottom-right and 3 bottom-left of the field."""
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICTIONARY)
    half = board.marker_size_mm / 2
    for marker_id, (center_x, center_y) in board.marker_centers_mm().items():
        marker = cv2.aruco.generateImageMarker(dictionary, marker_id, 600)
        left = field_left + center_x - half
        top = field_top + center_y - half
        image = ImageReader(Image.fromarray(marker))
        pdf.drawImage(image, left * mm, _y(top + board.marker_size_mm), board.marker_size_mm * mm, board.marker_size_mm * mm)


def _draw_grid(pdf, board, field_left, field_top):
    """Light grey lines every 0.1; low saturation, so the detection ignores them."""
    size = board.field_size_mm
    pdf.setStrokeGray(0.75)
    pdf.setLineWidth(0.4)
    for i in range(1, 10):
        offset = i * size / 10
        pdf.line((field_left + offset) * mm, _y(field_top), (field_left + offset) * mm, _y(field_top + size))
        pdf.line(field_left * mm, _y(field_top + offset), (field_left + size) * mm, _y(field_top + offset))
    pdf.setStrokeGray(0.35)
    pdf.setLineWidth(1.0)
    pdf.rect(field_left * mm, _y(field_top + size), size * mm, size * mm)


def _draw_axis_labels(pdf, board, field_left, field_top):
    """Origin (0,0) bottom-left, x to the right, y upwards, (1,1) top-right."""
    size = board.field_size_mm
    bottom = field_top + size
    pdf.setFont("Helvetica", 6)
    pdf.setFillGray(0.45)
    for i in range(11):
        label = f"{i / 10:.1f}"
        offset = i * size / 10
        pdf.drawCentredString((field_left + offset) * mm, _y(bottom + 3.5), label)
        pdf.drawRightString((field_left - 1.5) * mm, _y(bottom - offset + 1), label)
    pdf.setFont("Helvetica-Bold", 8)
    pdf.setFillGray(0.3)
    pdf.drawString((field_left + 1.5) * mm, _y(bottom - 2.5), "(0,0)")
    pdf.drawRightString((field_left + size - 1.5) * mm, _y(field_top + 4.5), "(1,1)")
    pdf.drawString((field_left + size - 12) * mm, _y(bottom - 2.5), "x →")
    pdf.drawString((field_left + 1.5) * mm, _y(field_top + 4.5), "y ↑")


if __name__ == "__main__":
    main()
