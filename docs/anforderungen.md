# Anforderungen

Die Anforderungen stammen aus der Aufgabenstellung „PALG: Anwendung OpenCV“ (Version 1.2). Jede hat eine Kennung; Tests, die eine Anforderung prüfen, beginnen mit dieser Kennung, etwa `test_A3_board_fits_on_a4`.

| Kennung | Anforderung | Umsetzung (Modul) | Tests |
|---|---|---|---|
| A1 | Die Form jedes Objekts wird erkannt: Würfel, Quader, Dreiecksprisma, Kugel. | `shapes.py`, `prism_pair.py`, `detector.py` | `test_shapes.py`, `test_prism_pair.py`, `test_detector.py` |
| A2 | Die Farbe jedes Objekts wird erkannt: rot, gelb, grün, blau. | `segmentation.py`, `detector.py` | `test_segmentation.py`, `test_detector.py` |
| A3 | Die Position wird im Einheitsquadrat (0,0)–(1,1) ausgegeben und ist mit dem aufgedruckten Raster überprüfbar. | `board.py`, `tools/make_board.py` | `test_board.py`, `test_config.py` |
| A4 | Veränderungen (neu, entfernt, verschoben) werden automatisch erkannt und gemeldet. | `smoothing.py`, `tracker.py`, `hand.py` | `test_smoothing.py`, `test_tracker.py`, `test_hand.py` |
| A5 | Objekte, die sich berühren, werden getrennt erkannt. | `segmentation.py` | `test_segmentation.py`, `test_detector.py` |
| A6 | Live-Bild einer Kamera; keine Netzwerkkommunikation zur Laufzeit; keine LLMs zur Erkennung. | `camera.py`, `display.py`, `main.py` | `test_camera.py`, `test_display.py` |
| A7 | Kamera, Farbbereiche und Schwellwerte sind ohne Codeänderung einstellbar (config.json, Kalibriermodus). | `config.py`, `calibration.py` | `test_config.py`, `test_calibration.py` |

## Objektmenge

| Anzahl | Objekt |
|---|---|
| 2 | rote Würfel |
| 2 | blaue Würfel |
| 2 | gelbe Quader |
| 1 | grüner Quader |
| 2 | blaue Dreiecksprismen |
| 1 | rotes Dreiecksprisma |
| 1 | gelbe Kugel |
| 1 | grüne Kugel |

12 Objekte: 4 Formen, 4 Farben, jede Form in mindestens 2 Farben, 4 Objekte doppelt, alle einfarbig und dreidimensional. Die Dreiecksprismen liegen auf ihrer Dreiecksfläche, damit sie von oben als Dreieck zu sehen sind.
