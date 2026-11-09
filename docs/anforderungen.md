# Anforderungen

Die Anforderungen stammen aus der Aufgabenstellung „PALG: Anwendung OpenCV“ (Version 1.2). Jede hat eine Kennung; Tests, die eine Anforderung prüfen, beginnen mit dieser Kennung, etwa `test_A3_board_fits_on_a4`.

| Kennung | Anforderung | Umsetzung (Modul) | Tests |
|---|---|---|---|
| A1 | Die Form jedes Objekts wird erkannt: Würfel, Quader, Dreiecksprisma, Kugel. | `shapes.py` | `test_shapes.py` |
| A2 | Die Farbe jedes Objekts wird erkannt: rot, gelb, grün, blau. | `colors.py`, `segmentation.py` | `test_colors.py` |
| A3 | Die Position wird im Einheitsquadrat (0,0)–(1,1) ausgegeben und ist mit dem aufgedruckten Raster überprüfbar. | `board.py`, `tools/make_board.py` | `test_board.py`, `test_config.py` |
| A4 | Veränderungen (neu, entfernt, verschoben) werden automatisch erkannt und gemeldet. | `tracker.py` | `test_tracker.py` |
| A5 | Objekte, die sich berühren, werden getrennt erkannt. | `segmentation.py` | `test_segmentation.py` |
| A6 | Live-Bild einer Kamera; keine Netzwerkkommunikation zur Laufzeit; keine LLMs zur Erkennung. | `camera.py`, `main.py` | – (Vorführung) |
| A7 | Kamera, Farbbereiche und Schwellwerte sind ohne Codeänderung einstellbar (config.json, Kalibriermodus). | `config.py`, `calibration.py` | `test_config.py` |

## Objektmenge

| Anzahl | Objekt |
|---|---|
| 2 | rote Würfel |
| 2 | blaue Würfel |
| 2 | gelbe Quader |
| 1 | grüner Quader |
| 2 | grüne Dreiecksprismen |
| 1 | rotes Dreiecksprisma |
| 2 | blaue Kugeln |
| 1 | gelbe Kugel |

4 Formen, 4 Farben, jede Form in 2 Farben, 5 Objekte doppelt, alle einfarbig und dreidimensional.
