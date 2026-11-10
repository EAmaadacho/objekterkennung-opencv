# Objekterkennung mit OpenCV

Das Programm erkennt mit einer Webcam Form, Farbe und Position von Holzobjekten auf einem 18 cm × 18 cm großen Spielfeld und meldet jede Veränderung. Ein Beispiel: Liegt ein roter Würfel genau in der Mitte des Feldes, zeigt es „rot Würfel (0.50 | 0.50)“; schiebt man ihn nach rechts oben, meldet es „VERSCHOBEN: rot Würfel (0.50 | 0.50) -> (0.80 | 0.70)“. Gebaut ist es mit Python und OpenCV, ohne Netzwerk und ohne KI-Modelle zur Erkennung.

Entstanden im Modul PALG (Projekt Algorithmen) an der FHDW, Studiengruppe MFAX424A.

## Aufbau vor dem Start

1. `assets/Spielfeld_A4.pdf` auf weißes A4-Papier drucken, mit **100 % / „Tatsächliche Größe“**. Den Feldrahmen mit dem Lineal nachmessen (18 cm) und das Blatt auf Pappe kleben.
2. Eine USB-Webcam etwa 45–50 cm senkrecht über der Mitte des Blatts befestigen. Alle vier Marker müssen im Bild sein.
3. Eine feste Lampe einschalten; Fensterlicht vermeiden.

Das Koordinatensystem: (0,0) unten links, x nach rechts, y nach oben, (1,1) oben rechts – so wie auf dem Spielfeld aufgedruckt.

## Installieren und starten

Vorausgesetzt ist Python 3.10 oder neuer (`python --version`).

Mit Windows PowerShell im Ordner des Repositorys:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-dev.txt
.venv\Scripts\python main.py
```

Unter macOS oder Linux:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python main.py
```

Mit `main.py --calibrate` startet der Kalibriermodus: Mit Schiebereglern werden die Farbbereiche an das Licht angepasst und in `config.json` gespeichert.

## Bedienung

| Taste | Wirkung |
|---|---|
| `q` | beenden |
| `space` | aktuelle Objektliste in der Konsole ausgeben |
| `s` | Bildschirmfoto in `screenshots/` speichern |
| `c` | Kalibriermodus öffnen |

## Tests

Die Tests brauchen keine Kamera; sie arbeiten mit Bildern, die sie selbst zeichnen. Unter Windows `.venv\Scripts\python -m pytest`, unter macOS oder Linux `.venv/bin/python -m pytest`. Bei jedem Push auf `main` laufen sie auch auf GitHub.

## Aufbau des Repositorys

| Ordner / Datei | Inhalt |
|---|---|
| `main.py` | Startpunkt: liest die Einstellungen, öffnet die Kamera und führt die Schleife pro Bild aus. |
| `config.json` | Kamera, Spielfeldmaße, Farbbereiche (HSV) und Schwellwerte. |
| [`objekterkennung/`](objekterkennung) | Der Code, ein Modul pro Aufgabe: `config` (Einstellungen), `camera` (Kamera), `board` (Marker finden, Feld entzerren), `segmentation` (Farbmasken, Trennen berührender Objekte), `shapes` (Form), `detector` (Farbe, Form und Position aller Objekte), `tracker` (Veränderungen), `models` (Datenklassen), `display` (Anzeige), `calibration` (Kalibriermodus). |
| [`tests/`](tests) | Automatische Tests mit pytest. |
| [`tools/`](tools) | `make_board.py` erzeugt das Spielfeld-PDF aus den Maßen in `config.json`. |
| [`assets/`](assets) | Das druckfertige Spielfeld. |
| [`docs/`](docs) | [Anforderungen](docs/anforderungen.md) mit Kennungen A1–A7. |
| [`.github/`](.github) | CI-Pipeline, die bei jedem Push alle Tests ausführt. |

## Arbeitsweise

Commit-Nachrichten folgen den [Conventional Commits](https://www.conventionalcommits.org/de/), auf Deutsch, zum Beispiel `feat: Spielfeld über ArUco-Marker entzerren`. Gearbeitet wird in kleinen Commits direkt auf `main`; die CI prüft jeden Push. Code, Kommentare und Namen sind auf Englisch (Vorgabe der Aufgabenstellung), Dokumentation und Commits auf Deutsch.
