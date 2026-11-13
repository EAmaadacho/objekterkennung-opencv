# Objekterkennung mit OpenCV

Das Programm erkennt mit einer Webcam Form, Farbe und Position von Holzobjekten auf einem 18 cm × 18 cm großen Spielfeld und meldet jede Veränderung. Ein Beispiel: Liegt ein roter Würfel genau in der Mitte des Feldes, zeigt es „red cube (0.50 | 0.50)“; schiebt man ihn nach rechts oben, meldet es „MOVED: red cube (0.50 | 0.50) -> (0.80 | 0.70)“. Wie der Code sind alle Ausgaben und die Oberfläche auf Englisch. Gebaut ist es mit Python und OpenCV, ohne Netzwerk und ohne KI-Modelle zur Erkennung.

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

Nach dem Start kommt zuerst der **Einrichtungsmodus**, am besten mit leerem Feld. Er prüft live fünf Punkte und zeigt sie als Ampeln: Sind alle 4 Marker zu sehen? Füllt das Feld genug vom Bild (Ziel: mindestens die Hälfte der Bildhöhe)? Schaut die Kamera gerade von oben (gegenüberliegende Feldseiten gleich lang, Feld nicht verdreht)? Liegt das Feld in der Bildmitte? Ist das Papier hell genug, gleichmäßig ausgeleuchtet und ohne überbelichtete Stellen? Zu jeder gelben oder roten Ampel steht, was zu tun ist, z. B. „move camera lower“. Ein Fadenkreuz und ein Zielrahmen helfen beim Ausrichten. Mit Enter oder dem Button START beginnt die Erkennung; `main.py --skip-setup` überspringt den Modus.

Mit `main.py --calibrate` startet der Kalibriermodus: Mit Schiebereglern werden die Farbbereiche an das Licht angepasst und in `config.json` gespeichert.

## Bedienung

Über dem Bild liegt eine Statusleiste mit drei Buttons für die Maus: Protokoll, Pause und Beenden. Läuft das Protokoll, ist sein Button rot („REC“). Während einer Pause ist der Pause-Button orange und das ganze Fenster orange umrandet. Daneben steht in Worten, was das Programm gerade tut („LIVE“ oder „PAUSED“). Jeder Button hat auch eine Taste:

| Taste | Wirkung |
|---|---|
| `q` | beenden |
| `p` | Protokoll starten/stoppen: jede Beschriftung als CSV, alle 30 Bilder Feld und Fenster, Zusammenfassung in `recordings/` |
| `u` | Pause zum Umbauen an/aus: keine Erkennung, keine Meldungen, kein Protokoll; danach werden die Veränderungen gemeldet |
| `c` | Kalibriermodus öffnen |

## Tests

Die Tests brauchen keine Kamera; sie arbeiten mit Bildern, die sie selbst zeichnen. Unter Windows `.venv\Scripts\python -m pytest`, unter macOS oder Linux `.venv/bin/python -m pytest`. Bei jedem Push auf `main` laufen sie auch auf GitHub.

## Verfahren und Bezug zum Vorlesungsskript

Wo es geht, nutzt das Programm die Verfahren aus dem Vorlesungsskript. Abgewichen wird nur dort, wo das Skript-Verfahren für diese Aufgabe nachweislich nicht ausreicht.

| Schritt | Verfahren | Skript |
|---|---|---|
| Bild holen, anzeigen | `VideoCapture`, `imshow`, `waitKey`, `np.hstack` | Kap. 1/2 |
| Farbe | `cvtColor` nach HSV, `inRange` | Kap. 2.2 |
| Maske säubern | Öffnen/Schließen (`morphologyEx`), Löcher füllen mit `findContours`/`drawContours` | Kap. 5.2, 6.1 |
| Kante zwischen Klötzen | `GaussianBlur`, `Canny`, `erode`/`dilate` | Kap. 4.1, 4.2, 5.1 |
| Flächen, Form, Position | `findContours`, `contourArea`, `minEnclosingCircle`, `moments`, `boundingRect` | Kap. 6.1 |
| Zwei Prismen als Quadrat | Ausschnitt (ROI), `getRotationMatrix2D`, `warpAffine`, Helligkeitsprofil | Kap. 3.2, 3.3 |
| Kalibrierung | Schieberegler (`createTrackbar`) | Kap. 4.3.4 |

Abweichungen und ihr Grund:

| Verfahren | Warum nicht mit Skript-Mitteln |
|---|---|
| ArUco-Marker, `findHomography`, `warpPerspective`, `perspectiveTransform` | Die Kamera schaut leicht schräg auf das Feld. Das Skript kennt nur `warpAffine` (Drehen, Verschieben), das eine schräge Ansicht nicht gerade rücken kann. Der Einrichtungsmodus rechnet mit derselben Homographie die Feldecken zurück ins Kamerabild (`perspectiveTransform`). |
| `distanceTransform`, `watershed` | Zwei gleichfarbige Würfel, die bündig aneinander liegen, bilden eine einzige Fläche. Erodieren allein trennt sie nicht. |
| `convexHull`, `minAreaRect` in der Formregel | Die Ecken-Zählung aus Übung 11 (`approxPolyDP`) stimmte auf 1 140 Objekten aus echten Aufnahmen nur zu 62 %, weil die Klötze Seitenflächen und runde Kanten zeigen. |
| `HoughCircles` | Eine Kugel, die an einem gleichfarbigen Quader liegt, verschmilzt mit ihm. Nur ihr runder Umriss trennt sie, denn Würfel mit Seitenflächen sind im Bild genauso dick wie die Kugel. |
| `setMouseCallback` für die Buttons in der Statusleiste | Das Skript kennt als Bedienung nur Tasten (`waitKey`) und Schieberegler. Die Buttons sind zusätzlich zu den Tasten da, damit man während der Vorführung mit der Maus pausieren und das Protokoll starten kann; jede Funktion bleibt auch über ihre Taste erreichbar. |
| Kameraeinstellungen (`CAP_DSHOW`, MJPG, Full HD) | Ohne sie liefert die Webcam nur einen gezoomten Ausschnitt in 640 × 480. |

## Aufbau des Repositorys

| Ordner / Datei | Inhalt |
|---|---|
| `main.py` | Startpunkt: liest die Einstellungen, öffnet die Kamera und führt die Schleife pro Bild aus. |
| `config.json` | Kamera, Spielfeldmaße, Farbbereiche (HSV) und Schwellwerte. |
| [`objekterkennung/`](objekterkennung) | Der Code, ein Modul pro Aufgabe: `config` (Einstellungen), `camera` (Kamera), `board` (Marker finden, Feld entzerren), `segmentation` (Farbmasken, Trennen berührender Objekte), `ball` (Kugel aus berührenden Objekten herausschneiden), `shapes` (Form), `prism_pair` (zwei Prismen als Quadrat trennen), `detector` (Farbe, Form und Position aller Objekte), `smoothing` (Glättung über mehrere Bilder), `tracker` (Veränderungen), `hand` (Hand im Feld erkennen, Meldungen pausieren), `models` (Datenklassen), `display` (Anzeige), `setup_check` (Einrichtungsmodus für Kamera und Licht), `calibration` (Kalibriermodus), `protocol` (Protokoll der Beschriftungen). |
| [`tests/`](tests) | Automatische Tests mit pytest. |
| [`tools/`](tools) | `make_board.py` erzeugt das Spielfeld-PDF aus den Maßen in `config.json`. |
| [`assets/`](assets) | Das druckfertige Spielfeld. |
| [`docs/`](docs) | [Anforderungen](docs/anforderungen.md) mit Kennungen A1–A7. |
| [`.github/`](.github) | CI-Pipeline, die bei jedem Push alle Tests ausführt. |

## Arbeitsweise

Commit-Nachrichten folgen den [Conventional Commits](https://www.conventionalcommits.org/de/), auf Deutsch, zum Beispiel `feat: Spielfeld über ArUco-Marker entzerren`. Gearbeitet wird in kleinen Commits direkt auf `main`; die CI prüft jeden Push. Code, Kommentare und Namen sind auf Englisch (Vorgabe der Aufgabenstellung), Dokumentation und Commits auf Deutsch.
