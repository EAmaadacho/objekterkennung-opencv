# Objekterkennung mit OpenCV

Das Programm erkennt mit einer Webcam Form, Farbe und Position von Holzobjekten auf einem 18 cm × 18 cm großen Spielfeld und meldet jede Veränderung. Ein Beispiel: Liegt ein roter Würfel genau in der Mitte des Feldes, zeigt es „red cube (0.50 | 0.50)“; schiebt man ihn nach rechts oben, meldet es „MOVED: red cube (0.50 | 0.50) -> (0.80 | 0.70)“. Wie der Code sind alle Ausgaben und die Oberfläche auf Englisch. Gebaut ist es mit Python und OpenCV, ohne Netzwerk und ohne trainierte Modelle.

Entstanden im Modul PALG (Projekt Algorithmen) an der FHDW, Studiengruppe MFAX424A. Die Anforderungen A1–A7 und die Objektmenge stehen in `docs/anforderungen.md`.

## Wo steht was

Das Programm besteht nur aus `main.py` und dem Ordner `objekterkennung/`. Alles andere sind Tests, Hilfsprogramme und Dokumente, die beim Erkennen nicht mitlaufen.

```
main.py                  Start und Schleife pro Kamerabild
config.json              Kamera, Spielfeldmaße, Farbbereiche (HSV), Lichtbereich
objekterkennung/         das Programm
  config.py              Einstellungen aus config.json lesen
  capture/               1. Bild holen
    camera.py              Webcam öffnen, Bild lesen
    board.py               ArUco-Marker finden, Spielfeld gerade entzerren
    setup_check.py         Einrichtungsmodus: Kamera und Licht prüfen
    calibration.py         Kalibriermodus: Farbbereiche mit Schiebereglern einstellen
  detection/             2. Objekte erkennen
    detector.py            Einstieg: Farbe, Form und Position jedes Objekts
    segmentation.py        Farbmasken (HSV), berührende Objekte trennen
    shapes.py              Form bestimmen: Würfel, Quader, Prisma, Kugel
    ball.py                Kugel aus einer gleichfarbigen Fläche herausschneiden
    prism_pair.py          zwei Prismen, die als Quadrat liegen, trennen
    hand.py                Hand im Feld erkennen
  output/                3. Ergebnis zeigen
    models.py              Datenklasse für ein erkanntes Objekt, erwartete Objektmenge
    smoothing.py           Glättung über mehrere Bilder
    tracker.py             Veränderungen melden (neu, entfernt, verschoben)
    display.py             Fenster, Beschriftungen und Statusleiste
    protocol.py            Protokoll der Beschriftungen für Testläufe (Taste p)
tests/                   Tests mit pytest, eine Datei pro Modul, ohne Kamera
tools/                   Hilfsprogramme, nicht Teil des Programmlaufs
  make_board.py            Spielfeld-PDF aus den Maßen in config.json erzeugen
  camera_check.py          prüfen, was die Webcam liefert
  evaluate_tests.py        Protokoll eines Testlaufs nach Lagen auswerten
assets/Spielfeld_A4.pdf  druckfertiges Spielfeld
docs/anforderungen.md    Anforderungen, Objektmenge, zugehörige Module und Tests
```

## Lesereihenfolge

Die Reihenfolge folgt einem Kamerabild auf seinem Weg durch das Programm.

1. `main.py`: die Schleife, die für jedes Kamerabild läuft. Hier stehen alle Schritte auf einer Seite.
2. `capture/board.py`: die vier Marker finden, das Feld gerade entzerren und Pixel in Koordinaten (x | y) umrechnen.
3. `detection/detector.py`: der Einstieg in die Erkennung. Jede Farbe wird für sich gesucht, danach bekommt jedes Teil eine Form.
4. `detection/segmentation.py`: die Farbmaske in HSV und das Trennen von Teilen, die sich berühren. Das ist der umfangreichste Schritt.
5. `detection/shapes.py`: die Regel, nach der aus einem Umriss Würfel, Quader, Prisma oder Kugel wird.
6. `detection/ball.py`: der Sonderfall einer Kugel, die an einem gleichfarbigen Stein liegt.
7. `output/smoothing.py` und `output/tracker.py`: die Ergebnisse über mehrere Bilder glätten und Veränderungen melden.
8. `output/display.py`: das Fenster mit Kamerabild und beschriftetem Feld.

Der Rest dient dem Aufbau und der Bedienung: `capture/setup_check.py` (Einrichtungsmodus), `capture/calibration.py` (Kalibriermodus), `detection/hand.py` (Hand im Feld) und `output/protocol.py` (Protokoll). `detection/prism_pair.py` trennt zwei gleichfarbige Prismen, die zusammen ein Quadrat bilden; in der jetzigen Objektmenge gibt es jedes Prisma nur einmal, deshalb ist dieser Schritt in `config.json` abgeschaltet.

Jede Datei beginnt mit einem kurzen Text, der sagt, welche Anforderung sie umsetzt und, wo es passt, auf welchem Kapitel des Vorlesungsskripts sie beruht. Die Schwellwerte stehen als benannte Konstanten oben in ihrer Datei, jeweils mit dem Grund für ihren Wert, meist gemessen an Aufnahmen aus den Testläufen. Die Tests tragen die Kennung der Anforderung im Namen, etwa `test_A5_two_touching_cubes_are_split`.

## Aufbau

1. `assets/Spielfeld_A4.pdf` auf weißes A4-Papier drucken, mit 100 % („Tatsächliche Größe“). Den Feldrahmen nachmessen (18 cm) und das Blatt auf Pappe kleben.
2. Die Webcam etwa 45–50 cm senkrecht über der Blattmitte befestigen, alle vier Marker im Bild.
3. Eine feste Lampe seitlich oberhalb, Raum abgedunkelt, kein Fensterlicht. Getestet wurde mit einer EMEET C960 und einer Klemmleuchte auf der dunkelsten Stufe.

Koordinaten: (0,0) unten links, x nach rechts, y nach oben, (1,1) oben rechts, so wie auf dem Spielfeld aufgedruckt.

## Installieren und starten

Vorausgesetzt ist Python 3.10 oder neuer. Unter Windows (PowerShell) im Ordner des Repositorys:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-dev.txt
.venv\Scripts\python main.py
```

Unter macOS oder Linux dasselbe mit `python3` und `.venv/bin/python`.

Nach dem Start kommt der **Einrichtungsmodus**. Er prüft live fünf Punkte und zeigt sie als Ampeln: alle vier Marker zu sehen (Markers), Feld groß genug (Size, mindestens 35 % der Bildhöhe), Kamera gerade von oben (Straight), Feld in der Bildmitte (Centre) und Papier gleichmäßig beleuchtet mit Helligkeit 150–165 (Light). Zu jeder gelben oder roten Ampel steht, was zu tun ist, z. B. „move camera lower“; Fadenkreuz, Wasserwaage und Pfeile helfen beim Ausrichten. Mit `h` werden diese Hilfslinien aus- und eingeblendet. Mit Enter oder START beginnt die Erkennung.

## Bedienung

Über dem Bild liegt eine Statusleiste mit Buttons für Protokoll, Pause und Beenden und einer Licht-Anzeige, die das Licht in jedem Bild weiter prüft. Jeder Button hat auch eine Taste:

| Taste | Wirkung |
|---|---|
| `q` | beenden |
| `u` | Pause zum Umbauen an/aus; danach werden die Veränderungen gemeldet |
| `p` | Protokoll starten/stoppen: jede Beschriftung als CSV, alle 30 Bilder Feld und Fenster, in `recordings/` |
| `c` | Kalibriermodus öffnen |

`main.py --calibrate` startet mit dem Kalibriermodus, `main.py --skip-setup` überspringt den Einrichtungsmodus.

## Einstellungen in config.json

In `config.json` steht alles, was vom Aufbau abhängt. Die Namen im Code sind dieselben: `settings.colors.min_saturation` ist der Wert `colors.min_saturation`.

| Feld | Bedeutung | Wert |
|---|---|---|
| `camera.number`, `width_px`, `height_px` | welche Webcam, in welcher Auflösung | 1, 1920 × 1080 |
| `board.*` | Maße des gedruckten Spielfelds und Größe des geraden Bilds | 180 mm Feld, 600 px |
| `colors.hue_ranges` | Farbtonbereich pro Farbe (Rot hat zwei, weil es an beiden Enden der Skala liegt) | z. B. blau 100–130 |
| `colors.min_saturation`, `min_brightness` | wie kräftig und hell ein Pixel sein muss, um zu einer Farbe zu zählen | z. B. rot 80 / 60 |
| `objects.sphere_diameter_mm` | Kugeldurchmesser, daraus der Radius für `HoughCircles` | 46 |
| `objects.prism_pair_colors` | Farben, von denen zwei Prismen vorkommen | leer |
| `light.paper_brightness_min/max` | Zielbereich der Licht-Ampel | 150–165 |
| `detection.min_object_area_px` | kleinste Fläche, die als Objekt zählt | 3000 px (≈ 16 × 16 mm) |
| `changes.frames_until_reported` | so viele Bilder muss eine Änderung stabil sein | 10 |
| `changes.min_move_share_of_field` | ab dieser Strecke gilt ein Objekt als verschoben | 0,02 (≈ 3,6 mm) |

## Verfahren und Bezug zum Vorlesungsskript

Wo es geht, nutzt das Programm die Verfahren aus dem Skript. Abgewichen wird nur dort, wo das Skript-Verfahren für diese Aufgabe nachweislich nicht ausreicht.

| Schritt | Verfahren | Skript |
|---|---|---|
| Bild holen, anzeigen | `VideoCapture`, `imshow`, `waitKey`, `np.hstack` | Kap. 9.1, 2.3, 3.3.6 |
| Farbe | `cvtColor` nach HSV, `inRange` | Kap. 2.2, 7.2 |
| Maske säubern | Öffnen/Schließen (`morphologyEx`), Löcher füllen mit `findContours`/`drawContours` | Kap. 5.2, 6.1, Übung 12 |
| Naht zwischen Steinen | `GaussianBlur`, `Canny`, `erode`/`dilate` | Kap. 4.1, 4.2, 5.1 |
| Fläche, Form, Position | `findContours`, `contourArea`, `moments`, `boundingRect`, `minEnclosingCircle` | Kap. 6.1 |
| Zwei Prismen als Quadrat | Ausschnitt (ROI), `getRotationMatrix2D`, `warpAffine` | Kap. 3.2, 3.3 |
| Kalibrierung | Schieberegler (`createTrackbar`) | Kap. 4.3.4 |

| Abweichung | Grund |
|---|---|
| ArUco-Marker, `findHomography`, `warpPerspective`, `perspectiveTransform` | Die Kamera schaut leicht schräg auf das Feld; `warpAffine` aus dem Skript kann eine schräge Ansicht nicht gerade rücken. |
| `distanceTransform`, `watershed` | Zwei gleichfarbige Steine, die bündig aneinanderliegen, bilden eine einzige Fläche; Erodieren allein trennt sie nicht. |
| `convexHull`, `minAreaRect`, `boxPoints`, `fillPoly` | Die Ecken-Zählung mit `approxPolyDP` (Übung 11) stimmte an 1 140 Objekten aus echten Aufnahmen nur zu 62 %, weil die Steine Seitenflächen und runde Kanten zeigen. |
| `HoughCircles` | Eine Kugel an einem gleichfarbigen Stein verschmilzt mit ihm; nur ihr runder Umriss trennt sie. |
| `setMouseCallback` | Buttons zusätzlich zu den Tasten, damit man in der Vorführung mit der Maus bedienen kann. |
| Kameraeinstellungen (`CAP_DSHOW`, MJPG, Full HD) | Ohne sie liefert die Webcam nur einen gezoomten Ausschnitt in 640 × 480. |

## Grenzen

- Die Kamera sieht hohe Steine am Feldrand leicht von der Seite. In den Ecken weicht die Position deshalb um bis zu etwa 0,02–0,035 ab, in der Mitte praktisch nicht.
- Gleichfarbige Steine, die bündig aneinanderliegen, werden nur über die sichtbare Naht getrennt. Ist sie sehr schwach, etwa zwischen einem roten Würfel und dem roten Prisma, können beide zu einem Teil verschmelzen.
- Liegt die Kugel auf der Mitte des gleichfarbigen Quaders, kann ihr Schatten den Quader in zwei Stücke teilen.
- Zu helles Licht macht Oberseiten blass, zu dunkles lässt Grün im Schatten verschwinden; deshalb prüft die Licht-Ampel auch während der Erkennung weiter.

## Tests

Die Tests brauchen keine Kamera; sie zeichnen ihre Bilder selbst. Start mit `.venv\Scripts\python -m pytest` (Windows) oder `.venv/bin/python -m pytest`. Bei jedem Push auf `main` laufen sie auch auf GitHub. Commit-Nachrichten sind auf Deutsch und folgen den Conventional Commits, z. B. `feat: Spielfeld über ArUco-Marker entzerren`.
