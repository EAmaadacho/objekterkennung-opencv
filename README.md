# Objekterkennung mit OpenCV

Das Programm erkennt mit einer Webcam Form, Farbe und Position von Holzobjekten auf einem 18 cm × 18 cm großen Spielfeld und meldet jede Veränderung. Ein Beispiel: Liegt ein roter Würfel genau in der Mitte des Feldes, zeigt es „red cube (0.50 | 0.50)“; schiebt man ihn nach rechts oben, meldet es „MOVED: red cube (0.50 | 0.50) -> (0.80 | 0.70)“. Wie der Code sind alle Ausgaben und die Oberfläche auf Englisch. Gebaut ist es mit Python und OpenCV, ohne Netzwerk und ohne KI-Modelle zur Erkennung.

Entstanden im Modul PALG (Projekt Algorithmen) an der FHDW, Studiengruppe MFAX424A.

## Wo steht was

Das Programm besteht nur aus `main.py` und dem Ordner `objekterkennung/`. Alles andere sind Tests, Hilfsprogramme und Dokumente, die beim Erkennen nicht mitlaufen.

```
main.py                  Start und Schleife pro Kamerabild
config.json              Kamera, Spielfeldmaße, Farbbereiche (HSV), Schwellwerte
objekterkennung/         DAS PROGRAMM
  config.py              Einstellungen aus config.json lesen
  capture/               1. Bild holen
    camera.py              Webcam öffnen, Bild lesen
    board.py               ArUco-Marker finden, Spielfeld gerade entzerren
    setup_check.py         Einrichtungsmodus: Kamera und Licht prüfen
    calibration.py         Kalibriermodus: Farbbereiche mit Schiebereglern einstellen
  detection/             2. Objekte erkennen
    segmentation.py        Farbmasken (HSV), berührende Objekte trennen
    ball.py                Kugel aus berührenden Objekten herausschneiden
    prism_pair.py          zwei Prismen, die als Quadrat liegen, trennen
    shapes.py              Form bestimmen: Würfel, Quader, Prisma, Kugel
    hand.py                Hand im Feld erkennen
    detector.py            alles zusammen: Farbe, Form, Position jedes Objekts
  output/                3. Ergebnis zeigen
    models.py              Datenklasse für ein erkanntes Objekt
    smoothing.py           Glättung über mehrere Bilder
    tracker.py             Veränderungen melden (neu, entfernt, verschoben)
    display.py             Beschriftungen und Statusleiste im Fenster
    protocol.py            Protokoll der Beschriftungen (Taste p)
tests/                   automatische Tests mit pytest, eine Datei pro Modul
tools/                   Hilfsprogramme, nicht Teil des Programmlaufs
  make_board.py            Spielfeld-PDF aus den Maßen in config.json erzeugen
  camera_check.py          prüfen, was die Webcam liefert
  evaluate_tests.py        Protokoll mit vielen Testlagen an den Pausen aufteilen und
                           jede Lage mit den 12 erwarteten Teilen vergleichen
assets/                  druckfertiges Spielfeld
docs/                    Anforderungen A1–A7 mit Verweis auf Code und Tests
.github/                 CI, führt bei jedem Push alle Tests aus
```

**Lesereihenfolge:** `main.py` zeigt den Ablauf pro Bild. Danach in der Reihenfolge, in der ein Bild durchs Programm läuft: `capture/board.py` → `detection/detector.py` (ruft `segmentation`, `ball`, `prism_pair` und `shapes` auf) → `output/smoothing.py` → `output/tracker.py` → `output/display.py`.

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

Nach dem Start kommt zuerst der **Einrichtungsmodus**. Er prüft live sechs Punkte und zeigt sie als Ampeln: Sind alle 4 Marker zu sehen? Füllt das Feld genug vom Bild (Ziel: mindestens die Hälfte der Bildhöhe)? Schaut die Kamera gerade von oben (gegenüberliegende Feldseiten gleich lang, Feld nicht verdreht)? Liegt das Feld in der Bildmitte? Ist das Papier im Zielbereich 155–170 (Helligkeit 0–255; in diesem Licht wurden die Farbbereiche eingestellt), gleichmäßig ausgeleuchtet und ohne überbelichtete Stellen? Glänzen die Steine (dafür ein paar Steine ins Feld legen)? Zu jeder gelben oder roten Ampel steht, was zu tun ist, z. B. „move camera lower“. Ein Fadenkreuz und ein Zielrahmen helfen beim Ausrichten, eine Wasserwaage in der Bildmitte zeigt die Neigung der Kamera: Der Punkt wandert in die Richtung, in die die Kamera schaut, und muss in den grünen Ring. Mit Enter oder dem Button START beginnt die Erkennung; `main.py --skip-setup` überspringt den Modus.

Mit `main.py --calibrate` startet der Kalibriermodus: Mit Schiebereglern werden die Farbbereiche an das Licht angepasst und in `config.json` gespeichert.

## Einstellungen in config.json

Hier steht alles, was sich mit dem Aufbau ändert: Kamera, Blatt, Farben und Licht. Die Schwellen der einzelnen Verfahren (z. B. für Canny) stehen dagegen als benannte Konstanten mit Begründung im Code, weil sie nicht vom Aufbau abhängen. Die Farbwerte stellt man am einfachsten mit dem Kalibriermodus ein (`main.py --calibrate`), er speichert sie hier.

| Feld | Bedeutung | Einheit, Wert | Wann ändern? |
|---|---|---|---|
| `camera_index` | Welche Kamera benutzt wird: 0 ist meist die eingebaute, 1 die erste USB-Webcam. | Zahl, 1 | Wenn „No camera found“ erscheint oder das falsche Bild kommt. |
| `board.field_mm` | Seitenlänge des Spielfelds. | mm, 180 | Nur bei einem anders gedruckten Spielfeld. |
| `board.margin_mm` | Wie viel Papier rund um das Feld im geraden Bild mit gezeigt wird. Dadurch wird ein hoher Stein am Rand noch ganz gesehen, auch wenn die Kamera etwas schräg auf ihn schaut. Erkannt wird nur, was mit seinem Mittelpunkt im Feld liegt. | mm, 15 | Größer, wenn Steine am Rand abgeschnitten werden; höchstens so groß wie Markerabstand + Markergröße. |
| `board.marker_mm` | Kantenlänge der vier ArUco-Marker. | mm, 30 | Nur bei einem anders gedruckten Spielfeld. |
| `board.marker_gap_mm` | Abstand der Marker von der Feldlinie. | mm, 8 | Nur bei einem anders gedruckten Spielfeld. |
| `board.warp_px` | Wie groß das Feld im geraden Bild ist. Mehr Pixel zeigen feinere Kanten, kosten aber Rechenzeit. | Pixel, 600 (≈ 3,3 Pixel pro mm) | Selten; dann auch `min_area_px` anpassen. |
| `saturation_min` | Mindest-Sättigung pro Farbe: wie kräftig eine Farbe sein muss, um zu zählen. Weißes Papier und graue Schatten liegen darunter. | 0–255, z. B. rot 100 | Höher, wenn Schatten oder Haut als Farbe erkannt werden; niedriger, wenn Steine bei starkem Licht blass werden und fehlen. |
| `value_min` | Mindest-Helligkeit: dunklere Pixel zählen zu keiner Farbe. | 0–255, 60 | Niedriger, wenn dunkle Steine oder ihre Seitenflächen fehlen. |
| `hue_ranges` | Farbtonbereich pro Farbe. Rot liegt an beiden Enden der Skala und hat deshalb zwei Bereiche. | 0–180 (OpenCV), z. B. blau 100–130 | Wenn eine Farbe mit einer anderen verwechselt wird; am besten im Kalibriermodus. |
| `min_area_px` | Kleinste Fläche, die als Objekt zählt; alles Kleinere ist Rauschen. | Pixel im geraden Bild, 3000 (≈ 16 × 16 mm) | Höher, wenn kleine Flecken als Objekt erscheinen. |
| `stable_frames` | Wie viele Bilder hintereinander eine Änderung gleich bleiben muss, bevor sie gemeldet wird. | Bilder, 10 (≈ 1 s) | Höher, wenn Meldungen flackern; niedriger, wenn sie zu spät kommen. |
| `move_threshold` | Ab welcher Strecke ein Objekt als verschoben gemeldet wird. | Anteil der Feldbreite, 0.02 (≈ 3,6 mm) | Höher, wenn ruhig liegende Steine als verschoben gemeldet werden. |

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
| Kante zwischen Klötzen | `GaussianBlur`, `Canny` (bei großen Flächen ein zweiter, feinerer Durchlauf für schwache Nähte), `erode`/`dilate` | Kap. 4.1, 4.2, 5.1 |
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

## Arbeitsweise

Commit-Nachrichten folgen den [Conventional Commits](https://www.conventionalcommits.org/de/), auf Deutsch, zum Beispiel `feat: Spielfeld über ArUco-Marker entzerren`. Gearbeitet wird in kleinen Commits direkt auf `main`; die CI prüft jeden Push. Code, Kommentare und Namen sind auf Englisch (Vorgabe der Aufgabenstellung), Dokumentation und Commits auf Deutsch.
