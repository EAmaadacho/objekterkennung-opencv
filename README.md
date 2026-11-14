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

## Lesereihenfolge

Die Reihenfolge folgt einem Kamerabild auf seinem Weg durch das Programm. Die Schritte 1 bis 7 sind die eigentliche Erkennung, Schritt 8 zeigt das Ergebnis an.

1. `main.py`: die Schleife, die für jedes Kamerabild läuft. Hier stehen alle Schritte auf einer Seite.
2. `capture/board.py`: die vier Marker finden, das Feld gerade entzerren und Pixel in Koordinaten (x | y) umrechnen.
3. `detection/detector.py`: der Einstieg in die Erkennung. Jede Farbe wird für sich gesucht, danach wird jedem Teil eine Form zugeordnet.
4. `detection/segmentation.py`: die Farbmaske in HSV und das Trennen von Teilen, die sich berühren. Das ist der umfangreichste Schritt.
5. `detection/shapes.py`: die Regel, nach der aus einem Umriss Würfel, Quader, Prisma oder Kugel wird.
6. `detection/ball.py` und `detection/prism_pair.py`: zwei Sonderfälle beim Trennen, nämlich eine Kugel an einem gleichfarbigen Quader und zwei Prismen, die zusammen wie ein Würfel aussehen.
7. `output/smoothing.py` und `output/tracker.py`: die Ergebnisse über mehrere Bilder glätten und Veränderungen melden.
8. `output/display.py`: das Fenster mit Kamerabild und beschriftetem Feld.

Der Rest dient dem Aufbau und der Bedienung: `capture/setup_check.py` (Einrichtungsmodus), `capture/calibration.py` (Kalibriermodus), `detection/hand.py` (Hand im Feld) und `output/protocol.py` (Protokoll für die Testläufe).

Jede Datei beginnt mit einem kurzen Text, der sagt, welche Anforderung sie umsetzt (A1–A7, siehe [docs/anforderungen.md](docs/anforderungen.md)) und, wo es passt, auf welchem Kapitel des Vorlesungsskripts sie beruht. Die Schwellwerte der Verfahren stehen als benannte Konstanten oben in ihrer Datei, jeweils mit dem Grund für ihren Wert. Die Tests tragen die Kennung der Anforderung im Namen, etwa `test_A5_two_touching_cubes_are_split` in `tests/test_segmentation.py`.

## Aufbau vor dem Start

1. `assets/Spielfeld_A4.pdf` auf weißes A4-Papier drucken, mit **100 % / „Tatsächliche Größe“**. Den Feldrahmen mit dem Lineal nachmessen (18 cm) und das Blatt auf Pappe kleben.
2. Eine USB-Webcam etwa 45–50 cm senkrecht über der Mitte des Blatts befestigen. Alle vier Marker müssen im Bild sein.
3. Eine feste Lampe einschalten; Fensterlicht vermeiden.

So wurde das Programm getestet, und so lässt sich der Aufbau wiederholen:

- Webcam EMEET C960 am Kamera-Arm, etwa 45–50 cm über der Blattmitte, Objektiv parallel zum Blatt.
- Klemmleuchte auf Stufe 4, seitlich oberhalb; das Papier hat dann im Einrichtungsmodus Helligkeit 155–170 (Licht-Ampel grün).
- Alle fünf Ampeln im Einrichtungsmodus grün, dann mit Enter starten.
- Bei anderem Licht oder anderer Kamera einmal `main.py --calibrate` ausführen und die Farbbereiche neu einstellen.

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

Nach dem Start kommt zuerst der **Einrichtungsmodus**. Er prüft live fünf Punkte und zeigt sie als Ampeln:

- **Markers:** Sind alle 4 Marker zu sehen?
- **Size:** Füllt das Feld genug vom Bild (Ziel: mindestens 35 % der Bildhöhe)?
- **Straight:** Schaut die Kamera gerade von oben? Dann sind gegenüberliegende Feldseiten gleich lang und das Feld ist nicht verdreht.
- **Centre:** Liegt das Feld in der Bildmitte?
- **Light:** Liegt das Papier im Zielbereich 155–170 (Helligkeit 0–255; in diesem Licht wurden die Farbbereiche eingestellt), ist es gleichmäßig ausgeleuchtet und ohne überbelichtete Stellen?

Zu jeder gelben oder roten Ampel steht, was zu tun ist, z. B. „move camera lower“. Beim Ausrichten helfen ein helles Fadenkreuz durch die Bildmitte und ein Zielrahmen. Durch die Mitte des erkannten Felds läuft ein zweites Kreuz in der Farbe des Feldrahmens; liegt es auf dem hellen Fadenkreuz, hängt die Kamera genau über der Feldmitte. Eine Wasserwaage in der Bildmitte zeigt die Neigung der Kamera. Ist sie zu schräg, zeigt ein Pfeil vom Punkt zur Mitte, wohin die Linse gekippt werden muss. Liegt das Feld nicht in der Mitte, zeigt ein zweiter Pfeil, wohin die Kamera verschoben werden muss.

Beide Pfeile werden in jedem Bild neu aus den Markern berechnet. Damit „links“ und „oben“ dieselbe Richtung meinen wie am Tisch, wenn man vor der unteren Feldkante steht, wird das Kamerabild automatisch so gedreht, dass (0,0) unten links liegt, egal wie Kamera oder Blatt stehen. Gedreht wird in Vierteldrehungen (`np.rot90`), weil dabei kein Pixel neu berechnet wird; eine Restdrehung bis 45° meldet die Straight-Ampel („turn paper“). Auch im Hauptfenster ist das Kamerabild so gedreht. Ob die Pfeile richtig herum zeigen, prüfen Tests mit einer nachgerechneten Kamera (Lochkameramodell).

Mit Enter oder dem Button START beginnt die Erkennung; `main.py --skip-setup` überspringt den Modus.

Was „grün“ bedeutet, ist an den Testaufnahmen vom 12. und 13.11. geprüft:

| Ampel | Grün, wenn | Beleg aus den Aufnahmen |
|---|---|---|
| Markers | alle 4 Marker gefunden | Ohne sie kann das Feld nicht entzerrt werden. |
| Size | Feld ≥ 35 % der Bildhöhe | Der Testaufbau erreichte 36–38 %; damit wurden bei grünem Licht 97 % der Teile richtig erkannt. |
| Straight | Neigung ≤ 5 %, Drehung ≤ 2° | Am 13.11. immer erfüllt (etwa 1 %); 5 % verschieben die Oberseite eines Würfels nur um etwa 4 mm. |
| Centre | Feldmitte ≤ 5 % neben der Bildmitte | Am 12.11. lag das Feld zu weit unten, ein blaues Prisma am Rand erschien abgeschnitten. |
| Light | Papier 155–170, Gleichmäßigkeit ≥ 65 % | Gleiche Lage, gleiche Software: bei Papier 171 wurden 0 von 16 roten Würfeln erkannt, eine Lampenstufe dunkler (155–161) 18 von 18. Über alle Bilder vom 13.11.: bei grünem Licht 97 % der Teile richtig, sonst 90 %. |

Mit `main.py --calibrate` startet der Kalibriermodus: Mit Schiebereglern werden die Farbbereiche an das Licht angepasst und in `config.json` gespeichert.

## Einstellungen in config.json

Hier steht alles, was sich mit dem Aufbau ändert: Kamera, Blatt, Farben und Licht. Die Schwellen der einzelnen Verfahren (z. B. für Canny) stehen dagegen als benannte Konstanten mit Begründung im Code, weil sie nicht vom Aufbau abhängen. Die Farbwerte stellt man am einfachsten mit dem Kalibriermodus ein (`main.py --calibrate`), er speichert sie hier.

Die Namen im Code sind dieselben wie hier: `settings.colors.min_saturation` ist im Code genau der Wert `colors.min_saturation` aus config.json. So sieht man beim Lesen sofort, woher ein Wert kommt.

| Feld | Bedeutung | Einheit, Wert | Wann ändern? |
|---|---|---|---|
| `camera.number` | Welche Kamera benutzt wird: 0 ist meist die eingebaute, 1 die erste USB-Webcam. | Zahl, 1 | Wenn „No camera found“ erscheint oder das falsche Bild kommt. |
| `camera.width_px`, `camera.height_px` | Bildgröße, die von der Webcam angefordert wird. | Pixel, 1920 × 1080 | Nur bei einer anderen Webcam. |
| `board.field_size_mm` | Seitenlänge des Spielfelds. | mm, 180 | Nur bei einem anders gedruckten Spielfeld. |
| `board.margin_around_field_mm` | Wie viel Papier rund um das Feld im geraden Bild mit gezeigt wird. Dadurch wird ein hoher Stein am Rand noch ganz gesehen, auch wenn die Kamera etwas schräg auf ihn schaut. Erkannt wird nur, was mit seinem Mittelpunkt im Feld liegt. | mm, 15 | Kleiner, wenn am linken oder rechten Bildrand Tisch zu sehen ist: Links und rechts endet das Papier 15 mm neben dem Feld, größer geht dort nicht. |
| `board.marker_size_mm` | Kantenlänge der vier ArUco-Marker. | mm, 30 | Nur bei einem anders gedruckten Spielfeld. |
| `board.marker_distance_to_field_mm` | Abstand der Marker von der Feldlinie. | mm, 8 | Nur bei einem anders gedruckten Spielfeld. |
| `board.field_size_in_image_px` | Wie groß das Feld im geraden Bild ist. Mehr Pixel zeigen feinere Kanten, kosten aber Rechenzeit. | Pixel, 600 (≈ 3,3 Pixel pro mm) | Selten; dann auch `detection.min_object_area_px` anpassen. |
| `colors.hue_ranges` | Farbtonbereich pro Farbe. Rot liegt an beiden Enden der Skala und hat deshalb zwei Bereiche. | 0–180 (OpenCV), z. B. blau 100–130 | Wenn eine Farbe mit einer anderen verwechselt wird; am besten im Kalibriermodus. |
| `colors.min_saturation` | Mindest-Sättigung pro Farbe: wie kräftig eine Farbe sein muss, um zu zählen. Weißes Papier und graue Schatten liegen darunter. | 0–255, z. B. rot 100 | Höher, wenn Schatten oder Haut als Farbe erkannt werden; niedriger, wenn Steine bei starkem Licht blass werden und fehlen. |
| `colors.min_brightness` | Mindest-Helligkeit pro Farbe: dunklere Pixel zählen nicht zu dieser Farbe. Grün liegt niedriger, weil der dunkelgrüne Quader nur Helligkeit 27–54 hat. | 0–255, z. B. grün 45, sonst 60 | Niedriger, wenn dunkle Steine oder ihre Seitenflächen fehlen. |
| `objects.sphere_diameter_mm` | Durchmesser der Kugeln. Daraus folgt der Radius, nach dem `HoughCircles` sucht. | mm, 46 | Bei anderen Kugeln. |
| `objects.prism_pair_colors` | Farben, von denen zwei Prismen im Feld liegen. Nur bei ihnen wird geprüft, ob ein Quadrat eigentlich zwei Prismen ist. | Liste, `["blue"]` | Wenn sich die Objektmenge ändert. |
| `light.paper_brightness_min`, `light.paper_brightness_max` | Zielbereich für die Helligkeit des weißen Papiers im Einrichtungsmodus. In diesem Licht (Lampe Stufe 4) wurden die Farbbereiche eingestellt. | 0–255, 155–170 | Nach einer neuen Kalibrierung bei anderem Licht. |
| `detection.min_object_area_px` | Kleinste Fläche, die als Objekt zählt; alles Kleinere ist Rauschen. | Pixel im geraden Bild, 3000 (≈ 16 × 16 mm) | Höher, wenn kleine Flecken als Objekt erscheinen. |
| `changes.frames_until_reported` | Wie viele Bilder hintereinander eine Änderung gleich bleiben muss, bevor sie gemeldet wird. Über genauso viele Bilder wird auch die angezeigte Position geglättet (Median). | Bilder, 10 (≈ 1–1,5 s) | Höher, wenn Meldungen flackern; niedriger, wenn sie zu spät kommen. |
| `changes.min_move_share_of_field` | Ab welcher Strecke ein Objekt als verschoben gemeldet wird. | Anteil der Feldbreite, 0.02 (≈ 3,6 mm) | Höher, wenn ruhig liegende Steine als verschoben gemeldet werden. |

**Genauigkeit der Position:** Die Kamera sieht hohe Steine am Feldrand leicht von der Seite, ihre Oberseite erscheint dadurch etwas weiter außen. In den Ecken weicht die Position deshalb um bis zu etwa 0,02–0,035 ab, in der Mitte praktisch nicht. Das Programm korrigiert das nicht, weil die Höhe eines Steins aus einem Bild von oben nicht bekannt ist.

**Würfel und Quader:** Beide zählen als eigene Form. Ein Würfel ist von oben ein Quadrat (Seiten 1 : 1), ein Quader ein Rechteck mit Seiten etwa 2,4 : 1. Die Grenze liegt bei 1,5 : 1 (siehe `detection/shapes.py`). Zwei bündig aneinanderliegende gleichfarbige Würfel ergeben von oben ebenfalls ein Rechteck 2 : 1; sie werden nur über die sichtbare Naht zwischen ihnen getrennt (Canny, Watershed).

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
| Bild holen, anzeigen | `VideoCapture`, `imshow`, `waitKey`, `np.hstack` | Kap. 9.1, 2.3, 3.3.6 |
| Farbe | `cvtColor` nach HSV, `inRange` | Kap. 2.2, 7.2 (Farbsegmentierung) |
| Maske säubern | Öffnen/Schließen (`morphologyEx`), Löcher füllen mit `findContours`/`drawContours` | Kap. 5.2, Übung 12 (7.2.2), Kap. 6.1 |
| Kante zwischen Klötzen | `GaussianBlur`, `Canny` (bei großen Flächen ein zweiter, feinerer Durchlauf für schwache Nähte), `erode`/`dilate` | Kap. 4.1, 4.2, 5.1 |
| Flächen, Form, Position | `findContours`, `contourArea`, `moments`, `boundingRect`, `minEnclosingCircle` | Kap. 6.1, 6.1.7–6.1.9 |
| Zwei Prismen als Quadrat | Ausschnitt (ROI), `getRotationMatrix2D`, `warpAffine`, Helligkeitsprofil | Kap. 3.2, 3.3 |
| Kalibrierung | Schieberegler (`createTrackbar`) | Kap. 4.3.4 |

Abweichungen und ihr Grund:

| Verfahren | Warum nicht mit Skript-Mitteln |
|---|---|
| ArUco-Marker, `findHomography`, `warpPerspective`, `perspectiveTransform` | Die Kamera schaut leicht schräg auf das Feld. Das Skript kennt nur `warpAffine` (Drehen, Verschieben), das eine schräge Ansicht nicht gerade rücken kann. Der Einrichtungsmodus rechnet mit derselben Homographie die Feldecken zurück ins Kamerabild (`perspectiveTransform`). |
| `distanceTransform`, `watershed` | Zwei gleichfarbige Würfel, die bündig aneinander liegen, bilden eine einzige Fläche. Erodieren allein trennt sie nicht. |
| `convexHull`, `minAreaRect` in der Formregel, `minAreaRect` auch beim Trennen, beim Prismenpaar und für den Streifen, den die Kugel verdeckt | Die Ecken-Zählung aus Übung 11 (`approxPolyDP`) stimmte auf 1 140 Objekten aus echten Aufnahmen nur zu 62 %, weil die Klötze Seitenflächen und runde Kanten zeigen. |
| `HoughCircles` | Eine Kugel, die an einem gleichfarbigen Quader liegt, verschmilzt mit ihm. Nur ihr runder Umriss trennt sie, denn Würfel mit Seitenflächen sind im Bild genauso dick wie die Kugel. |
| `setMouseCallback` für die Buttons in der Statusleiste (nur Komfort) | Das Skript kennt als Bedienung nur Tasten (`waitKey`) und Schieberegler. Die Buttons sind zusätzlich zu den Tasten da, damit man während der Vorführung mit der Maus pausieren und das Protokoll starten kann; jede Funktion bleibt auch über ihre Taste erreichbar. |
| Kameraeinstellungen (`CAP_DSHOW`, MJPG, Full HD) | Ohne sie liefert die Webcam nur einen gezoomten Ausschnitt in 640 × 480. |

## Arbeitsweise

Commit-Nachrichten folgen den [Conventional Commits](https://www.conventionalcommits.org/de/), auf Deutsch, zum Beispiel `feat: Spielfeld über ArUco-Marker entzerren`. Gearbeitet wird in kleinen Commits direkt auf `main`; die CI prüft jeden Push. Code, Kommentare und Namen sind auf Englisch (Vorgabe der Aufgabenstellung), Dokumentation und Commits auf Deutsch.
