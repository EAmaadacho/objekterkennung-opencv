# Objekterkennung mit OpenCV

Das Programm erkennt mit einer Webcam Form, Farbe und Position von Holzobjekten auf einem 18 cm × 18 cm großen Spielfeld und meldet jede Veränderung. Ein Beispiel: Liegt ein roter Würfel genau in der Mitte des Feldes, zeigt es „red cube (0.50 | 0.50)“; schiebt man ihn nach rechts oben, meldet es „MOVED: red cube (0.50 | 0.50) -> (0.80 | 0.70)“. Wie der Code sind alle Ausgaben und die Oberfläche auf Englisch. Gebaut ist es mit Python und OpenCV und baut auf den Verfahren des Vorlesungsskripts auf; wo sie nicht ausreichen, kommen weitere OpenCV-Funktionen dazu. Es läuft komplett offline, ohne Netzwerk und ohne KI oder trainierte Modelle.

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

Jede Datei beginnt mit einem kurzen Text, der sagt, welche Anforderung sie umsetzt und auf welchem Kapitel des Vorlesungsskripts sie beruht. Die Schwellwerte stehen als benannte Konstanten oben in ihrer Datei. Die Tests tragen die Kennung der Anforderung im Namen, etwa `test_A5_two_touching_cubes_are_split`.

## Aufbau

1. `assets/Spielfeld_A4.pdf` auf weißes A4-Papier drucken, mit 100 % („Tatsächliche Größe“). Den Feldrahmen nachmessen (18 cm) und das Blatt auf Pappe kleben.
2. Die Webcam etwa 45–50 cm senkrecht über der Blattmitte befestigen, alle vier Marker im Bild.
3. Eine feste Lampe seitlich oberhalb, Raum abgedunkelt, kein Fensterlicht.

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

## Einstellungen

In `config.json` steht alles, was vom Aufbau abhängt: Kamera, Maße des Spielfelds, Farbbereiche in HSV und der Zielbereich für das Licht. Die Farbbereiche lassen sich im Kalibriermodus (`c`) mit Schiebereglern einstellen und speichern.

## Tests

Die Tests brauchen keine Kamera; sie zeichnen ihre Bilder selbst. Start mit `.venv\Scripts\python -m pytest` (Windows) oder `.venv/bin/python -m pytest`. Bei jedem Push auf `main` laufen sie auch auf GitHub.
