# Fenster und Linienlage: Das Fenster tauscht Auflösung gegen Leckage, das Gitter ist nur ein Raster – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-fenster-linienlage-demo.streamlit.app/)**

Zweite Demo der Reihe **Spektral- und Wavelet-Analyse** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“, aufbauend auf der
Demo zum **Periodogramm**. Ein simuliertes Signal aus zwei Linien (bekannte Lage, bekannte Stärke) in weißem oder steil abfallendem Hintergrund wird mit sechs Fenstern analysiert;
Auflösung, Leckage, Gipfelverlust und die Schätzung der Linienlage werden gegen die Wahrheit gemessen.

**Namensfalle:** Mit „Spectral Clustering“ (`spectral-demo`, Eigenwerte eines Graphen) hat diese Demo nichts zu tun.

## Warum dieses Problem

Das Periodogramm der Vorgängerdemo hat zwei Mängel: Die Täler liegen bei hohem Dynamikumfang weit über der wahren Dichte (Leckage des Rechteckfensters), und eine Linie zwischen zwei Bins
erscheint zu niedrig und an der falschen Stelle. Das Fenster senkt die Leckage (je nach Fenster im Tal bis unter 0,1 dB) und kostet Auflösung; Auffüllen mit Nullen ändert die Fensterkennzahlen nicht und liefert nicht mehr Information,
interpoliert aber das Spektrum: Gipfelhöhe und -lage lassen sich besser ablesen, und zwei Linien nahe der Grenze erscheinen ab Faktor 2 bis 4 öfter getrennt (darüber hinaus ändert sich nichts); die Lage der Linie
lässt sich aus drei Bins auf einen Bruchteil eines Bins schätzen. Die relative Streuung je Bin (Std/Mittel = 1 bei Rauschen) senkt kein Fenster; im Leckage-Tal des Rechtecks ist sie sogar größer (Thema der Folgedemos).

## Modell

Zwei Linien $A_1\cos(2\pi f_1 m/f_s + \varphi_1) + A_2\cos(2\pi f_2 m/f_s + \varphi_2)$ plus Rauschen (weiß oder AR-Hintergrund „Steil“ mit 56,7 dB Spannweite, Varianz auf 1 normiert). Fenster
sind periodisch (DFT-gerade). Das gefensterte Periodogramm hat einen exakt berechenbaren Erwartungswert über die Fensterautokorrelation (Formel im 📐-Expander der App). Die Abtastrate ist fest 1000 Hz.

## Methodik

Sechs Fenster (Rechteck, Hann, Hamming, Blackman, Kaiser mit β-Regler, Flat-Top), zwei Normierungen (Leistungsdichte für Rauschen, Linienhöhe für Gipfel), sechs Schätzer der Linienlage (Spitze,
Parabel, Quinn 1, Jacobsen, Rife, Candan) mit der Cramér-Rao-Schranke und dem feinen Maximum als Referenz. Experimente über feste Zufallsströme, Auflösung als Anteil der Läufe über Phasen und Lagen.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stehen in `tests/test_claims.py` (Anteile bei 100 bis 150 Läufen: Standardfehler 4 bis 5 Prozentpunkte; nicht alle Zahlen sind Anteile).

| Frage | Befund |
|---|---|
| Kennzahlen der Fenster (n = 256): Rauschbandbreite / Hauptkeule bei −3 dB / höchste Nebenkeule / Gipfelverlust bei ½ Bin | Rechteck 1,00 / 0,89 Bins / −13,3 dB / −3,92 dB; Hann 1,50 / 1,44 / −31,5 / −1,42; Hamming 1,36 / 1,30 / −42,7 / −1,75; Blackman 1,73 / 1,64 / −58,1 / −1,10; Kaiser (β = 8) 1,67 / 1,59 / −58,7 / −1,18; Flat-Top 3,77 / 3,73 / −92,7 / −0,01. |
| Was kostet die Normierung auf Linienhöhe statt Leistungsdichte? | Der fensterabhängige Unterschied der beiden Normierungen ist die Rauschbandbreite in dB (gegenüber dem Rechteck): Hann 1,76 dB, Blackman 2,37 dB, Flat-Top 5,76 dB. |
| Wie hoch liegt der Mittelwert im Tal (Hintergrund „Steil“, bei 0,49 der Abtastrate)? | Rechteck um 17,9 / 11,9 / 6,7 dB über der wahren Dichte bei n = 128 / 512 / 2048; Hamming 3,0 / 0,9 / 0,2 dB; Hann, Blackman, Kaiser, Flat-Top unter 0,1 dB. |
| Wie dicht dürfen zwei gleich starke Linien liegen (aufgelöst = zwei Gipfel mit mindestens 10 % Prominenz)? | Bei 1,0 Bin lösen das Rechteck in rund 53 %, Hann in 1 %, Blackman und Kaiser in 0 % der Läufe auf; bei 1,5 Bins 100 / 48 / 40 / 42 %; bei 2,0 Bins 100 / 77 / 61 / 65 %; ab 2,5 Bins alle 100 %. |
| Hilft Auffüllen mit Nullen (Rechteck, 1,0 Bin Abstand)? | Der Anteil steigt von 0 % ohne Auffüllen auf 53 % bei Faktor 2 und 54 % ab Faktor 4; mehr ändert nichts. Die Grenze setzt das Fenster. |
| Sieht man eine schwache Linie (−40 dB) neben einer starken? | Bei ganzzahligem Abstand (Nullstelle des Rechtecks) sieht das Rechteck sie in rund 71 / 91 / 98 % der Läufe bei 3 / 8 / 12 Bins; Hann, Hamming, Blackman und Kaiser in 100 %; das Flat-Top bei 3 Bins gar nicht (0 %), ab 5 Bins in 100 %. Bei nicht ganzzahligem Abstand (3,5 bis 10,5 Bins, 300 Läufe) sieht das Rechteck sie in rund 0 %; Hann nur bei 3,5 Bins unvollständig (56 %), Hamming bei 3,5 bis 6,5 Bins in 55 bis 66 %, Blackman und Kaiser in 100 %, das Flat-Top ab 4,5 Bins in 100 %. |
| Wie weit liegt die Spitze allein daneben, wie die Interpolatoren (ohne Rauschen, n = 256)? | Spitze bis 0,50 Bin; Parabel bis 0,17; Quinn 1 und Rife bis 0,003; Jacobsen und Candan unter 0,001. |
| Wie nahe kommen sie der Cramér-Rao-Schranke (RMSE geteilt durch Schranke, 15 dB je Abtastwert, n = 256 / 1024)? | Spitze 49 / 98; Parabel 20 / 40; Rife 3,8 / 3,5; Jacobsen und Candan 1,6; Quinn 1 1,3; feines Maximum 1,1 (beide Längen); die Zahlen sind auf rund 4 % genau (300 Läufe). |
| Gibbs-Phänomen | Das Überschwingen der Teilsumme einer Rechteckschwingung beträgt rund 17,9 % der Amplitude und hängt ab etwa 50 Oberschwingungen nicht mehr von deren Zahl ab (bei 9: 18,2 %; Theorie 17,9 %); mit Fejér-Gewichten gibt es keines. |

## Ehrliche Grenzen

- **Zwei Linien, kein Rauschen** in den Auflösungsexperimenten; mit Rauschen wird die Grenze unschärfer, mehr als zwei Linien sind nicht untersucht.
- **Auflösung hängt vom Kriterium ab** (10 % Prominenz, Auffüllfaktor 16) und von Phase und Lage der Linien; eine andere Schwelle verschiebt die Zahlen, nicht die Reihenfolge der Fenster.
- **Die Schätzer lesen das ungefensterte Spektrum** ohne Auffüllen; sie sind für das Rechteckfenster hergeleitet. Die Schranke gilt für eine reelle Linie in weißem Rauschen fern von 0 und
  Nyquist; bei farbigem Rauschen oder mehreren Linien nicht in dieser Form.
- **Das feine Maximum** ist die Gitterverfeinerung des Periodogramms (32-fach, Parabel), keine unabhängige ML-Schätzung.
- **Eine Linie genau auf einem Gitterpunkt liegt in der Nullstelle des Rechtecks:** In den Presets liegt die schwache Linie deshalb nicht auf einem ganzzahligen Abstand (10,5 statt 10 Bins).
- **Die Streuung je Bin bleibt:** die relative Streuung (Std/Mittel = 1 bei Rauschen) senkt kein Fenster; im Leckage-Tal des Rechtecks ist sie sogar größer.

## Verifikation

- **Kern gegen Handrechnung:** Hann der Länge 8 (Summe 4, Summe der Quadrate 3, Rauschbandbreite 1,5), Gipfelverlust des Rechtecks bei N = 8 (−3,867 dB), Mini-Instanz der Schätzer
  (komplexer Ton bei 1,25 Bins, N = 8: Jacobsen 0,2378, Quinn 0,2378, Rife 0,2524, Candan 0,2508, Parabel 0,0913), Cramér-Rao-Schranke von Hand.
- **Orakel (`tests/test_oracle_fenster.py`):** Fenster gegen `scipy.signal.get_window` und `scipy.signal.windows.kaiser`; gefenstertes Periodogramm gegen `scipy.signal.periodogram`;
  exakter Erwartungswert gegen Simulation (z-Werte) und gegen die Doppelsumme; Gipfelzählung gegen `scipy.signal.find_peaks` auf 300 Zufallsfolgen; Gibbs gegen den Sinusintegral-Wert.
- **Generator** (`tests/test_generator.py`): wortgleich aus der Vorgängerdemo (Strom gegen unabhängige Fassung, Pole, AR-Filter, Dichte, Erwartungswert).
- **Zahlen:** jede Zahl dieser README in `tests/test_claims.py`; Festwerte der Experimente in `tests/test_experiments_golden.py`; Verdikte in `tests/test_verdicts.py`.
- **Oberfläche:** AppTest-Rauchtest (jeder Button, jeder Regler an den Grenzen), alle Fenster und Auffüllfaktoren, Permalink, Kaiser-Regler, Footer.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App |
| `signal_generator.py` | Gemeinsamer Generator der Reihe (wortgleich) |
| `fnl_windows.py` | Kern: Fenster, Kennzahlen, gefenstertes Periodogramm, Erwartungswert, Gipfelzählung, Gibbs |
| `fnl_estimators.py` | Schätzer der Linienlage, Schranke, feines Maximum |
| `fnl_constants.py` | Regler-Grenzen, feste Annahmen, Experimente, Presets |
| `fnl_evaluation.py` | Analyse, Experimente, Verdikte |
| `fnl_presets.py` | Permalink (Begrenzen), Presets, Zufalls-Seed-Button |
| `fnl_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `fnl_pdf_export.py` | PDF der aktuellen Einstellung (erst nach Klick) |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `requirements-dev.txt` | Laufzeit plus pytest und scipy (nur Gegenproben) |
| `tests/` | Kern, Orakel, Generator, Zahlen, Verdikte, Presets, Randfälle, AppTest, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

- **Mitteln und Taper:** Das Senken der Streuung (Bartlett und Welch, Multitaper) ist Thema der Folgedemos, nicht dieser.
- **Parametrische und Unterraumverfahren:** Sie umgehen oder durchbrechen die Auflösungsgrenze des Fensters; ebenfalls Folgedemos.
- **Fenster der Schätzer:** Die Interpolatoren sind für das Rechteckfenster hergeleitet; Varianten für Hann sind nicht enthalten.
- **Gemessene Daten:** alles ist simuliert, damit die Wahrheit bekannt ist.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Python 3.12 oder neuer.

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy, pandas und fpdf2.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Spektren und Wavelets: von Fourier zum Vergleich](https://sebastianhanisch.net/konzepte-spektral-wavelet-analyse.html).
