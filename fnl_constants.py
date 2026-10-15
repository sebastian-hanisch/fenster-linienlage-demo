"""Konstanten: Regler-Grenzen, feste Annahmen, Hintergründe, Experimente, Farben und Presets (Kern: fnl_windows.py, fnl_estimators.py)."""

import fnl_windows as W

FS = 1000.0                                  # Abtastrate in Hz (feste Annahme)
N_MIN, N_MAX, N_STEP, DEFAULT_N = 128, 1024, 64, 256     # Kosten: O(n * n * pad) für den exakten Erwartungswert bei n = 1024, pad = 16
PAD_CHOICES = (1, 2, 4, 8, 16)
DEFAULT_PAD = 8
LINE_MIN, LINE_MAX, LINE_STEP, DEFAULT_LINE = 0.05, 0.45, 0.0005, 0.2     # Linie 1 als Bruchteil der Abtastrate (bins = n * Bruchteil)
GAP_MIN, GAP_MAX, GAP_STEP, DEFAULT_GAP = 0.5, 20.0, 0.25, 1.5            # Abstand der Linie 2 in Bins
REL_MIN, REL_MAX, DEFAULT_REL = -70, 0, 0                                  # Stärke der Linie 2 gegen Linie 1 in dB
SNR_MIN, SNR_MAX, DEFAULT_SNR = -10, 50, 30                                # Linie 1 gegen Rauschvarianz 1, in dB je Abtastwert
BETA_MIN, BETA_MAX, DEFAULT_BETA = 0.0, 14.0, 8.0                          # Kaiser-Parameter
SEED_MAX, DEFAULT_SEED = 999999, 35
DEFAULT_WINDOW = "Hann"
WINDOWS = W.NAMES
NORMS = W.NORMS
DEFAULT_NORM = NORMS[0]

BACKGROUNDS = {"weiß": (), "Steil": ((0.2, 0.99), (0.0, 0.9), (0.0, 0.9))}      # Pole je Hintergrund, Varianz auf 1 normiert
DEFAULT_BACKGROUND = "weiß"

# Experimente (feste Ströme)
SWEEP_SEED = 200000
EXP_TABLE_N = 256
EXP_VALLEY_N = (128, 512, 2048)
EXP_VALLEY_FRAC = 0.49
EXP_RES_N = 256
EXP_RES_RUNS = 150
EXP_RES_DELTAS = (1.0, 1.25, 1.5, 2.0, 2.5, 3.0)
EXP_RES_WINDOWS = ("Rechteck", "Hann", "Blackman", "Kaiser")
EXP_PAD_DELTAS = (1.0, 1.25, 1.5, 2.0)
EXP_PAD_FACTORS = (1, 2, 4, 8, 16)
EXP_WEAK_REL_DB = -40
EXP_WEAK_DISTANCES = (3.0, 5.0, 8.0, 12.0)
EXP_WEAK_RUNS = 100
EXP_EST_N = (64, 256, 1024)
EXP_EST_SNR = (5, 15)
EXP_EST_RUNS = 300
EXP_BIAS_OFFSETS = 101
EXP_GIBBS_M = (9, 49, 199, 999)

COLOR_I = "rgba(76,120,168,0.75)"
COLOR_EXPECTED = "#E8833A"
COLOR_TRUE = "#2F6B65"
COLOR_LINE = "#B03A2E"
COLOR_WINDOW = "#7B5EA7"

PRESETS = {
    "Zwei nahe Linien": {"n": 256, "pad": 8, "window": "Hann", "line": 0.2, "gap": 1.5, "rel": 0, "snr": 30, "background": "weiß", "norm": "Leistungsdichte",
                         "beta": 8.0, "seed": DEFAULT_SEED},
    "Schwache neben starker": {"n": 256, "pad": 8, "window": "Rechteck", "line": 0.2, "gap": 10.5, "rel": -55, "snr": 40, "background": "weiß",
                               "norm": "Leistungsdichte", "beta": 8.0, "seed": DEFAULT_SEED},
    "Linie zwischen Bins": {"n": 256, "pad": 8, "window": "Rechteck", "line": 0.2012, "gap": 20.0, "rel": -70, "snr": 30, "background": "weiß",
                            "norm": "Linienhöhe", "beta": 8.0, "seed": DEFAULT_SEED},
    "Steiler Hintergrund": {"n": 128, "pad": 4, "window": "Rechteck", "line": 0.1, "gap": 20.0, "rel": -70, "snr": 20, "background": "Steil",
                            "norm": "Leistungsdichte", "beta": 8.0, "seed": DEFAULT_SEED},
}
PRESET_HELP = {
    "Zwei nahe Linien": "Zwei gleich starke Linien im Abstand von 1,5 Bins, Hann-Fenster: Mit dem Rechteck wären sie aufgelöst, mit Hann verschmelzen sie (Hauptkeule 1,5 Bins breit).",
    "Schwache neben starker": "Eine Linie 55 dB unter der starken, 10 Bins entfernt, Rechteckfenster: Die Nebenkeulen der starken Linie decken sie zu; mit Hann oder Blackman erscheint sie.",
    "Linie zwischen Bins": "Linie fast genau zwischen zwei Bins, Rechteckfenster, Linienhöhe: Der Gipfel liegt knapp 4 dB zu tief und die Spitze 0,5 Bin neben der Linie.",
    "Steiler Hintergrund": "Steil abfallender Hintergrund mit 57 dB Spannweite, Rechteckfenster: Im Tal liegt das Spektrum weit über der wahren Dichte; mit Hann liegt es darauf.",
}
