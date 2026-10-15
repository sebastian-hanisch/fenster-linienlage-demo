"""Analyse, Kennzahlen, Experimente (feste Ströme) und Verdikte der Demo. Reine Funktionen; Zufall nur über SplitMix64."""

import math
from dataclasses import dataclass

import numpy as np

import fnl_constants as C
import fnl_estimators as E
import fnl_windows as W
import signal_generator as G


def de(x: float, digits: int = 1) -> str:
    """Zahl mit Dezimalkomma."""
    return f"{x:.{digits}f}".replace(".", ",")


@dataclass(frozen=True)
class Settings:
    window: str = C.DEFAULT_WINDOW
    beta: float = C.DEFAULT_BETA
    n: int = C.DEFAULT_N
    pad: int = C.DEFAULT_PAD
    line: float = C.DEFAULT_LINE
    gap: float = C.DEFAULT_GAP
    rel: int = C.DEFAULT_REL
    snr: int = C.DEFAULT_SNR
    background: str = C.DEFAULT_BACKGROUND
    norm: str = C.DEFAULT_NORM
    seed: int = C.DEFAULT_SEED

    @property
    def f1_bins(self) -> float:
        return self.line * self.n

    @property
    def f2_bins(self) -> float:
        """Lage der Linie 2 in Bins: oberhalb von Linie 1; läge sie jenseits der Nyquist-Grenze (n / 2 - 2), dann unterhalb im selben Abstand."""
        up = self.line * self.n + self.gap
        return up if up <= self.n / 2 - 2 else self.line * self.n - self.gap


def noise_spec(background: str):
    """(Pole, Innovations-Sigma, weißes Sigma) eines Hintergrunds, auf Varianz 1 normiert."""
    poles = C.BACKGROUNDS[background]
    if poles:
        return poles, G.ar_unit_sigma(G.ar_from_poles(poles)), 0.0
    return (), 0.0, 1.0


def search_range(n: int) -> tuple[int, int]:
    """Bin-Bereich [lo, hi), in dem die Spitze der Linie gesucht wird: ohne die zwei Randbins an jedem Ende (Gleichanteil, Nyquist, Spiegelung)."""
    return 2, n // 2 - 2


def amplitude(snr_db: float) -> float:
    return math.sqrt(2.0 * 10.0 ** (snr_db / 10.0))


@dataclass(frozen=True)
class Analysis:
    settings: Settings
    f_bins: np.ndarray            # Frequenzachse der aufgefüllten Spektren in Bins
    x: np.ndarray
    w: np.ndarray
    spec: np.ndarray              # gefenstertes Periodogramm in der gewählten Normierung
    expected: np.ndarray          # exakter Erwartungswert
    true_density: np.ndarray      # wahre Dichte des Hintergrunds (nur in Leistungsdichte vergleichbar)
    enbw: float
    sidelobe_db: float
    main_lobe: float
    scallop_db: float
    resolved: bool                # beide Linien als zwei Gipfel
    weak_visible: bool            # Linie 2 hebt ihren Wert um >= 3 dB gegen dasselbe Signal ohne sie
    estimates: dict               # Schätzung der Lage der Linie 1 in Bins je Schätzer (Rechteckfenster, ohne Auffüllen)
    errors: dict                  # Fehler in Bins gegen die wahre Lage
    ml: float                     # feines Maximum in Bins
    crb: float                    # Cramér-Rao-Schranke in Bins


def build_signal(s: Settings):
    rng = G.SplitMix64(s.seed)
    ph = 2.0 * math.pi * rng.uniform(2)
    poles, ar_sigma, white = noise_spec(s.background)
    f1_hz = s.f1_bins * C.FS / s.n
    f2_hz = s.f2_bins * C.FS / s.n
    a1 = amplitude(s.snr)
    a2 = a1 * 10.0 ** (s.rel / 20.0)
    lines = [(f1_hz, a1, float(ph[0])), (f2_hz, a2, float(ph[1]))]
    sig = G.make_signal(s.n, C.FS, lines, poles, ar_sigma, white, rng)
    return sig, lines, (poles, ar_sigma, white)


def analyse(s: Settings) -> Analysis:
    sig, lines, (poles, ar_sigma, white) = build_signal(s)
    n, pad = s.n, s.pad
    w = W.window(s.window, n, s.beta)
    spec = W.spectrum(sig.x, w, C.FS, pad, s.norm)
    f_bins = np.arange(len(spec)) / pad
    f_hz = f_bins * C.FS / n
    r = G.noise_autocov(poles, ar_sigma, white, n)
    expected = W.expected_spectrum(f_hz, C.FS, w, r, sig.clean, s.norm)
    true_density = G.true_psd(f_hz, C.FS, poles, ar_sigma, white)
    power = W.spectrum(sig.x, w, C.FS, pad, "Leistungsdichte")
    resolved = W.two_lines_resolved(power, pad, s.f1_bins, s.f2_bins)
    line2 = G.line_signal(n, C.FS, [lines[1]])
    without = W.spectrum(sig.x - line2, w, C.FS, pad, "Leistungsdichte")
    k2 = int(round(s.f2_bins * pad))                  # f2_bins <= n / 2 - 2: liegt immer im Raster
    weak_visible = bool(power[k2] >= 2.0 * without[k2])
    X = np.fft.fft(sig.x)[None, : n // 2 + 1]
    lo, hi = search_range(n)
    est = {name: float(v[0]) for name, v in E.estimate_all(X, lo, hi).items()}
    ml = float(E.refined_maximum(sig.x, lo, hi)[0])
    err = {name: v - s.f1_bins for name, v in est.items()}
    return Analysis(s, f_bins, sig.x, w, spec, expected, true_density, W.enbw(w), W.sidelobe_peak_db(w), W.main_lobe_3db(w), W.scallop_loss_db(w),
                    bool(resolved), weak_visible, est, err, ml - s.f1_bins, E.crb_bins(n, s.snr))


# --- Experimente ---------------------------------------------------------------------------------------------------------------------------


def table_windows(n: int = C.EXP_TABLE_N, beta: float = C.DEFAULT_BETA) -> list[dict]:
    """Kennzahlen je Fenster (deterministisch)."""
    rows = []
    for name in C.WINDOWS:
        w = W.window(name, n, beta)
        e = W.enbw(w)
        rows.append({"window": name, "enbw": e, "main_lobe": W.main_lobe_3db(w), "sidelobe_db": W.sidelobe_peak_db(w), "scallop_db": W.scallop_loss_db(w),
                     "line_vs_density_db": 10.0 * math.log10(e)})
    return rows


def valley_bias(ns=C.EXP_VALLEY_N, beta: float = C.DEFAULT_BETA, frac: float = C.EXP_VALLEY_FRAC, background: str = "Steil") -> list[dict]:
    """Fehler des Erwartungswerts im Tal gegen die wahre Dichte (exakte Formel, deterministisch), je Fenster und Länge."""
    poles, ar_sigma, white = noise_spec(background)
    rows = []
    for n in ns:
        k = int(round(frac * n))
        f = k * C.FS / n
        s_true = float(G.true_psd([f], C.FS, poles, ar_sigma, white)[0])
        r = G.noise_autocov(poles, ar_sigma, white, n)
        for name in C.WINDOWS:
            e = float(W.expected_spectrum([f], C.FS, W.window(name, n, beta), r)[0])
            rows.append({"n": n, "window": name, "bias_db": 10.0 * math.log10(e / s_true)})
    return rows


def _two_line_signal(n: int, f1: float, f2: float, a2: float, ph1, ph2):
    t = np.arange(n)
    return (np.cos(2.0 * np.pi * f1[:, None] * t[None, :] / n + ph1[:, None]) + a2 * np.cos(2.0 * np.pi * f2[:, None] * t[None, :] / n + ph2[:, None]))


def sweep_resolution(windows=C.EXP_RES_WINDOWS, deltas=C.EXP_RES_DELTAS, n: int = C.EXP_RES_N, runs: int = C.EXP_RES_RUNS, pad: int = 16,
                     beta: float = C.DEFAULT_BETA) -> list[dict]:
    """Anteil der Läufe, in denen zwei gleich starke Linien (zufällige Phasen und Lage zum Gitter) im Abstand delta als zwei Gipfel erscheinen."""
    rows = []
    rng = G.SplitMix64(C.SWEEP_SEED)
    for d in deltas:
        ph1, ph2 = 2.0 * math.pi * rng.uniform(runs), 2.0 * math.pi * rng.uniform(runs)
        center = 40.0 + 0.5 * rng.uniform(runs)
        f1, f2 = center - d / 2, center + d / 2
        x = _two_line_signal(n, f1, f2, 1.0, ph1, ph2)
        row = {"delta": d, "runs": runs}
        for name in windows:
            w = W.window(name, n, beta)
            sp = np.abs(np.fft.rfft(x * w[None, :], n * pad, axis=1)) ** 2
            row[name] = float(np.mean([W.two_lines_resolved(sp[i], pad, f1[i], f2[i]) for i in range(runs)]))
        rows.append(row)
    return rows


def sweep_padding(deltas=C.EXP_PAD_DELTAS, pads=C.EXP_PAD_FACTORS, n: int = C.EXP_RES_N, runs: int = C.EXP_RES_RUNS) -> list[dict]:
    """Wie sweep_resolution, Rechteckfenster, gegen den Auffüllfaktor."""
    rows = []
    rng = G.SplitMix64(C.SWEEP_SEED + 1)
    w = np.ones(n)
    for d in deltas:
        ph1, ph2 = 2.0 * math.pi * rng.uniform(runs), 2.0 * math.pi * rng.uniform(runs)
        center = 40.0 + 0.5 * rng.uniform(runs)
        f1, f2 = center - d / 2, center + d / 2
        x = _two_line_signal(n, f1, f2, 1.0, ph1, ph2)
        row = {"delta": d, "runs": runs}
        for pad in pads:
            sp = np.abs(np.fft.rfft(x * w[None, :], n * pad, axis=1)) ** 2
            row[pad] = float(np.mean([W.two_lines_resolved(sp[i], pad, f1[i], f2[i]) for i in range(runs)]))
        rows.append(row)
    return rows


def sweep_weak(windows=C.WINDOWS, distances=C.EXP_WEAK_DISTANCES, rel_db: float = C.EXP_WEAK_REL_DB, n: int = C.EXP_RES_N, runs: int = C.EXP_WEAK_RUNS,
               pad: int = 16, beta: float = C.DEFAULT_BETA) -> list[dict]:
    """Anteil der Läufe, in denen eine schwache Linie ohne Rauschen neben einer starken ihren Spektralwert um mindestens 3 dB gegen dasselbe Signal ohne sie hebt."""
    rows = []
    rng = G.SplitMix64(C.SWEEP_SEED + 2)
    a2 = 10.0 ** (rel_db / 20.0)
    t = np.arange(n)
    for d in distances:
        ph1, ph2 = 2.0 * math.pi * rng.uniform(runs), 2.0 * math.pi * rng.uniform(runs)
        f1 = 40.0 + rng.uniform(runs)
        f2 = f1 + d
        strong = np.cos(2.0 * np.pi * f1[:, None] * t[None, :] / n + ph1[:, None])
        both = strong + a2 * np.cos(2.0 * np.pi * f2[:, None] * t[None, :] / n + ph2[:, None])
        k = np.round(f2 * pad).astype(int)
        row = {"distance": d, "runs": runs}
        for name in windows:
            w = W.window(name, n, beta)
            sp_b = np.abs(np.fft.rfft(both * w[None, :], n * pad, axis=1)) ** 2
            sp_s = np.abs(np.fft.rfft(strong * w[None, :], n * pad, axis=1)) ** 2
            idx = np.arange(runs)
            row[name] = float(np.mean(sp_b[idx, k] >= 2.0 * sp_s[idx, k]))
        rows.append(row)
    return rows


def estimator_bias(n: int = 256, offsets: int = C.EXP_BIAS_OFFSETS) -> list[dict]:
    """Größte und mittlere Abweichung der Schätzer ohne Rauschen über die Lage der Linie zwischen -0,5 und 0,5 Bins um das Bin n / 4."""
    base = n // 4
    t = np.arange(n)
    deltas = np.linspace(-0.5, 0.5, offsets)
    truth = base + deltas
    x = np.cos(2.0 * np.pi * truth[:, None] * t[None, :] / n + 0.3)
    X = np.fft.fft(x, axis=1)[:, : n // 2 + 1]
    est = E.estimate_all(X, *search_range(n))
    return [{"estimator": name, "max": float(np.max(np.abs(v - truth))), "mean": float(np.mean(np.abs(v - truth)))} for name, v in est.items()]


def estimator_rmse(ns=C.EXP_EST_N, snrs=C.EXP_EST_SNR, runs: int = C.EXP_EST_RUNS) -> list[dict]:
    """RMSE der Schätzer geteilt durch die Cramér-Rao-Schranke (Linie in weißem Rauschen, zufällige Lage und Phase)."""
    rows = []
    rng = G.SplitMix64(C.SWEEP_SEED + 3)
    for n in ns:
        for snr_db in snrs:
            k0 = np.floor(n / 8 + (n / 8) * rng.uniform(runs))
            ftrue = k0 + rng.uniform(runs) - 0.5
            ph = 2.0 * math.pi * rng.uniform(runs)
            t = np.arange(n)
            x = amplitude(snr_db) * np.cos(2.0 * np.pi * ftrue[:, None] * t[None, :] / n + ph[:, None]) + rng.normal(runs * n).reshape(runs, n)
            X = np.fft.fft(x, axis=1)[:, : n // 2 + 1]
            est = E.estimate_all(X, *search_range(n))
            est["ML (fein)"] = E.refined_maximum(x, *search_range(n))
            crb = E.crb_bins(n, snr_db)
            row = {"n": n, "snr": snr_db, "crb": crb, "runs": runs}
            for name, v in est.items():
                err = v - ftrue
                err = err[np.abs(err) < 5.0]                  # falsche Spitzen (Rauschen) zählen getrennt
                row[name] = float(np.sqrt(np.mean(err ** 2)) / crb)
            row["outliers"] = float(np.mean(np.abs(est["Spitze"] - ftrue) >= 5.0))
            rows.append(row)
    return rows


def gibbs_table(ms=C.EXP_GIBBS_M) -> list[dict]:
    return [{"m": m, "rect": W.gibbs_overshoot(m, False), "fejer": W.gibbs_overshoot(m, True)} for m in ms]


# --- Verdikte (reine Funktionen: (Art, Text)) ----------------------------------------------------------------------------------------------


def verdict_resolution(a: Analysis) -> tuple[str, str]:
    s = a.settings
    if a.resolved:
        return "success", (f"Die beiden Linien im Abstand von {de(s.gap, 2)} Bins erscheinen als **zwei Gipfel**; die Hauptkeule des Fensters „{s.window}“ ist "
                           f"{de(a.main_lobe, 2)} Bins breit, die höchste Nebenkeule liegt bei {de(a.sidelobe_db, 1)} dB.")
    return "warning", (f"Die beiden Linien im Abstand von {de(s.gap, 2)} Bins **verschmelzen**: Die Hauptkeule des Fensters „{s.window}“ ist {de(a.main_lobe, 2)} Bins breit, "
                       f"Auffüllen mit Nullen ändert daran nichts.")


def verdict_weak_line(a: Analysis) -> tuple[str, str]:
    s = a.settings
    if a.weak_visible:
        return "success", (f"Die Linie {abs(s.rel)} dB unter der starken, {de(s.gap, 2)} Bins entfernt, hebt ihren Wert um mindestens 3 dB: Das Fenster „{s.window}“ "
                           f"lässt sie **sichtbar** (höchste Nebenkeule {de(a.sidelobe_db, 1)} dB).")
    return "warning", (f"Die Linie {abs(s.rel)} dB unter der starken, {de(s.gap, 2)} Bins entfernt, ist **nicht zu sehen**: Die Nebenkeulen des Fensters „{s.window}“ "
                       f"(höchste {de(a.sidelobe_db, 1)} dB) decken sie zu.")


def verdict_table(rows: list[dict]) -> tuple[str, str]:
    by = {r["window"]: r for r in rows}
    return "info", (f"Das Rechteck hat die schmalste Hauptkeule ({de(by['Rechteck']['main_lobe'], 2)} Bins), aber die höchste Nebenkeule ({de(by['Rechteck']['sidelobe_db'], 1)} dB) "
                    f"und verliert zwischen zwei Bins {de(abs(by['Rechteck']['scallop_db']), 1)} dB; Hann liegt bei {de(by['Hann']['main_lobe'], 2)} Bins, {de(by['Hann']['sidelobe_db'], 1)} dB "
                    f"und {de(abs(by['Hann']['scallop_db']), 1)} dB. **Kein Fenster ist in allem besser.**")


def verdict_valley(rows: list[dict]) -> tuple[str, str]:
    first = min(r["n"] for r in rows)
    rect = [r for r in rows if r["window"] == "Rechteck" and r["n"] == first][0]
    hann = [r for r in rows if r["window"] == "Hann" and r["n"] == first][0]
    return "success", (f"Bei n = {first} liegt der Erwartungswert im Tal mit dem Rechteck um {de(rect['bias_db'], 1)} dB über der wahren Dichte, mit Hann um {de(hann['bias_db'], 1)} dB: "
                       "**das Fenster senkt die Leckage, nicht die Streuung.**")


def verdict_resolution_sweep(rows: list[dict]) -> tuple[str, str]:
    d = 1.5
    r = [x for x in rows if x["delta"] == d][0]
    return "warning", (f"Bei {de(d, 1)} Bins Abstand erscheinen die Linien mit dem Rechteck in {de(100 * r['Rechteck'], 0)} % der Läufe als zwei Gipfel, "
                       f"mit Hann in {de(100 * r['Hann'], 0)} %, mit Blackman in {de(100 * r['Blackman'], 0)} %: **was die Nebenkeulen verbessern, kostet Auflösung.**")


def verdict_padding(rows: list[dict]) -> tuple[str, str]:
    r = [x for x in rows if x["delta"] == 1.0][0]
    return "info", (f"Bei 1,0 Bin Abstand steigt der Anteil aufgelöster Läufe von {de(100 * r[1], 0)} % ohne Auffüllen auf {de(100 * r[4], 0)} % bei vierfachem und "
                    f"{de(100 * r[16], 0)} % bei 16-fachem Auffüllen: Ab Faktor 4 **ändert sich nichts mehr**, die Grenze setzt das Fenster.")


def verdict_weak(rows: list[dict]) -> tuple[str, str]:
    r = [x for x in rows if x["distance"] == 8.0][0]
    return "info", (f"Eine Linie 40 dB unter der starken, 8 Bins entfernt, hebt ihren Wert mit dem Rechteck in {de(100 * r['Rechteck'], 0)} % der Läufe um mindestens 3 dB, "
                    f"mit Hann in {de(100 * r['Hann'], 0)} %.")


def verdict_estimators(rows: list[dict], bias: list[dict]) -> tuple[str, str]:
    r = [x for x in rows if x["n"] == max(C.EXP_EST_N) and x["snr"] == max(C.EXP_EST_SNR)][0]
    b = {x["estimator"]: x for x in bias}
    return "success", (f"Die Spitze allein liegt bis {de(b['Spitze']['max'], 2)} Bin daneben und erreicht das {de(r['Spitze'], 0)}-fache der Schranke; Quinn und Jacobsen "
                       f"(drei Bins) liegen bei {de(r['Quinn 1'], 1)}- bzw. {de(r['Jacobsen'], 1)}-fach, das feine Maximum bei {de(r['ML (fein)'], 2)}-fach.")


def verdict_gibbs(rows: list[dict]) -> tuple[str, str]:
    r = rows[-1]
    return "info", (f"Das Überschwingen der Teilsumme beträgt {de(100 * r['rect'], 1)} % der Amplitude, auch bei {r['m']} Oberschwingungen (Theorie 17,9 %); "
                    f"mit Fejér-Gewichten bleibt es bei {de(100 * max(r['fejer'], 0.0), 1)} %.")
