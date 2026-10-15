"""Gezielte Tests für Stellen, die der Mutationstest als ungeprüft meldete (Randfälle, Gleichstände, Grenzen, unabhängige Referenzfassungen)."""

import math

import numpy as np
import pytest

import fnl_constants as C
import fnl_estimators as E
import fnl_evaluation as ev
import fnl_windows as W
import signal_generator as G


def approx(x):
    return pytest.approx(x, rel=1e-6, abs=1e-9)


# --- Schätzer: unabhängige skalare Referenzfassung -----------------------------------------------------------------------------------------


def quinn1_reference(x_m, x_0, x_p):
    a1 = (x_m / x_0).real
    a2 = (x_p / x_0).real
    d1 = a1 / (1 - a1)
    d2 = -a2 / (1 - a2)
    if d1 > 0 and d2 > 0:
        return d2
    return d1


def test_quinn_matches_a_scalar_reference_on_random_tones_with_and_without_noise():
    rng = G.SplitMix64(41)
    n = 64
    t = np.arange(n)
    for trial in range(200):
        f = 10.0 + 20.0 * rng.uniform(1)[0]
        x = np.cos(2 * np.pi * f * t / n + 6.0 * rng.uniform(1)[0]) + (0.3 * rng.normal(n) if trial % 2 else 0.0)
        X = np.fft.fft(x)[None, : n // 2 + 1]
        k = E.peak_bins(X, 2, n // 2 - 2)
        xm, x0, xp = E.three_bins(X, k)
        assert E.est_quinn1(xm, x0, xp, n)[0] == pytest.approx(quinn1_reference(xm[0], x0[0], xp[0]), rel=1e-12, abs=1e-12)


def test_rife_matches_a_scalar_reference():
    rng = G.SplitMix64(42)
    n = 64
    for _ in range(100):
        f = 10.0 + 20.0 * rng.uniform(1)[0]
        X = np.fft.fft(np.cos(2 * np.pi * f * np.arange(n) / n + 0.4))[None, : n // 2 + 1]
        k = E.peak_bins(X, 2, n // 2 - 2)
        xm, x0, xp = E.three_bins(X, k)
        am, a0, ap = abs(xm[0]), abs(x0[0]), abs(xp[0])
        ref = ap / (a0 + ap) if ap > am else -am / (a0 + am)
        assert E.est_rife(xm, x0, xp, n)[0] == pytest.approx(ref, rel=1e-12, abs=1e-12)


def test_search_range_excludes_two_bins_at_each_end():
    assert ev.search_range(128) == (2, 62) and ev.search_range(1024) == (2, 510) and ev.search_range(130) == (2, 63)


def test_peak_search_ignores_a_stronger_bin_inside_the_excluded_margin():
    X = np.zeros((1, 33), dtype=complex)
    X[0, 1] = 10.0                     # im ausgeschlossenen Rand
    X[0, 20] = 3.0
    X[0, 31] = 20.0                    # im ausgeschlossenen Rand (n = 64: hi = 30)
    lo, hi = ev.search_range(64)
    assert E.peak_bins(X, lo, hi)[0] == 20


def test_refined_estimate_of_a_clean_line_is_close_to_the_truth():
    a = ev.analyse(ev.Settings(window="Rechteck", pad=1, snr=40, rel=-70, line=0.3123, gap=20.0, n=256))
    assert abs(a.ml) < 0.02 and abs(a.errors["Candan"]) < 0.01


# --- Lage der Linie 2 -------------------------------------------------------------------------------------------------------------------------


def test_second_line_sits_exactly_at_the_limit_above_the_first():
    s = ev.Settings(n=128, line=0.4375, gap=6.0)           # 56 + 6 = 62 = n / 2 - 2
    assert s.f1_bins == 56.0 and s.f2_bins == 62.0


def test_second_line_goes_below_just_after_the_limit():
    assert ev.Settings(n=128, line=0.4375, gap=6.5).f2_bins == pytest.approx(49.5)       # 62,5 > 62
    assert ev.Settings(n=128, line=0.4375, gap=8.0).f2_bins == pytest.approx(48.0)       # 64: knapp über der Grenze, nicht erst jenseits n / 2 + 2
    assert ev.Settings(n=128, line=0.4375, gap=5.75).f2_bins == pytest.approx(61.75)


# --- Verdikt, Spalten, Gipfelzählung -----------------------------------------------------------------------------------------------------------


def test_valley_verdict_picks_the_hann_row_of_the_shortest_length_even_if_other_rows_come_first():
    rows = [{"n": 512, "window": "Hann", "bias_db": 9.9}, {"n": 128, "window": "Rechteck", "bias_db": 17.9}, {"n": 128, "window": "Hann", "bias_db": 0.0},
            {"n": 512, "window": "Rechteck", "bias_db": 11.9}]
    kind, text = ev.verdict_valley(rows)
    assert kind == "success" and "n = 128" in text and "17,9 dB" in text and "um 0,0 dB" in text and "9,9" not in text


def test_count_peaks_edge_cases():
    assert W.count_peaks([0, 1, 0], 0.1) == 1                       # kürzeste Folge mit innerem Punkt
    assert W.count_peaks([0, 1], 0.1) == 0
    assert W.count_peaks([0, 1, 2, 3, 0], 0.0) == 1                 # steigende Flanke ist kein Gipfel
    assert W.count_peaks([0, 3, 2, 1, 0], 0.0) == 1
    assert W.count_peaks([0, 1, 1, 0], 0.1) == 1                    # Plateau: ein Gipfel
    assert W.count_peaks([0, 1, 0], 1.0) == 1                       # Prominenz genau an der Schwelle zählt


def test_count_peaks_equal_height_neighbours_use_the_lower_base_beyond_them():
    assert W.count_peaks([0, 3, 1, 3, 5, 0], 0.5) == 1
    assert W.count_peaks([0, 5, 3, 1, 3, 0], 0.5) == 1
    assert W.count_peaks([0, 3, 1, 3, 5, 0], 0.5) != W.count_peaks([0, 3, 1, 3, 5, 0], 0.3)


# --- Normierung und Gibbs ------------------------------------------------------------------------------------------------------------------------


def test_line_height_normalisation_of_the_expected_spectrum():
    n, fs = 16, 8.0
    w = W.window("Hann", n)
    r = G.ar_autocov([0.5], 1.0, n)
    s = np.cos(2 * np.pi * 3 * np.arange(n) / n)
    f = np.array([0.0, 1.0, 3.0 * fs / n])
    dens = W.expected_spectrum(f, fs, w, r, s, "Leistungsdichte")
    line = W.expected_spectrum(f, fs, w, r, s, "Linienhöhe")
    assert np.allclose(line / dens, 2 * fs * (w ** 2).sum() / w.sum() ** 2)
    assert line[2] > 0.0


def gibbs_reference(m_max, fejer):
    th = np.linspace(1e-9, math.pi - 1e-9, 40001)
    s = np.zeros_like(th)
    for k in range(1, m_max + 1, 2):
        wk = (1 - k / (m_max + 1.0)) if fejer else 1.0
        s += wk * 4 / (math.pi * k) * np.sin(k * th)
    return s.max() - 1


@pytest.mark.parametrize("m,fejer", [(9, False), (10, False), (12, False), (10, True), (49, True)])
def test_gibbs_matches_a_plain_reference(m, fejer):
    assert W.gibbs_overshoot(m, fejer) == pytest.approx(gibbs_reference(m, fejer), abs=5e-4)


def test_fejer_sum_stays_below_one_and_approaches_it():
    values = [W.gibbs_overshoot(m, True) for m in (9, 49, 199)]
    assert all(v < 0 for v in values) and values == sorted(values)
    assert values[-1] == pytest.approx(-2 / (math.pi * 200), rel=0.05)


# --- Sweeps: Festwerte mit mehr Läufen (fangen Vorzeichen, Exponenten, Lagen) ---------------------------------------------------------------


def test_resolution_sweep_fixed_values_with_sixty_runs():
    rows = ev.sweep_resolution(windows=("Rechteck", "Hann", "Blackman", "Kaiser"), deltas=(1.0, 1.25, 1.5, 2.0), runs=60)
    expect = [(1.0, 0.55, 0.05, 0.0, 0.0), (1.25, 46 / 60, 0.4, 13 / 60, 19 / 60), (1.5, 1.0, 37 / 60, 31 / 60, 33 / 60), (2.0, 1.0, 0.8, 40 / 60, 42 / 60)]
    for r, (d, a, b, c, k) in zip(rows, expect):
        assert r["delta"] == d
        assert (r["Rechteck"], r["Hann"], r["Blackman"], r["Kaiser"]) == (approx(a), approx(b), approx(c), approx(k))


def test_padding_sweep_fixed_values_with_sixty_runs():
    rows = ev.sweep_padding(deltas=(1.0, 1.25, 1.5), pads=(1, 2, 4), runs=60)
    expect = [(0.0, 29 / 60, 31 / 60), (7 / 60, 37 / 60, 39 / 60), (20 / 60, 1.0, 1.0)]
    for r, (p1, p2, p4) in zip(rows, expect):
        assert (r[1], r[2], r[4]) == (approx(p1), approx(p2), approx(p4))


def test_weak_line_sweep_fixed_values_with_sixty_runs():
    rows = ev.sweep_weak(windows=("Rechteck", "Hann", "Flat-Top"), distances=(3.0, 5.0, 8.0), runs=60)
    expect = [(46 / 60, 1.0, 0.0), (50 / 60, 1.0, 1.0), (56 / 60, 1.0, 1.0)]
    for r, (a, b, c) in zip(rows, expect):
        assert (r["Rechteck"], r["Hann"], r["Flat-Top"]) == (approx(a), approx(b), approx(c))


def test_resolved_criterion_looks_only_half_a_distance_beyond_the_second_line():
    pad = 4
    bins = np.arange(0, 40, 1 / pad)
    merged = np.exp(-((bins - 10.75) / 0.9) ** 2)                 # ein breiter Gipfel zwischen den Linien bei 10 und 11,5
    far = merged.copy()
    far[int(12.6 * pad)] += 0.3                                   # Spitze bei 12,6 Bins: jenseits von f2 + d / 2 = 12,25 und vor f2 + d = 13
    assert not W.two_lines_resolved(far, pad, 10.0, 11.5)
    edge = merged.copy()
    edge[49] += 0.3                                               # letzter Punkt des Ausschnitts (12,25 Bins): nur als Rand, zählt nicht
    assert not W.two_lines_resolved(edge, pad, 10.0, 11.5)
