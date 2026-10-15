"""Kern: Fenster, Kennzahlen, gefenstertes Periodogramm, exakter Erwartungswert, Gipfelzählung, Gibbs (Einheiten gegen Handrechnung)."""

import math

import numpy as np
import pytest

import fnl_windows as W
import signal_generator as G


def test_hann_of_length_8_by_hand(hann8):
    w = W.window("Hann", 8)
    assert np.allclose(w, hann8, atol=1e-12)
    assert w.sum() == pytest.approx(4.0) and (w ** 2).sum() == pytest.approx(3.0)
    assert W.enbw(w) == pytest.approx(1.5)
    assert W.coherent_gain(w) == pytest.approx(0.5)


def test_rectangular_window_is_all_ones_and_has_unit_bandwidth():
    w = W.window("Rechteck", 16)
    assert w.tolist() == [1.0] * 16 and W.enbw(w) == 1.0 and W.coherent_gain(w) == 1.0


def test_hamming_bandwidth_by_formula():
    # Summe w = 0,54 n, Summe w^2 = n (0,54^2 + 0,46^2 / 2): ENBW = 0,3974 / 0,2916
    assert W.enbw(W.window("Hamming", 512)) == pytest.approx((0.54 ** 2 + 0.46 ** 2 / 2) / 0.54 ** 2, rel=1e-6)


def test_kaiser_with_beta_zero_is_the_rectangular_window():
    assert np.allclose(W.window("Kaiser", 32, 0.0), 1.0)


def test_unknown_window_is_rejected():
    with pytest.raises(ValueError):
        W.window("Gauß", 8)


def test_scallop_loss_of_the_rectangular_window_by_hand():
    # N = 8: |sin(pi/2) / (8 sin(pi/16))| = 0,6407 -> -3,867 dB; N -> unendlich: 20 log10 sinc(1/2) = -3,922 dB
    assert W.scallop_loss_db(np.ones(8)) == pytest.approx(20 * math.log10(1 / (8 * math.sin(math.pi / 16))), abs=1e-9)
    assert W.scallop_loss_db(np.ones(8)) == pytest.approx(-3.867, abs=1e-3)
    assert W.scallop_loss_db(np.ones(4096)) == pytest.approx(-3.922, abs=1e-3)
    assert W.scallop_loss_db(W.window("Flat-Top", 256)) == pytest.approx(0.0, abs=0.02)


def test_main_lobe_and_sidelobes_match_the_textbook_values():
    table = {"Rechteck": (0.89, -13.3), "Hann": (1.44, -31.5), "Hamming": (1.30, -42.7), "Blackman": (1.64, -58.1)}
    for name, (lobe, side) in table.items():
        w = W.window(name, 256)
        assert W.main_lobe_3db(w) == pytest.approx(lobe, abs=0.03), name
        assert W.sidelobe_peak_db(w) == pytest.approx(side, abs=0.3), name


def test_response_is_normalised_and_symmetric_in_the_window():
    f, r = W.response(W.window("Hann", 64), 16)
    assert r[0] == pytest.approx(1.0) and f[1] == pytest.approx(1 / 16) and f[-1] == pytest.approx(32.0)


def test_spectrum_normalisations_by_hand(cosine8):
    w = np.ones(8)
    dens = W.spectrum(cosine8, w, 1.0, 1, "Leistungsdichte")
    line = W.spectrum(cosine8, w, 1.0, 1, "Linienhöhe")
    assert dens[2] == pytest.approx(16 / 8)                 # |X|^2 / (fs sum w^2)
    assert line[2] == pytest.approx(2 * 16 / 64)            # = A^2 / 2 mit A = 1
    assert np.allclose(np.delete(dens, 2), 0.0, atol=1e-12)
    with pytest.raises(ValueError):
        W.spectrum(cosine8, w, 1.0, 1, "unsinn")


def test_hann_makes_the_line_height_independent_of_the_window_gain(cosine8):
    # Linienhöhe: 2 |X|^2 / (sum w)^2 = A^2 / 2 für jedes Fenster, wenn die Linie genau auf einem Bin liegt
    for name in ("Rechteck", "Hann", "Hamming", "Blackman"):
        w = W.window(name, 64)
        x = np.cos(2 * np.pi * 8 * np.arange(64) / 64)
        assert W.spectrum(x, w, 1.0, 1, "Linienhöhe")[8] == pytest.approx(0.5, rel=1e-9), name


def test_padding_keeps_the_original_bins():
    x = G.SplitMix64(4).normal(64)
    w = W.window("Hann", 64)
    s1 = W.spectrum(x, w, 1.0, 1)
    s4 = W.spectrum(x, w, 1.0, 4)
    assert len(s4) == 64 * 4 // 2 + 1 and np.allclose(s4[::4], s1)


def test_expected_spectrum_matches_the_double_sum_definition():
    n, fs = 10, 4.0
    w = W.window("Hann", n)
    r = G.ar_autocov([0.6, -0.2], 1.3, n)
    s = np.cos(2 * np.pi * 0.7 * np.arange(n) / fs + 0.4)
    f = np.array([0.0, 0.3, 1.0, 2.0])
    got = W.expected_spectrum(f, fs, w, r, s)
    for fk, g in zip(f, got):
        om = 2 * np.pi * fk / fs
        noise = sum(w[i] * w[j] * r[abs(i - j)] * np.cos(om * (i - j)) for i in range(n) for j in range(n))
        det = abs(sum(w[i] * s[i] * np.exp(-1j * om * i) for i in range(n))) ** 2
        assert g == pytest.approx((noise + det) / (fs * (w ** 2).sum()), rel=1e-10)


def test_count_peaks_by_hand():
    assert W.count_peaks([0, 1, 0, 2, 0], 0.1) == 2
    assert W.count_peaks([3, 2, 1], 0.1) == 0 and W.count_peaks([1, 2, 3], 0.1) == 0       # Ränder zählen nicht
    assert W.count_peaks([0, 1], 0.1) == 0 and W.count_peaks([], 0.1) == 0
    shoulder = [0, 1, 0.95, 1.2, 0]                                                          # Schulter: Prominenz 1 - 0,95 = 0,05 gegen 1,2
    assert W.count_peaks(shoulder, 0.1) == 1 and W.count_peaks(shoulder, 0.03) == 2
    assert W.count_peaks([0, 5, 4.9, 5.0, 0], 0.1) == 2                                      # zwei Gipfel, beide mit voller Höhe über dem Rand


def test_count_peaks_second_peak_counts_when_above_threshold():
    assert W.count_peaks([0, 5, 0, 1, 0], 0.1) == 2
    assert W.count_peaks([0, 5, 0, 1, 0], 0.25) == 1


def test_two_lines_resolved_on_synthetic_spectra():
    pad = 4
    bins = np.arange(0, 40, 1 / pad)
    two = np.exp(-((bins - 10) / 0.3) ** 2) + np.exp(-((bins - 11.5) / 0.3) ** 2)
    one = np.exp(-((bins - 10.7) / 0.9) ** 2)
    assert W.two_lines_resolved(two, pad, 10.0, 11.5)
    assert not W.two_lines_resolved(one, pad, 10.0, 11.5)


def test_gibbs_overshoot_matches_the_closed_form():
    theory = 2 / math.pi * 1.8519370519824662 - 1
    assert W.gibbs_overshoot(999) == pytest.approx(theory, abs=2e-3)
    assert W.gibbs_overshoot(49) == pytest.approx(theory, abs=5e-3)
    assert W.gibbs_overshoot(49, fejer=True) < 0.0
