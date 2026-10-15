"""Schätzer: Mini-Instanz von Hand, Bias ohne Rauschen, Schranke, feines Maximum."""

import math

import numpy as np
import pytest

import fnl_estimators as E
import signal_generator as G


def complex_tone(n, f_bins):
    return np.fft.fft(np.exp(2j * np.pi * f_bins * np.arange(n) / n))[None, :]


def test_mini_instance_complex_tone_at_one_and_a_quarter_bins():
    # N = 8, Ton bei 1,25 Bins: Beträge 1,5; 7,214; 2,436 an den Bins 0, 1, 2; wahr 0,25 neben dem Spitzen-Bin 1
    X = complex_tone(8, 1.25)
    k = E.peak_bins(X, 1, 4)
    assert k.tolist() == [1]
    assert np.abs(X[0, :3]).round(3).tolist() == [1.5, 7.214, 2.436]
    xm, x0, xp = E.three_bins(X, k)
    assert E.est_jacobsen(xm, x0, xp, 8)[0] == pytest.approx(0.2378, abs=1e-4)
    assert E.est_quinn1(xm, x0, xp, 8)[0] == pytest.approx(0.2378, abs=1e-4)
    assert E.est_rife(xm, x0, xp, 8)[0] == pytest.approx(0.2524, abs=1e-4)
    assert E.est_candan(xm, x0, xp, 8)[0] == pytest.approx(0.2508, abs=1e-4)
    assert E.est_parabola(xm, x0, xp, 8)[0] == pytest.approx(0.0913, abs=1e-4)
    assert E.est_peak(xm, x0, xp, 8)[0] == 0.0


def test_candan_equals_jacobsen_times_the_length_factor():
    X = complex_tone(16, 3.3)
    k = E.peak_bins(X, 1, 8)
    xm, x0, xp = E.three_bins(X, k)
    assert E.est_candan(xm, x0, xp, 16)[0] == pytest.approx(math.tan(math.pi / 16) / (math.pi / 16) * E.est_jacobsen(xm, x0, xp, 16)[0])


def test_estimate_all_returns_bin_plus_delta_for_every_estimator():
    est = E.estimate_all(complex_tone(64, 20.3), 2, 30)
    assert set(est) == set(E.NAMES)
    assert est["Spitze"][0] == 20.0
    assert est["Candan"][0] == pytest.approx(20.3, abs=1e-3) and est["Jacobsen"][0] == pytest.approx(20.3, abs=0.05)


def test_noise_free_bias_ordering_for_a_real_line():
    n = 256
    deltas = np.linspace(-0.5, 0.5, 51)
    x = np.cos(2 * np.pi * (60.0 + deltas)[:, None] * np.arange(n)[None, :] / n + 0.3)
    est = E.estimate_all(np.fft.fft(x, axis=1)[:, : n // 2 + 1], 2, n // 2 - 2)
    worst = {k: float(np.max(np.abs(v - (60.0 + deltas)))) for k, v in est.items()}
    assert worst["Spitze"] == pytest.approx(0.5, abs=1e-9)
    assert worst["Parabel"] == pytest.approx(0.17, abs=0.02)
    assert worst["Quinn 1"] < 0.01 and worst["Rife"] < 0.01 and worst["Jacobsen"] < 0.001 and worst["Candan"] < 0.001
    assert worst["Spitze"] > worst["Parabel"] > worst["Quinn 1"]


def test_crb_by_hand():
    # n = 8, SNR = 1 (0 dB): sqrt(3 / (pi^2 * 8 * 63)) * 8
    assert E.crb_bins(8, 0.0) == pytest.approx(math.sqrt(3 / (math.pi ** 2 * 8 * 63)) * 8)
    assert E.crb_bins(256, 10.0) == pytest.approx(E.crb_bins(256, 0.0) / math.sqrt(10.0))
    assert E.crb_bins(512, 0.0) < E.crb_bins(256, 0.0)


def test_refined_maximum_is_exact_for_a_noise_free_tone():
    n = 128
    for f in (20.0, 20.3, 33.77, 50.5):
        x = np.cos(2 * np.pi * f * np.arange(n) / n + 0.7)
        assert E.refined_maximum(x, 2, n // 2 - 2)[0] == pytest.approx(f, abs=0.02)


def test_maximum_estimator_reaches_the_bound_at_high_snr():
    n, runs, snr_db = 256, 300, 15
    rng = G.SplitMix64(17)
    f = 30 + 20 * rng.uniform(runs)
    ph = 2 * np.pi * rng.uniform(runs)
    amp = math.sqrt(2 * 10 ** (snr_db / 10))
    x = amp * np.cos(2 * np.pi * f[:, None] * np.arange(n)[None, :] / n + ph[:, None]) + rng.normal(runs * n).reshape(runs, n)
    err = E.refined_maximum(x, 2, n // 2 - 2) - f
    ratio = math.sqrt(np.mean(err ** 2)) / E.crb_bins(n, snr_db)
    assert 0.85 < ratio < 1.25
