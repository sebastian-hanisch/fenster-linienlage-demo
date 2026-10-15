"""Orakel: Kernroutinen gegen einen unabhängigen Rechenweg (scipy, Aufzählung, geschlossene Formeln; DEMO-PLAYBOOK §12)."""

import math

import numpy as np
import pytest

import fnl_constants as C
import fnl_evaluation as ev
import fnl_windows as W
import signal_generator as G

sps = pytest.importorskip("scipy.signal")
spw = pytest.importorskip("scipy.signal.windows")
spe = pytest.importorskip("scipy.special")


@pytest.mark.parametrize("n", [16, 17, 100, 256])
def test_windows_equal_scipy_periodic_windows(n):
    ref = {"Rechteck": "boxcar", "Hann": "hann", "Hamming": "hamming", "Blackman": "blackman", "Flat-Top": "flattop"}
    for name, sname in ref.items():
        assert np.allclose(W.window(name, n), sps.get_window(sname, n, fftbins=True), atol=1e-7), name
    for beta in (0.0, 4.0, 8.0, 14.0):
        assert np.allclose(W.window("Kaiser", n, beta), spw.kaiser(n, beta, sym=False), atol=1e-12), beta


def test_windowed_spectrum_equals_scipy_periodogram():
    x = G.SplitMix64(5).normal(100)
    for name in ("Hann", "Blackman", "Rechteck"):
        w = W.window(name, 100)
        f, p = sps.periodogram(x, fs=250.0, window=w, detrend=False, return_onesided=False, scaling="density")
        assert np.allclose(W.spectrum(x, w, 250.0, 1), p[:51], rtol=1e-10, atol=1e-15), name


def test_expected_windowed_spectrum_agrees_with_simulation_for_a_steep_background():
    poles, sg, wh = ev.noise_spec("Steil")
    n, runs = 128, 1500
    x = G.make_noise_batch(runs, n, poles, sg, wh, G.SplitMix64(33))
    r = G.noise_autocov(poles, sg, wh, n)
    ks = [6, 20, 32, 45, 58, 63]
    for name in ("Rechteck", "Hann", "Hamming", "Blackman"):
        w = W.window(name, n)
        I = np.abs(np.fft.rfft(x * w[None, :], axis=1)) ** 2 / (C.FS * (w ** 2).sum())
        e = W.expected_spectrum(np.array(ks) * C.FS / n, C.FS, w, r)
        for k, ek in zip(ks, e):
            z = (I[:, k].mean() - ek) / (I[:, k].std() / math.sqrt(runs))
            assert abs(z) < 4.0, (name, k, z)


def test_hann_removes_the_valley_bias_that_the_rectangle_has():
    rows = {(r["n"], r["window"]): r["bias_db"] for r in ev.valley_bias(ns=(128,))}
    assert rows[(128, "Rechteck")] > 15.0 and abs(rows[(128, "Hann")]) < 0.1


def test_gibbs_overshoot_equals_the_sine_integral_value():
    assert W.gibbs_overshoot(999) == pytest.approx(2 / math.pi * spe.sici(math.pi)[0] - 1, abs=2e-3)


def test_peak_counting_equals_scipy_find_peaks_on_random_arrays():
    rng = G.SplitMix64(8)
    for trial in range(300):
        y = rng.uniform(int(8 + trial % 25))
        ref = float(y.max())
        for rel in (0.0, 0.05, 0.2, 0.5):
            expected = len(sps.find_peaks(y, prominence=rel * ref)[0]) if rel > 0 else len(sps.find_peaks(y)[0])
            got = W.count_peaks(y, rel, reference=ref)
            if rel == 0.0:
                continue                                        # scipy gibt bei prominence 0 alle Gipfel, wir zählen mit Schwelle >= 0: gleichwertig
            assert got == expected, (trial, rel)


def test_resolved_criterion_agrees_with_find_peaks_on_a_two_line_signal():
    n, pad = 256, 16
    t = np.arange(n)
    rng = G.SplitMix64(12)
    for d in (1.0, 1.5, 2.5):
        for _ in range(10):
            ph = 2 * math.pi * rng.uniform(2)
            f1 = 40.0 + rng.uniform(1)[0]
            f2 = f1 + d
            x = np.cos(2 * np.pi * f1 * t / n + ph[0]) + np.cos(2 * np.pi * f2 * t / n + ph[1])
            sp = np.abs(np.fft.rfft(x * W.window("Hann", n), n * pad)) ** 2
            lo, hi = int((f1 - d / 2) * pad), int((f2 + d / 2) * pad) + 1
            ref = len(sps.find_peaks(sp[lo:hi], prominence=0.1 * sp.max())[0]) >= 2
            assert W.two_lines_resolved(sp, pad, f1, f2) == ref
