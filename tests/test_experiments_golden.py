"""Experimente und Analyse: feste Zufallsströme, festgeschriebene Werte (Regression) und eine unabhängige Nachrechnung der Analyse.

Die Festwerte stammen aus einem Lauf mit den festen Strömen (SplitMix64, FFT); Toleranz 1e-6 relativ gegen Plattformunterschiede der C-Bibliothek. Sie fangen
Änderungen an Vorzeichen, Fenstergrenzen, Läufen und Kanten der Experimentfunktionen, die die weiten Toleranzen der Befund-Tests (`test_claims.py`) nicht sehen."""

import numpy as np
import pytest

import fnl_constants as C
import fnl_evaluation as ev
import fnl_windows as W


def approx(x):
    return pytest.approx(x, rel=1e-6, abs=1e-9)


def test_resolution_sweep_fixed_values():
    rows = ev.sweep_resolution(windows=("Rechteck", "Hann"), deltas=(1.25, 1.5), runs=12)
    assert rows == [{"delta": 1.25, "runs": 12, "Rechteck": approx(10 / 12), "Hann": approx(6 / 12)},
                    {"delta": 1.5, "runs": 12, "Rechteck": approx(1.0), "Hann": approx(8 / 12)}]


def test_padding_sweep_fixed_values():
    rows = ev.sweep_padding(deltas=(1.25,), pads=(1, 2), runs=12)
    assert rows == [{"delta": 1.25, "runs": 12, 1: approx(1 / 12), 2: approx(8 / 12)}]


def test_weak_line_sweep_fixed_values():
    rows = ev.sweep_weak(windows=("Rechteck", "Hann"), distances=(3.0,), runs=12)
    assert rows == [{"distance": 3.0, "runs": 12, "Rechteck": approx(7 / 12), "Hann": approx(1.0)}]


def test_estimator_rmse_fixed_values():
    r = ev.estimator_rmse(ns=(64,), snrs=(5,), runs=20)[0]
    assert r == {"n": 64, "snr": 5, "crb": approx(0.03875910933380446), "runs": 20, "Spitze": approx(8.204243014151425), "Parabel": approx(4.317011418345093),
                 "Quinn 1": approx(0.8940838192478329), "Jacobsen": approx(1.6651982296617132), "Rife": approx(2.192631140366209),
                 "Candan": approx(1.6598728961616078), "ML (fein)": approx(0.694859866519545), "outliers": 0.0}


def test_estimator_bias_fixed_values_for_a_short_record():
    rows = {r["estimator"]: r for r in ev.estimator_bias(n=64, offsets=5)}
    assert rows["Spitze"]["max"] == approx(0.5) and rows["Spitze"]["mean"] == approx(0.3)
    assert rows["Parabel"]["max"] == approx(0.17088910797724566) and rows["Quinn 1"]["max"] == approx(0.008399918333612533)
    assert rows["Jacobsen"]["max"] == approx(0.0010531101190913716) and rows["Rife"]["mean"] == approx(0.005782610445796976)
    assert rows["Candan"]["max"] == approx(0.0008892397689397313)


def test_valley_bias_fixed_values():
    rows = {r["window"]: r["bias_db"] for r in ev.valley_bias(ns=(128,))}
    assert rows == {"Rechteck": approx(17.880333227029833), "Hann": approx(0.004451240852701004), "Hamming": approx(2.950258056721424),
                    "Blackman": approx(0.006031066015493575), "Kaiser": approx(0.010351582793568107), "Flat-Top": approx(0.02102720570557666)}


def test_gibbs_table_fixed_values():
    rows = ev.gibbs_table(ms=(9, 49))
    assert rows == [{"m": 9, "rect": approx(0.18232820885760526), "fejer": approx(-0.06426998537717399)},
                    {"m": 49, "rect": approx(0.17911310188829455), "fejer": approx(-0.012737478268876279)}]


def test_window_table_fixed_values():
    r = ev.table_windows(n=64)[0]
    assert r == {"window": "Rechteck", "enbw": approx(1.0), "main_lobe": approx(0.8859129545857033), "sidelobe_db": approx(-13.256466474732289),
                 "scallop_db": approx(-3.921525470127501), "line_vs_density_db": approx(0.0)}


@pytest.mark.parametrize("window", ["Rechteck", "Hann", "Blackman"])
def test_analysis_matches_an_independent_recomputation(window):
    s = ev.Settings(window=window, n=128, pad=4, line=0.2, gap=3.0, rel=-30, snr=20, seed=9)
    a = ev.analyse(s)
    sig, lines, _ = ev.build_signal(s)
    w = W.window(window, 128)
    X = np.fft.rfft(sig.x * w, 128 * 4)
    ref = np.abs(X) ** 2 / (C.FS * (w ** 2).sum())
    assert np.allclose(a.spec, ref, rtol=1e-10)
    assert a.f_bins[4] == pytest.approx(1.0) and len(a.f_bins) == 128 * 4 // 2 + 1
    assert a.enbw == pytest.approx(128 * (w ** 2).sum() / w.sum() ** 2)
    assert s.f1_bins == pytest.approx(25.6) and s.f2_bins == pytest.approx(28.6)


def test_second_line_goes_below_the_first_when_it_would_pass_the_nyquist_bin():
    s = ev.Settings(n=128, line=0.45, gap=20.0)
    assert s.f1_bins == pytest.approx(57.6) and s.f2_bins == pytest.approx(37.6)
    assert ev.Settings(n=128, line=0.2, gap=20.0).f2_bins == pytest.approx(45.6)


def test_a_strong_hann_resolved_pair_is_found_and_a_buried_weak_line_is_not_visible_with_the_rectangle():
    close = C.PRESETS["Zwei nahe Linien"]
    assert not ev.analyse(ev.Settings(**close)).resolved
    assert ev.analyse(ev.Settings(**{**close, "window": "Rechteck"})).resolved
    weak = C.PRESETS["Schwache neben starker"]
    assert not ev.analyse(ev.Settings(**weak)).weak_visible
    assert ev.analyse(ev.Settings(**{**weak, "window": "Hann"})).weak_visible
