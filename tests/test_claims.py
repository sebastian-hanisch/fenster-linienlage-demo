"""Jede Zahl aus README.md und den App-Texten wird hier über dieselben Auswertungsfunktionen nachgerechnet (DEMO-PLAYBOOK §4).

Alle Experimente laufen über feste Zufallsströme (C.SWEEP_SEED und folgende), nie über einen einzelnen Lauf. Toleranzen sind größer als der Standardfehler der
jeweiligen Läufe (150 Läufe: etwa 4 Prozentpunkte), damit Bibliotheksversionen (die CI installiert ungepinnt) nichts kippen.
"""

import math

import pytest

import fnl_constants as C
import fnl_evaluation as ev


@pytest.fixture(scope="module")
def table():
    return {r["window"]: r for r in ev.table_windows()}


@pytest.fixture(scope="module")
def valley():
    return {(r["n"], r["window"]): r["bias_db"] for r in ev.valley_bias()}


@pytest.fixture(scope="module")
def resolution():
    return {r["delta"]: r for r in ev.sweep_resolution()}


@pytest.fixture(scope="module")
def padding():
    return {r["delta"]: r for r in ev.sweep_padding()}


@pytest.fixture(scope="module")
def weak():
    return {r["distance"]: r for r in ev.sweep_weak()}


@pytest.fixture(scope="module")
def bias():
    return {r["estimator"]: r for r in ev.estimator_bias()}


@pytest.fixture(scope="module")
def rmse():
    return {(r["n"], r["snr"]): r for r in ev.estimator_rmse()}


def test_experiment_streams_are_the_documented_ones():
    assert C.SWEEP_SEED == 200000 and C.EXP_RES_RUNS == 150 and C.EXP_WEAK_RUNS == 100 and C.EXP_EST_RUNS == 300 and C.EXP_TABLE_N == 256


@pytest.mark.parametrize("name,enbw,lobe,side,scallop", [
    ("Rechteck", 1.00, 0.89, -13.3, -3.92), ("Hann", 1.50, 1.44, -31.5, -1.42), ("Hamming", 1.36, 1.30, -42.7, -1.75),
    ("Blackman", 1.73, 1.64, -58.1, -1.10), ("Kaiser", 1.67, 1.59, -58.7, -1.18), ("Flat-Top", 3.77, 3.73, -92.7, -0.01)])
def test_window_figures_in_the_readme(table, name, enbw, lobe, side, scallop):
    r = table[name]
    assert r["enbw"] == pytest.approx(enbw, abs=0.01)
    assert r["main_lobe"] == pytest.approx(lobe, abs=0.02)
    assert r["sidelobe_db"] == pytest.approx(side, abs=0.2)
    assert r["scallop_db"] == pytest.approx(scallop, abs=0.02)


def test_normalisation_offset_equals_the_bandwidth_in_db(table):
    for name, r in table.items():
        assert r["line_vs_density_db"] == pytest.approx(10 * math.log10(r["enbw"]))
    assert table["Hann"]["line_vs_density_db"] == pytest.approx(1.76, abs=0.01)
    assert table["Blackman"]["line_vs_density_db"] == pytest.approx(2.37, abs=0.01)
    assert table["Flat-Top"]["line_vs_density_db"] == pytest.approx(5.76, abs=0.01)


@pytest.mark.parametrize("n,rect,hamming", [(128, 17.9, 3.0), (512, 11.9, 0.9), (2048, 6.7, 0.2)])
def test_valley_bias_by_window(valley, n, rect, hamming):
    assert valley[(n, "Rechteck")] == pytest.approx(rect, abs=0.1)
    assert valley[(n, "Hamming")] == pytest.approx(hamming, abs=0.1)
    for name in ("Hann", "Blackman", "Kaiser", "Flat-Top"):
        assert abs(valley[(n, name)]) < 0.1


@pytest.mark.parametrize("delta,rect,hann,blackman", [(1.0, 53, 1, 0), (1.5, 100, 48, 40), (2.0, 100, 77, 61), (2.5, 100, 100, 100)])
def test_resolution_by_window(resolution, delta, rect, hann, blackman):
    r = resolution[delta]
    assert 100 * r["Rechteck"] == pytest.approx(rect, abs=5)
    assert 100 * r["Hann"] == pytest.approx(hann, abs=7)
    assert 100 * r["Blackman"] == pytest.approx(blackman, abs=7)


def test_padding_gains_nothing_beyond_the_window_limit(padding):
    r = padding[1.0]
    assert r[1] < 0.1
    assert r[2] == pytest.approx(r[16], abs=0.1) and r[4] == pytest.approx(r[16], abs=0.03) and r[8] == pytest.approx(r[16], abs=0.03)
    assert 100 * r[16] == pytest.approx(54, abs=5)
    assert padding[1.5][1] < 0.6 and padding[1.5][4] == 1.0


@pytest.mark.parametrize("distance,rect", [(3.0, 71), (8.0, 91), (12.0, 98)])
def test_weak_line_with_the_rectangle(weak, distance, rect):
    assert 100 * weak[distance]["Rechteck"] == pytest.approx(rect, abs=8)
    for name in ("Hann", "Hamming", "Blackman", "Kaiser"):
        assert weak[distance][name] == 1.0


def test_flat_top_swallows_a_weak_line_at_three_bins(weak):
    assert weak[3.0]["Flat-Top"] == 0.0 and weak[5.0]["Flat-Top"] == 1.0


def test_estimator_bias_without_noise(bias):
    assert bias["Spitze"]["max"] == pytest.approx(0.5, abs=1e-9)
    assert bias["Parabel"]["max"] == pytest.approx(0.17, abs=0.01)
    assert bias["Quinn 1"]["max"] == pytest.approx(0.003, abs=0.001) and bias["Rife"]["max"] == pytest.approx(0.003, abs=0.001)
    assert bias["Jacobsen"]["max"] < 0.001 and bias["Candan"]["max"] < 0.001


def test_estimator_efficiency_against_the_bound(rmse):
    for key in ((256, 15), (1024, 15)):
        r = rmse[key]
        assert r["Spitze"] > 40 and r["Parabel"] > 15
        assert 3.0 < r["Rife"] < 5.0
        assert 1.2 < r["Quinn 1"] < 1.5 and 1.4 < r["Jacobsen"] < 1.7 and r["Candan"] == pytest.approx(r["Jacobsen"], rel=0.01)
        assert 1.0 <= r["ML (fein)"] < 1.25
    assert 90 < rmse[(1024, 15)]["Spitze"] < 105


def test_cramer_rao_values_in_the_text(rmse):
    assert rmse[(1024, 15)]["crb"] == pytest.approx(0.0031, abs=0.0002)
    assert rmse[(64, 5)]["crb"] == pytest.approx(0.039, abs=0.002)


def test_gibbs_overshoot_in_the_text():
    rows = {r["m"]: r for r in ev.gibbs_table()}
    for m in (49, 199, 999):
        assert 100 * rows[m]["rect"] == pytest.approx(17.9, abs=0.1)
    assert all(r["fejer"] <= 0.0 for r in rows.values())


def test_preset_help_texts_quote_the_documented_numbers():
    assert C.PRESETS["Zwei nahe Linien"]["gap"] == 1.5 and C.PRESETS["Schwache neben starker"]["rel"] == -55 and C.PRESETS["Schwache neben starker"]["gap"] == 10.5
    assert C.PRESETS["Steiler Hintergrund"]["n"] == 128
    assert C.BACKGROUNDS["Steil"] and C.EXP_WEAK_REL_DB == -40
