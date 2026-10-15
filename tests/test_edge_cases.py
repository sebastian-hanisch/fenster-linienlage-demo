"""Randfälle: kleinste und größte Länge, alle Fenster, alle Auffüllfaktoren, Linien an den Rändern, beide Normierungen und Hintergründe."""

import numpy as np
import pytest

import fnl_constants as C
import fnl_evaluation as ev
import fnl_windows as W


@pytest.mark.parametrize("window", C.WINDOWS)
@pytest.mark.parametrize("norm", C.NORMS)
def test_every_window_and_norm_runs_at_the_smallest_length(window, norm):
    a = ev.analyse(ev.Settings(window=window, norm=norm, n=C.N_MIN, pad=1))
    assert len(a.spec) == C.N_MIN // 2 + 1 and np.isfinite(a.spec).all() and np.isfinite(a.expected).all() and (a.expected > 0).all()


def test_largest_length_with_the_largest_padding_runs():
    a = ev.analyse(ev.Settings(n=C.N_MAX, pad=max(C.PAD_CHOICES), window="Flat-Top", background="Steil"))
    assert len(a.spec) == C.N_MAX * max(C.PAD_CHOICES) // 2 + 1 and np.isfinite(a.errors["Candan"])


@pytest.mark.parametrize("line", [C.LINE_MIN, C.LINE_MAX])
@pytest.mark.parametrize("gap", [C.GAP_MIN, C.GAP_MAX])
def test_lines_at_the_slider_limits_stay_inside_the_band(line, gap):
    for n in (C.N_MIN, C.N_MAX):
        s = ev.Settings(n=n, line=line, gap=gap)
        assert 1.0 < s.f1_bins < n / 2 - 1 and 1.0 < s.f2_bins <= n / 2 - 2 + 1e-9
        a = ev.analyse(s)
        assert np.isfinite(a.spec).all()


def test_extreme_second_line_levels():
    for rel in (C.REL_MIN, C.REL_MAX):
        a = ev.analyse(ev.Settings(rel=rel))
        assert np.isfinite(a.spec).all()


def test_noise_free_limit_with_the_lowest_snr_and_the_highest():
    for snr in (C.SNR_MIN, C.SNR_MAX):
        a = ev.analyse(ev.Settings(snr=snr, window="Rechteck", pad=1))
        assert np.isfinite(a.errors["Quinn 1"]) and np.isfinite(a.ml)


def test_kaiser_beta_changes_the_sidelobes_monotonically():
    side = [W.sidelobe_peak_db(W.window("Kaiser", 256, b)) for b in (2.0, 6.0, 10.0, 14.0)]
    assert side == sorted(side, reverse=True)
    assert W.main_lobe_3db(W.window("Kaiser", 256, 14.0)) > W.main_lobe_3db(W.window("Kaiser", 256, 2.0))


def test_same_seed_same_analysis_and_different_seed_differs():
    s = ev.Settings(seed=5)
    assert np.array_equal(ev.analyse(s).x, ev.analyse(s).x)
    assert not np.array_equal(ev.analyse(s).x, ev.analyse(ev.Settings(seed=6)).x)


def test_norm_switch_changes_only_the_scale_by_the_bandwidth():
    base = ev.Settings(window="Hann", seed=3)
    dens = ev.analyse(base)
    line = ev.analyse(ev.Settings(window="Hann", seed=3, norm="Linienhöhe"))
    ratio = line.spec / dens.spec
    n = base.n
    assert np.allclose(ratio, ratio[0]) and ratio[0] == pytest.approx(2 * C.FS * W.enbw(dens.w) / n, rel=1e-9)
