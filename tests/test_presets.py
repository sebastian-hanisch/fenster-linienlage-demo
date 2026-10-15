"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen; jedes Preset zeigt seinen Effekt (feste Seeds)."""

import numpy as np
import pytest

import fnl_constants as C
import fnl_evaluation as ev
import fnl_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_bounds():
    for name, p in C.PRESETS.items():
        assert C.N_MIN <= p["n"] <= C.N_MAX and p["n"] % C.N_STEP == 0 and p["pad"] in C.PAD_CHOICES, name
        assert C.LINE_MIN <= p["line"] <= C.LINE_MAX and C.GAP_MIN <= p["gap"] <= C.GAP_MAX and C.REL_MIN <= p["rel"] <= C.REL_MAX
        assert C.SNR_MIN <= p["snr"] <= C.SNR_MAX and C.BETA_MIN <= p["beta"] <= C.BETA_MAX and 0 <= p["seed"] <= C.SEED_MAX
        assert p["window"] in C.WINDOWS and p["background"] in C.BACKGROUNDS and p["norm"] in C.NORMS


def test_default_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS["Zwei nahe Linien"]) == ev.Settings()


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_preset_names_are_short_enough_for_the_buttons():
    assert max(len(n) for n in C.PRESETS) <= 24


def test_close_lines_merge_with_hann_and_separate_with_the_rectangle():
    p = C.PRESETS["Zwei nahe Linien"]
    assert not ev.analyse(ev.Settings(**p)).resolved
    assert ev.analyse(ev.Settings(**{**p, "window": "Rechteck"})).resolved


def test_weak_line_is_hidden_by_the_rectangle_and_visible_with_blackman():
    p = C.PRESETS["Schwache neben starker"]
    assert not ev.analyse(ev.Settings(**p)).weak_visible
    assert ev.analyse(ev.Settings(**{**p, "window": "Blackman"})).weak_visible


def test_between_bins_preset_shows_the_peak_error_and_the_interpolators_removing_it():
    a = ev.analyse(ev.Settings(**C.PRESETS["Linie zwischen Bins"]))
    assert abs(a.errors["Spitze"]) > 0.45
    assert abs(a.errors["Jacobsen"]) < 0.01 and abs(a.errors["Candan"]) < 0.01
    frac = a.settings.f1_bins - np.floor(a.settings.f1_bins)
    assert 0.45 < frac < 0.55


def test_steep_background_preset_shows_the_valley_bias_of_the_rectangle():
    p = C.PRESETS["Steiler Hintergrund"]
    rect = ev.analyse(ev.Settings(**p))
    hann = ev.analyse(ev.Settings(**{**p, "window": "Hann"}))
    top = rect.f_bins > 0.9 * rect.settings.n / 2 * 0.9
    gap = lambda a: 10 * np.log10(a.expected[top].mean() / a.true_density[top].mean())
    assert gap(rect) > 5.0 and abs(gap(hann)) < 1.0
