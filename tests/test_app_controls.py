"""AppTest: Auswahlfelder, Permalink, Kaiser-Regler, PDF (die Rauchtests in test_app.py decken nur Knöpfe und Slider ab)."""

import os

import pytest
from streamlit.testing.v1 import AppTest

import fnl_constants as C

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = 240


def _run(params=None):
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    if params:
        at.query_params.update(params)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


@pytest.mark.parametrize("window", C.WINDOWS)
def test_every_window_runs(window):
    at = _run()
    at.selectbox(key="window_select").set_value(window).run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert ("Kaiser-Parameter β" in [s.label for s in at.slider]) == (window == "Kaiser")


@pytest.mark.parametrize("pad", C.PAD_CHOICES)
def test_every_padding_factor_runs(pad):
    at = _run()
    at.select_slider(key="pad_select").set_value(pad).run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]


def test_norm_and_background_switches_run():
    at = _run()
    at.selectbox(key="norm_select").set_value("Linienhöhe").run(timeout=TIMEOUT)
    at.selectbox(key="background_select").set_value("Steil").run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]


def test_kaiser_beta_value_survives_switching_the_window_away_and_back():
    at = _run({"w": "Kaiser", "beta": "11.5"})
    beta = [s for s in at.slider if s.label == "Kaiser-Parameter β"][0]
    assert beta.value == 11.5
    at.selectbox(key="window_select").set_value("Hann").run(timeout=TIMEOUT)
    assert "Kaiser-Parameter β" not in [s.label for s in at.slider]
    at.selectbox(key="window_select").set_value("Kaiser").run(timeout=TIMEOUT)
    assert [s for s in at.slider if s.label == "Kaiser-Parameter β"][0].value == 11.5


def test_permalink_is_applied_and_clipped():
    at = _run({"n": "99999", "pad": "4", "w": "Blackman", "f": "9", "d": "0", "rel": "-999", "snr": "999", "bg": "Steil", "norm": "Linienhöhe", "seed": "42"})
    assert at.slider(key="n_slider").value == C.N_MAX
    assert at.select_slider(key="pad_select").value == 4
    assert at.selectbox(key="window_select").value == "Blackman"
    assert at.slider(key="line_slider").value == C.LINE_MAX or at.slider(key="line_slider").value == C.LINE_MIN
    assert at.slider(key="gap_slider").value == C.GAP_MIN
    assert at.slider(key="rel_slider").value == C.REL_MIN and at.slider(key="snr_slider").value == C.SNR_MAX
    assert at.selectbox(key="background_select").value == "Steil" and at.selectbox(key="norm_select").value == "Linienhöhe"


def test_permalink_ignores_unknown_choices():
    at = _run({"w": "Gauß", "pad": "3", "bg": "unsinn", "norm": "unsinn"})
    assert at.selectbox(key="window_select").value == C.DEFAULT_WINDOW
    assert at.select_slider(key="pad_select").value == C.DEFAULT_PAD
    assert at.selectbox(key="background_select").value == C.DEFAULT_BACKGROUND and at.selectbox(key="norm_select").value == C.DEFAULT_NORM


def test_preset_button_sets_the_widgets():
    at = _run()
    at.button(key="preset_Schwache neben starker").click().run(timeout=TIMEOUT)
    assert not at.exception
    assert at.selectbox(key="window_select").value == "Rechteck" and at.slider(key="rel_slider").value == -55


def test_pdf_button_builds_a_download():
    at = _run()
    at.button(key="pdf_make").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
