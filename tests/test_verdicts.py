"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Zahl im Text und jede Schwelle der Art steht hier."""

import numpy as np

import fnl_evaluation as ev


def _analysis(resolved=True, weak=True, window="Hann", gap=1.5, rel=0, main=1.44, side=-31.5):
    s = ev.Settings(window=window, gap=gap, rel=rel)
    z = np.zeros(3)
    return ev.Analysis(s, z, z, z, z, z, z, 1.5, side, main, -1.42, resolved, weak, {}, {}, 0.0, 0.01)


def test_resolution_verdict_for_resolved_and_merged_lines():
    kind, text = ev.verdict_resolution(_analysis(resolved=True, gap=2.5, main=1.44, side=-31.5))
    assert kind == "success"
    assert "Abstand von 2,50 Bins" in text and "„Hann“ ist 1,44 Bins breit" in text and "-31,5 dB" in text
    kind, text = ev.verdict_resolution(_analysis(resolved=False, gap=1.5, window="Blackman", main=1.66))
    assert kind == "warning" and "Abstand von 1,50 Bins" in text and "**verschmelzen**" in text and "„Blackman“ ist 1,66 Bins breit" in text


def test_weak_line_verdict():
    kind, text = ev.verdict_weak_line(_analysis(weak=True, gap=10.5, rel=-55, window="Hann", side=-31.5))
    assert kind == "success" and "55 dB unter der starken, 10,50 Bins" in text and "„Hann“" in text and "-31,5 dB" in text
    kind, text = ev.verdict_weak_line(_analysis(weak=False, gap=10.5, rel=-55, window="Rechteck", side=-13.3))
    assert kind == "warning" and "nicht zu sehen" in text and "„Rechteck“" in text and "-13,3 dB" in text


def test_table_verdict_names_rectangle_and_hann_numbers():
    rows = [{"window": "Rechteck", "main_lobe": 0.89, "sidelobe_db": -13.3, "scallop_db": -3.92}, {"window": "Hann", "main_lobe": 1.44, "sidelobe_db": -31.5, "scallop_db": -1.42}]
    kind, text = ev.verdict_table(rows)
    assert kind == "info"
    assert "(0,89 Bins)" in text and "(-13,3 dB)" in text and "3,9 dB" in text and "1,44 Bins, -31,5 dB" in text and "1,4 dB" in text


def test_valley_verdict_uses_the_shortest_length():
    rows = [{"n": 512, "window": "Rechteck", "bias_db": 11.9}, {"n": 512, "window": "Hann", "bias_db": 0.0}, {"n": 128, "window": "Rechteck", "bias_db": 17.9},
            {"n": 128, "window": "Hann", "bias_db": 0.0}]
    kind, text = ev.verdict_valley(rows)
    assert kind == "success" and "n = 128" in text and "17,9 dB" in text and "um 0,0 dB" in text


def test_resolution_sweep_verdict_takes_the_row_at_one_and_a_half_bins():
    rows = [{"delta": 1.0, "Rechteck": 0.5, "Hann": 0.0, "Blackman": 0.0}, {"delta": 1.5, "Rechteck": 1.0, "Hann": 0.48, "Blackman": 0.4}]
    kind, text = ev.verdict_resolution_sweep(rows)
    assert kind == "warning" and "Rechteck in 100 %" in text and "Hann in 48 %" in text and "Blackman in 40 %" in text


def test_padding_verdict_names_the_three_factors():
    rows = [{"delta": 1.0, 1: 0.0, 2: 0.5, 4: 0.54, 8: 0.54, 16: 0.54}]
    kind, text = ev.verdict_padding(rows)
    assert kind == "info" and "von 0 % ohne Auffüllen" in text and "54 % bei vierfachem" in text and "54 % bei 16-fachem" in text


def test_weak_sweep_verdict_takes_eight_bins():
    rows = [{"distance": 3.0, "Rechteck": 0.7, "Hann": 1.0}, {"distance": 8.0, "Rechteck": 0.91, "Hann": 1.0}]
    kind, text = ev.verdict_weak(rows)
    assert kind == "info" and "mit dem Rechteck in 91 %" in text and "mit Hann in 100 %" in text


def test_estimator_verdict_uses_the_longest_record_and_highest_snr():
    rmse = [{"n": 64, "snr": 15, "Spitze": 20.0, "Quinn 1": 2.0, "Jacobsen": 2.1, "ML (fein)": 1.2},
            {"n": 1024, "snr": 15, "Spitze": 93.5, "Quinn 1": 1.27, "Jacobsen": 1.55, "ML (fein)": 1.02}]
    bias = [{"estimator": "Spitze", "max": 0.5, "mean": 0.25}]
    kind, text = ev.verdict_estimators(rmse, bias)
    assert kind == "success"
    assert "bis 0,50 Bin" in text and "94-fache" in text and "1,3- bzw. 1,6-fach" in text and "1,02-fach" in text


def test_gibbs_verdict_takes_the_last_row_and_clips_negative_fejer_values():
    rows = [{"m": 9, "rect": 0.182, "fejer": -0.066}, {"m": 999, "rect": 0.179, "fejer": -0.001}]
    kind, text = ev.verdict_gibbs(rows)
    assert kind == "info" and "17,9 %" in text and "999 Oberschwingungen" in text and "bei 0,0 %" in text


def test_decimal_comma_helper():
    assert ev.de(3.14159, 2) == "3,14" and ev.de(-2.5, 1) == "-2,5" and ev.de(7.0, 0) == "7"
