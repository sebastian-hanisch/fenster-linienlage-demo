"""Permalink (Begrenzen), Presets und Zufalls-Seed-Button (Standardmuster aus dem Demo-Portfolio).

Alle Regler sind in jeder Einstellung wirksam. Einzige Ausnahme: der Kaiser-Parameter β wirkt nur beim Kaiser-Fenster und wird deshalb nur dann gezeichnet.
Sein Wert steht nicht im Zustand eines Reglers, sondern unter `beta_kept` (kein Frontend/Backend-Auseinanderlaufen: der Regler bekommt ihn als Startwert).
"""

import random
from dataclasses import dataclass

import streamlit as st

import fnl_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: type
    default: object
    lo: float | None = None
    hi: float | None = None
    choices: tuple | None = None


SETTING_SPECS = {
    "n_slider": SettingSpec("n", int, C.DEFAULT_N, C.N_MIN, C.N_MAX),
    "pad_select": SettingSpec("pad", int, C.DEFAULT_PAD, choices=C.PAD_CHOICES),
    "window_select": SettingSpec("w", str, C.DEFAULT_WINDOW, choices=C.WINDOWS),
    "beta_kept": SettingSpec("beta", float, C.DEFAULT_BETA, C.BETA_MIN, C.BETA_MAX),
    "line_slider": SettingSpec("f", float, C.DEFAULT_LINE, C.LINE_MIN, C.LINE_MAX),
    "gap_slider": SettingSpec("d", float, C.DEFAULT_GAP, C.GAP_MIN, C.GAP_MAX),
    "rel_slider": SettingSpec("rel", int, C.DEFAULT_REL, C.REL_MIN, C.REL_MAX),
    "snr_slider": SettingSpec("snr", int, C.DEFAULT_SNR, C.SNR_MIN, C.SNR_MAX),
    "background_select": SettingSpec("bg", str, C.DEFAULT_BACKGROUND, choices=tuple(C.BACKGROUNDS)),
    "norm_select": SettingSpec("norm", str, C.DEFAULT_NORM, choices=C.NORMS),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, C.SEED_MAX),
}
PRESET_KEYS = {"n": "n_slider", "pad": "pad_select", "window": "window_select", "beta": "beta_kept", "line": "line_slider", "gap": "gap_slider",
               "rel": "rel_slider", "snr": "snr_slider", "background": "background_select", "norm": "norm_select", "seed": "seed_input"}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, auf die Reglergrenzen begrenzt bzw. auf gültige Auswahlwerte geprüft."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param not in qp:
            continue
        try:
            value = spec.caster(qp[spec.url_param])
        except (ValueError, TypeError):
            continue
        if spec.choices is not None and value not in spec.choices:
            continue
        if spec.lo is not None:
            value = max(spec.lo, value)
        if spec.hi is not None:
            value = min(spec.hi, value)
        if state_key == "n_slider":
            value = min(C.N_MAX, max(C.N_MIN, int(round(value / C.N_STEP)) * C.N_STEP))
        st.session_state[state_key] = value
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        if key in C.PRESETS[name]:
            st.session_state[state_key] = C.PRESETS[name][key]


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)
