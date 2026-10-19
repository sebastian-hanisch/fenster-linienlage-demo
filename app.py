"""Fenster und Linienlage: Das Fenster tauscht Auflösung gegen Leckage, das Gitter ist nur ein Raster - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Stück 2 der Reihe „Spektral- und Wavelet-Analyse“. Ein simuliertes Signal mit zwei Linien (bekannte Lage, bekannte Stärke) wird mit sechs Fenstern
gefenstert; gezeigt werden Auflösung, Leckage, Gipfelverlust und die Schätzung der Linienlage. Aufbau und Regeln: DEMO-PLAYBOOK.md (§2).

Lauffähig mit: streamlit run app.py
"""

import pandas as pd
import streamlit as st

import fnl_constants as C
import fnl_evaluation as ev
from fnl_evaluation import Settings, analyse, de
from fnl_pdf_export import build_pdf
from fnl_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sync_query_params,
)
from fnl_visualization import (
    build_estimators,
    build_padding,
    build_resolution,
    build_spectrum,
    build_valley,
    build_window_response,
    build_window_shape,
)

st.set_page_config(page_title="Fenster und Linienlage – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner=False)
def _experiments(beta):
    return {"table": ev.table_windows(beta=beta), "valley": ev.valley_bias(beta=beta), "resolution": ev.sweep_resolution(beta=beta),
            "padding": ev.sweep_padding(), "weak": ev.sweep_weak(beta=beta), "bias": ev.estimator_bias(), "rmse": ev.estimator_rmse(), "gibbs": ev.gibbs_table()}


st.title("🪟 Fenster und Linienlage: das Fenster tauscht Auflösung gegen Leckage")
st.markdown(
    """
Wer ein Signal endlich lang aufzeichnet, schneidet es ab; das **Fenster** bestimmt, wie. Es legt drei Dinge zugleich fest: wie dicht zwei **Linien** liegen dürfen
(**Hauptkeule**), wie hoch die Täler bleiben (**Nebenkeulen**, Leckage) und wie viel eine Linie zwischen zwei **Bins** verliert (**Gipfelverlust**). **Auffüllen mit Nullen** ändert
an den Fensterkennzahlen nichts. Die Lage einer Linie dagegen lässt sich aus drei Bins auf einen Bruchteil eines Bins schätzen.

Dies ist die zweite Demo der Reihe „Spektral- und Wavelet-Analyse“, auf dem **Periodogramm** aufbauend; „Spectral Clustering“ ist etwas anderes. Alles ist **simuliert**: Lage und Stärke
der Linien und die Dichte des Hintergrunds sind bekannt. Weiter unten: **🔬 Experimente**, **🚧 Wo die Annahmen enden** und **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert das Fenster", expanded=True):
    st.markdown(
        """
1. **Messwerte gewichten:** Jeder Messwert wird mit dem Fensterwert multipliziert, bevor die FFT läuft. Das Rechteck lässt alles unverändert, **Hann**, **Hamming**, **Blackman**,
   **Kaiser** und **Flat-Top** blenden die Ränder sanft aus.
2. **Hauptkeule und Nebenkeulen:** Eine Linie erscheint als Hauptkeule mit Nebenkeulen. Sanftere Fenster haben viel niedrigere Nebenkeulen, aber eine breitere Hauptkeule.
3. **Raster:** Die FFT liefert Werte nur an den Bins; liegt die Linie dazwischen, ist der Gipfel zu niedrig und die Spitze zeigt auf das falsche Bin. **Auffüllen** zeichnet die Kurve glatter,
   gewinnt aber keine Information.
4. **Lage aus drei Bins:** Aus den Werten links, auf und rechts der Spitze schätzen Quinn, Jacobsen und Candan die Lage der Linie auf einen kleinen Bruchteil eines Bins.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
cols = st.columns(len(C.PRESETS))
for col, name in zip(cols, C.PRESETS):
    with col:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.markdown("**Fenster**")
    window = st.selectbox("Fenster", list(C.WINDOWS), key="window_select", help="Rechteck = kein Fenster. Die anderen blenden die Ränder sanft aus.")
    if window == "Kaiser":
        beta = st.slider("Kaiser-Parameter β", float(C.BETA_MIN), float(C.BETA_MAX), float(st.session_state["beta_kept"]), step=0.5,
                         help="Größeres β: niedrigere Nebenkeulen, breitere Hauptkeule. β = 0 ist das Rechteck.")
        st.session_state["beta_kept"] = float(beta)
    else:
        beta = float(st.session_state["beta_kept"])
    n = st.slider("Länge n der Folge", *bounds("n_slider"), step=C.N_STEP, key="n_slider", help="Anzahl der Messwerte.")
    pad = st.select_slider("Auffüllfaktor", options=list(C.PAD_CHOICES), key="pad_select", help="Nullen anhängen: interpoliert die Kurve, ändert die Fensterkennzahlen nicht und liefert nicht mehr Information.")
    norm = st.selectbox("Normierung", list(C.NORMS), key="norm_select",
                        help="Leistungsdichte (Σw²) ist für Rauschen richtig, Linienhöhe (Σw) für die Leistung einer Linie. Der fensterabhängige Unterschied ist die Rauschbandbreite des Fensters.")
    st.markdown("**Signal**")
    line = st.slider("Linie 1 (Bruchteil der Abtastrate)", *bounds("line_slider"), step=C.LINE_STEP, key="line_slider", format="%.4f",
                     help="Lage der ersten Linie; mal n ergibt die Lage in Bins.")
    snr = st.slider("Linie 1 gegen Rauschen (dB)", *bounds("snr_slider"), key="snr_slider", help="Linienleistung gegen die Rauschvarianz 1, je Abtastwert.")
    gap = st.slider("Abstand der Linie 2 (Bins)", *bounds("gap_slider"), step=C.GAP_STEP, key="gap_slider", help="Abstand der zweiten Linie, in Bins zur ersten.")
    rel = st.slider("Linie 2 gegen Linie 1 (dB)", *bounds("rel_slider"), key="rel_slider", help="0 dB: gleich stark; −55 dB: sehr schwach.")
    background = st.selectbox("Hintergrund", list(C.BACKGROUNDS), key="background_select", help="Weiß: flach. Steil: farbig mit 57 dB Spannweite, zeigt Leckage im Tal.")
    st.markdown("**Zufall**")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neu würfeln", width="stretch", on_click=randomize_seed)

sync_query_params({"n_slider": int(n), "pad_select": int(pad), "window_select": window, "beta_kept": float(beta), "line_slider": float(line), "gap_slider": float(gap),
                   "rel_slider": int(rel), "snr_slider": int(snr), "background_select": background, "norm_select": norm, "seed_input": int(seed)})

settings = Settings(window, float(beta), int(n), int(pad), float(line), float(gap), int(rel), int(snr), background, norm, int(seed))
a = _analysis(settings)

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
two_comparable = settings.rel >= -20
m1, m2, m3, m4 = st.columns(4)
m1.metric("Hauptkeule bei −3 dB", f"{de(a.main_lobe, 2)} Bins", delta=f"Rauschbandbreite {de(a.enbw, 2)} Bins", delta_color="off")
m2.metric("Höchste Nebenkeule", f"{de(a.sidelobe_db, 1)} dB", delta="gegen den Gipfel", delta_color="off")
m3.metric("Gipfelverlust zwischen Bins", f"{de(a.scallop_db, 2)} dB", delta="Linie genau dazwischen", delta_color="off")
m4.metric("Linien aufgelöst" if two_comparable else "Schwache Linie sichtbar", ("ja" if a.resolved else "nein") if two_comparable else ("ja" if a.weak_visible else "nein"),
          delta=f"Abstand {de(settings.gap, 2)} Bins", delta_color="off")
st.plotly_chart(build_spectrum(a, zoom=True), width="stretch", key="spectrum_zoom")
st.caption("Ausschnitt um die beiden Linien (rot gestrichelt: wahre Lage). Blau: Spektrum dieser Folge, orange: exakter Erwartungswert, grün gepunktet: wahre Dichte des Hintergrunds.")
st.plotly_chart(build_spectrum(a, zoom=False), width="stretch", key="spectrum_full")
wc1, wc2 = st.columns(2)
wc1.plotly_chart(build_window_shape(a), width="stretch", key="window_shape")
wc2.plotly_chart(build_window_response(a), width="stretch", key="window_response")

st.markdown("#### Wo liegt Linie 1?")
bins1 = settings.f1_bins
rows = [{"Schätzer": name, "Schätzung (Bins)": de(v, 3), "Fehler (Bins)": de(a.errors[name], 3), "Vielfaches der Schranke": de(abs(a.errors[name]) / a.crb, 1)}
        for name, v in a.estimates.items()]
rows.append({"Schätzer": "Feines Maximum (Referenz)", "Schätzung (Bins)": de(bins1 + a.ml, 3), "Fehler (Bins)": de(a.ml, 3), "Vielfaches der Schranke": de(abs(a.ml) / a.crb, 1)})
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
st.caption(f"Wahre Lage {de(bins1, 3)} Bins; Cramér-Rao-Schranke für die Standardabweichung {de(a.crb, 4)} Bins. Die Schätzer lesen das Spektrum ohne Fenster und ohne Auffüllen; "
           "eine zweite Linie in der Nähe verfälscht sie. Eine einzelne Folge sagt wenig über die Streuung, die Experimente unten messen sie.")

if st.button("📄 PDF erstellen", key="pdf_make"):
    st.session_state["pdf_bytes"] = build_pdf(a)
    st.session_state["pdf_for"] = settings
if st.session_state.get("pdf_for") == settings and "pdf_bytes" in st.session_state:
    st.download_button("⬇️ PDF herunterladen", st.session_state["pdf_bytes"], file_name="fenster_linienlage.pdf", mime="application/pdf", key="pdf_download")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Was zeigt das Fenster, was verdeckt es?")
kind, text = ev.verdict_resolution(a) if two_comparable else ev.verdict_weak_line(a)
getattr(st, kind)(text)

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experimente")
st.caption("Feste Zufallsströme (Seed 200000 und folgende), unabhängig von den Reglern außer dem Kaiser-Parameter β. Die Anteile gelten über zufällige Phasen und Lagen zum Gitter (Standardfehler 4 bis 5 Prozentpunkte).")
exp = _experiments(float(beta))

st.markdown("#### Die Fenster im Vergleich")
st.dataframe(pd.DataFrame([{"Fenster": r["window"], "Rauschbandbreite (Bins)": de(r["enbw"], 2), "Hauptkeule −3 dB (Bins)": de(r["main_lobe"], 2),
                            "Höchste Nebenkeule (dB)": de(r["sidelobe_db"], 1), "Gipfelverlust ½ Bin (dB)": de(r["scallop_db"], 2),
                            "Linienhöhe gegen Dichte (dB)": de(r["line_vs_density_db"], 2)} for r in exp["table"]]), hide_index=True, width="stretch")
st.caption(f"n = {C.EXP_TABLE_N}; die letzte Spalte ist der Unterschied zwischen den beiden Normierungen.")
kind, text = ev.verdict_table(exp["table"])
getattr(st, kind)(text)
st.plotly_chart(build_valley(exp["valley"]), width="stretch", key="exp_valley")
st.caption("Hintergrund „Steil“ (56,7 dB Spannweite), bei 0,49 der Abtastrate: Erwartungswert gegen wahre Dichte, exakt berechnet.")
kind, text = ev.verdict_valley(exp["valley"])
getattr(st, kind)(text)

st.markdown("#### Auflösung gegen Dynamik")
st.caption(f"Zwei gleich starke Linien ohne Rauschen, {C.EXP_RES_RUNS} Läufe je Abstand; aufgelöst heißt zwei Gipfel mit mindestens 10 % Prominenz, Auffüllfaktor 16.")
st.plotly_chart(build_resolution(exp["resolution"]), width="stretch", key="exp_resolution")
kind, text = ev.verdict_resolution_sweep(exp["resolution"])
getattr(st, kind)(text)
st.dataframe(pd.DataFrame([{"Abstand (Bins)": de(r["distance"], 1), **{nm: de(100 * r[nm], 0) + " %" for nm in C.WINDOWS}} for r in exp["weak"]]), hide_index=True, width="stretch")
st.caption(f"Schwache Linie ({C.EXP_WEAK_REL_DB} dB) neben einer starken: Anteil der Läufe, in denen sie ihren Wert um 3 dB gegen dasselbe Signal ohne sie hebt, bei ganzzahligem Abstand (Nullstelle des Rechtecks). Das Flat-Top hat eine so breite Hauptkeule, dass es sie bei 3 Bins verschluckt.")
kind, text = ev.verdict_weak(exp["weak"])
getattr(st, kind)(text)

st.markdown("#### Auffüllen mit Nullen")
st.caption("Rechteckfenster, gleicher Aufbau; gegen den Auffüllfaktor.")
st.plotly_chart(build_padding(exp["padding"]), width="stretch", key="exp_padding")
kind, text = ev.verdict_padding(exp["padding"])
getattr(st, kind)(text)

st.markdown("#### Wo die Linie wirklich liegt")
st.dataframe(pd.DataFrame([{"Schätzer": r["estimator"], "größte Abweichung ohne Rauschen (Bins)": de(r["max"], 4), "mittlere (Bins)": de(r["mean"], 4)} for r in exp["bias"]]),
             hide_index=True, width="stretch")
st.caption("Reelle Linie, n = 256, Rechteckfenster, Lage zwischen −0,5 und 0,5 Bins um das Gitter (ohne Rauschen).")
st.plotly_chart(build_estimators(exp["rmse"]), width="stretch", key="exp_estimators")
st.caption(f"Mit Rauschen: RMSE geteilt durch die Cramér-Rao-Schranke ({C.EXP_EST_RUNS} Läufe, zufällige Lage und Phase, SNR 5 und 15 dB je Abtastwert); 1 = Schranke erreicht.")
kind, text = ev.verdict_estimators(exp["rmse"], exp["bias"])
getattr(st, kind)(text)

st.markdown("#### Das Abschneiden hat einen Preis")
st.markdown("Das Rechteckfenster ist das Abschneiden der Fourier-Reihe: Die Teilsumme einer Rechteckschwingung schwingt am Sprung über, ab etwa 50 Oberschwingungen praktisch unabhängig davon, wie viele Terme man nimmt "
            "(**Gibbs-Phänomen**). Sanftes Abschneiden mit Dreieckgewichten (Fejér) beseitigt das Überschwingen.")
st.dataframe(pd.DataFrame([{"Oberschwingungen bis": r["m"], "Überschwingen, harte Teilsumme (%)": de(100 * r["rect"], 1),
                            "Überschwingen mit Fejér-Gewichten (%)": de(100 * max(r["fejer"], 0.0), 1)} for r in exp["gibbs"]]), hide_index=True, width="stretch")
kind, text = ev.verdict_gibbs(exp["gibbs"])
getattr(st, kind)(text)

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Zwei Linien, kein Rauschen** in den Auflösungsexperimenten | Mit Rauschen wird die Grenze unschärfer; mehr als zwei Linien sind nicht untersucht. | Unterraumverfahren |
| **Die Streuung je Bin bleibt** | Die relative Streuung (Std/Mittel = 1 bei Rauschen) senkt kein Fenster; im Leckage-Tal des Rechtecks ist sie sogar größer. | Bartlett und Welch, Multitaper |
| **Schätzer fern von 0 und Nyquist** | Eine reelle Linie nahe 0 oder der halben Abtastrate sieht ihr Spiegelbild und wird verzerrt. | – |
| **Eine Linie, weißes Rauschen** für die Schranke | Bei farbigem Rauschen oder mehreren Linien gilt die Schranke nicht in dieser Form. | AR-Spektrum, Unterraumverfahren |
| **Stationär** | Wandert die Linie, verschmiert das Fenster sie. | Spektrogramm |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Gefenstertes Periodogramm.** $I_w(f) = \dfrac{1}{f_s \sum_m w_m^2}\Big|\sum_{m=0}^{n-1} w_m x[m]\,e^{-2\pi i f m/f_s}\Big|^2$ (Leistungsdichte) oder
$2|X|^2 / (\sum w_m)^2$ (Linienleistung). **Rauschbandbreite:** $\mathrm{ENBW} = n\sum w_m^2 / (\sum w_m)^2$ Bins.

**Erwartungswert (exakt).** $E|X(f)|^2 = |\sum w_m s[m] e^{-i\omega m}|^2 + \sum_{|\tau|<n} r(\tau)\, c_w(\tau)\cos(\omega\tau)$ mit der Fensterautokorrelation
$c_w(\tau) = \sum_m w_m w_{m+\tau}$. Leckage ist Bias: der Erwartungswert ist die Faltung der wahren Dichte mit $|W(f)|^2$.

**Gipfelverlust** genau zwischen zwei Bins: $20\log_{10}\big|\sum w_m e^{-i\pi m/n}\big| / \sum w_m$; beim Rechteck $\to -3{,}92$ dB.

**Lage aus drei Bins.** Mit $X_{-1}, X_0, X_{+1}$ um die Spitze: Jacobsen $\delta = \mathrm{Re}\,\dfrac{X_{-1} - X_{+1}}{2X_0 - X_{-1} - X_{+1}}$, Candan mit dem Faktor $\tan(\pi/n)/(\pi/n)$,
Rife aus den Verhältnissen der Beträge, Quinn aus den Realteilen der Verhältnisse $X_{\pm 1}/X_0$. **Schranke** (reelle Linie, weißes Rauschen): $\sigma_f \ge \sqrt{3/(\pi^2\,\mathrm{SNR}\,n(n^2-1))}$ in Bruchteilen der Abtastrate, SNR $= A^2/(2\sigma^2)$.

**Gibbs.** Die Teilsumme $\frac{4}{\pi}\sum_{k\ \mathrm{ungerade}\le M}\frac{\sin k\theta}{k}$ überschwingt um $\frac{2}{\pi}\mathrm{Si}(\pi) - 1 \approx 0{,}179$ der Amplitude.

**Quellen.** Lehrbuchwissen (Harris 1978 zu Fenstern; Rife und Boorstyn 1974 zur Schranke; Quinn 1994; Jacobsen und Kootsookos 2007; Candan 2011). Formeln hier per Simulation, Handrechnung
und gegen scipy geprüft; die Quellen selbst wurden nicht gelesen.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Spektren und Wavelets: von Fourier zum Vergleich](https://sebastianhanisch.net/konzepte-spektral-wavelet-analyse.html)."
)
