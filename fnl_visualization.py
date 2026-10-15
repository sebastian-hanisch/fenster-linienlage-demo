"""Plotly-Figuren. Alle Achsen fest (fixedrange), damit Touch-Scrollen nicht am Diagramm hängen bleibt."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

import fnl_constants as C
import fnl_windows as W
from fnl_evaluation import Analysis

EPS = 1e-30


def _db(v):
    return 10.0 * np.log10(np.maximum(np.asarray(v, dtype=float), EPS))


def _fixed(fig: go.Figure, height: int = 360, **layout) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.25), **layout)
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def build_spectrum(a: Analysis, zoom: bool = True) -> go.Figure:
    """Spektrum in dB über der Frequenz in Bins; zoom: Ausschnitt um die beiden Linien, sonst das ganze Spektrum."""
    s = a.settings
    fb = a.f_bins
    if zoom:
        lo, hi = max(0.0, s.f1_bins - 8.0), min(s.n / 2.0, s.f2_bins + 8.0)
    else:
        lo, hi = 0.0, s.n / 2.0
    sel = (fb >= lo) & (fb <= hi)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fb[sel], y=_db(a.spec[sel]), mode="lines", line=dict(color=C.COLOR_I, width=1.5), name="Spektrum",
                             hovertemplate="%{x:.2f} Bins: %{y:.1f} dB<extra></extra>"))
    fig.add_trace(go.Scatter(x=fb[sel], y=_db(a.expected[sel]), mode="lines", line=dict(color=C.COLOR_EXPECTED, width=2.5), name="Erwartungswert (exakt)",
                             hovertemplate="%{x:.2f} Bins: %{y:.1f} dB<extra></extra>"))
    if s.norm == "Leistungsdichte":
        fig.add_trace(go.Scatter(x=fb[sel], y=_db(a.true_density[sel]), mode="lines", line=dict(color=C.COLOR_TRUE, width=2, dash="dot"), name="wahre Dichte des Hintergrunds",
                                 hovertemplate="%{x:.2f} Bins: %{y:.1f} dB<extra></extra>"))
    for pos in (s.f1_bins, s.f2_bins):
        if lo <= pos <= hi:
            fig.add_vline(x=pos, line=dict(color=C.COLOR_LINE, width=1.2, dash="dash"))
    fig.update_xaxes(title_text="Frequenz (Bins)", range=[lo, hi])
    fig.update_yaxes(title_text="Leistungsdichte (dB)" if s.norm == "Leistungsdichte" else "Linienleistung (dB)")
    return _fixed(fig, height=380)


def build_window_shape(a: Analysis) -> go.Figure:
    """Fensterform über den Messwerten, auf den Höchstwert 1 normiert."""
    w = a.w / a.w.max()
    fig = go.Figure(go.Scatter(x=np.arange(len(w)), y=w, mode="lines", line=dict(color=C.COLOR_WINDOW, width=2.5), hovertemplate="%{x}: %{y:.2f}<extra></extra>"))
    fig.update_xaxes(title_text="Messwert Nr.")
    fig.update_yaxes(title_text="Gewicht", range=[-0.05, 1.05])
    return _fixed(fig, height=300, showlegend=False)


def build_window_response(a: Analysis) -> go.Figure:
    """Betragsquadrat der Fensterantwort in dB über dem Abstand in Bins (±12 Bins)."""
    f, r = W.response(a.w, 32)
    sel = f <= 12.0
    xs = np.r_[-f[sel][:0:-1], f[sel]]
    ys = np.r_[r[sel][:0:-1], r[sel]]
    fig = go.Figure(go.Scatter(x=xs, y=_db(ys ** 2), mode="lines", line=dict(color=C.COLOR_WINDOW, width=2), hovertemplate="%{x:.2f} Bins: %{y:.1f} dB<extra></extra>"))
    fig.update_xaxes(title_text="Abstand in Bins")
    fig.update_yaxes(title_text="Antwort (dB)", range=[-120, 3])
    return _fixed(fig, height=300, showlegend=False)


def build_resolution(rows) -> go.Figure:
    names = [k for k in rows[0] if k not in ("delta", "runs")]
    fig = go.Figure()
    for nm in names:
        fig.add_trace(go.Scatter(x=[r["delta"] for r in rows], y=[100 * r[nm] for r in rows], mode="lines+markers", name=nm,
                                 hovertemplate=nm + ", %{x} Bins: %{y:.0f} %<extra></extra>"))
    fig.update_xaxes(title_text="Abstand der Linien (Bins)")
    fig.update_yaxes(title_text="Anteil aufgelöst (%)", range=[-3, 103])
    return _fixed(fig, height=340)


def build_padding(rows) -> go.Figure:
    pads = [k for k in rows[0] if k not in ("delta", "runs")]
    fig = go.Figure()
    for r in rows:
        fig.add_trace(go.Scatter(x=[str(p) for p in pads], y=[100 * r[p] for p in pads], mode="lines+markers", name=f"Abstand {r['delta']:g} Bins".replace(".", ","),
                                 hovertemplate="Auffüllfaktor %{x}: %{y:.0f} %<extra></extra>"))
    fig.update_xaxes(title_text="Auffüllfaktor (Rechteckfenster)", type="category")
    fig.update_yaxes(title_text="Anteil aufgelöst (%)", range=[-3, 103])
    return _fixed(fig, height=340)


def build_valley(rows) -> go.Figure:
    fig = go.Figure()
    for n in sorted({r["n"] for r in rows}):
        sel = [r for r in rows if r["n"] == n]
        fig.add_trace(go.Bar(x=[r["window"] for r in sel], y=[r["bias_db"] for r in sel], name=f"n = {n}",
                             hovertemplate="%{x}: %{y:.1f} dB<extra></extra>"))
    fig.update_yaxes(title_text="Erwartungswert über wahrer Dichte (dB)")
    fig.update_layout(barmode="group")
    return _fixed(fig, height=340)


def build_estimators(rows) -> go.Figure:
    names = [k for k in rows[0] if k not in ("n", "snr", "crb", "runs", "outliers")]
    fig = go.Figure()
    for r in rows:
        fig.add_trace(go.Bar(x=names, y=[r[nm] for nm in names], name=f"n = {r['n']}, {r['snr']} dB", hovertemplate="%{x}: %{y:.2f}-fach<extra></extra>"))
    fig.add_trace(go.Scatter(x=names, y=[1.0] * len(names), mode="lines", line=dict(color="gray", dash="dot"), name="Schranke (1,0)", hoverinfo="skip"))
    values = [r[nm] for r in rows for nm in names]
    fig.update_yaxes(title_text="RMSE geteilt durch Schranke", type="log", range=[float(np.log10(max(min(values), 0.1) * 0.8)), float(np.log10(max(values) * 1.4))])
    fig.update_layout(barmode="group")
    return _fixed(fig, height=360)
