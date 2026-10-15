"""PDF-Export der aktuellen Einstellung (fpdf2). Kernschriften sind latin-1: Umlaute gehen, Pfeile und griechische Buchstaben werden ersetzt."""

from __future__ import annotations

from fpdf import FPDF

from fnl_evaluation import Analysis, de

_REPLACE = {"–": "-", "—": "-", "−": "-", "²": "2", "€": "EUR", "→": "->", "β": "beta", "Σ": "Summe"}


def _s(text) -> str:
    out = str(text)
    for old, new in _REPLACE.items():
        out = out.replace(old, new)
    return out.encode("latin-1", "replace").decode("latin-1")


def build_pdf(a: Analysis) -> bytes:
    s = a.settings
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _s("Fenster und Linienlage: Einstellung und Kennzahlen"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    win = s.window + (f" (beta = {de(s.beta, 1)})" if s.window == "Kaiser" else "")
    pdf.multi_cell(0, 5, _s(f"Fenster: {win}, n = {s.n}, Auffüllen {s.pad}-fach, Linie 1 bei {de(s.f1_bins, 2)} Bins ({s.snr} dB je Abtastwert), "
                            f"Linie 2 im Abstand {de(s.gap, 2)} Bins ({s.rel} dB), Hintergrund: {s.background}, Normierung: {s.norm}, Seed {s.seed}. Alles simuliert."),
                   new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    lines = [
        f"Fenster: äquivalente Rauschbandbreite {de(a.enbw, 2)} Bins, Hauptkeule bei -3 dB {de(a.main_lobe, 2)} Bins, höchste Nebenkeule {de(a.sidelobe_db, 1)} dB, "
        f"Gipfelverlust zwischen zwei Bins {de(a.scallop_db, 2)} dB.",
        f"Linien aufgelöst: {'ja' if a.resolved else 'nein'}; schwache Linie sichtbar: {'ja' if a.weak_visible else 'nein'}.",
        "Fehler der Lage von Linie 1 in Bins: " + ", ".join(f"{k} {de(v, 3)}" for k, v in a.errors.items()) + f"; feines Maximum {de(a.ml, 3)}; Schranke {de(a.crb, 4)}.",
    ]
    for line in lines:
        pdf.multi_cell(0, 5, _s(line), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _s("Quelle: fenster-linienlage-demo, Sebastian Hanisch."), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
