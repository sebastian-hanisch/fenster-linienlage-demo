"""Kern: Fenster, ihre Kennzahlen, gefenstertes Periodogramm, exakter Erwartungswert, Gipfelzählung und Gibbs-Teilsummen.

Reine Funktionen, kein Zufall, kein Streamlit. Fenster sind periodisch (DFT-gerade), wie sie die Spektralanalyse braucht. Frequenzachse der
Kennzahlen in Bins (Vielfache von f_s / n).
"""

import math

import numpy as np

NAMES = ("Rechteck", "Hann", "Hamming", "Blackman", "Kaiser", "Flat-Top")
COSINE_SUM = {
    "Hann": (0.5, 0.5),
    "Hamming": (0.54, 0.46),
    "Blackman": (0.42, 0.5, 0.08),
    "Flat-Top": (0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368),
}
NORMS = ("Leistungsdichte", "Linienhöhe")


def window(name: str, n: int, beta: float = 8.0) -> np.ndarray:
    """Periodisches Fenster der Länge n. `beta` wirkt nur bei Kaiser."""
    if name == "Rechteck":
        return np.ones(n)
    if name == "Kaiser":
        return np.kaiser(n + 1, beta)[:-1]
    if name in COSINE_SUM:
        m = np.arange(n)
        w = np.zeros(n)
        for k, a in enumerate(COSINE_SUM[name]):
            w += (-1) ** k * a * np.cos(2.0 * np.pi * k * m / n)
        return w
    raise ValueError(f"unbekanntes Fenster: {name}")


def enbw(w) -> float:
    """Äquivalente Rauschbandbreite in Bins: n sum(w^2) / (sum w)^2."""
    w = np.asarray(w, dtype=float)
    return float(len(w) * (w ** 2).sum() / w.sum() ** 2)


def coherent_gain(w) -> float:
    """Mittelwert des Fensters (Gipfelhöhe einer Linie auf dem Gitterpunkt gegen das Rechteck)."""
    w = np.asarray(w, dtype=float)
    return float(w.sum() / len(w))


def response(w, pad: int = 64):
    """(Frequenz in Bins, Betragsantwort normiert auf den Gleichanteil) des Fensters, mit `pad`-fach aufgefüllter DFT."""
    w = np.asarray(w, dtype=float)
    n = len(w)
    r = np.abs(np.fft.rfft(w, n * pad))
    return np.arange(len(r)) / pad, r / r[0]


def main_lobe_3db(w, pad: int = 64) -> float:
    """Breite der Hauptkeule bei -3 dB in Bins (beidseitig)."""
    f, r = response(w, pad)
    below = np.flatnonzero(r < 1.0 / math.sqrt(2.0))
    i = int(below[0])
    # lineare Interpolation zwischen den Gitterpunkten i-1 und i
    t = (r[i - 1] - 1.0 / math.sqrt(2.0)) / (r[i - 1] - r[i])
    return 2.0 * float(f[i - 1] + t * (f[i] - f[i - 1]))


def sidelobe_peak_db(w, pad: int = 64) -> float:
    """Höchste Nebenkeule in dB gegen den Gipfel (erstes lokales Minimum der Antwort beendet die Hauptkeule)."""
    _, r = response(w, pad)
    mins = np.flatnonzero((r[1:-1] < r[:-2]) & (r[1:-1] <= r[2:])) + 1
    return 20.0 * math.log10(float(r[int(mins[0]):].max()))


def scallop_loss_db(w) -> float:
    """Gipfelverlust einer Linie genau zwischen zwei Bins (exakt, ohne Auffüllen): 20 log10 |sum w e^{-i pi m / n}| / sum w."""
    w = np.asarray(w, dtype=float)
    m = np.arange(len(w))
    return 20.0 * math.log10(abs(np.sum(w * np.exp(-1j * np.pi * m / len(w)))) / w.sum())


def spectrum(x, w, fs: float, pad: int = 1, norm: str = "Leistungsdichte") -> np.ndarray:
    """Gefenstertes Periodogramm an den Bins k fs / (n pad), k = 0 .. n pad / 2.

    Leistungsdichte: |X|^2 / (fs sum w^2) (Leistung je Hz, für Rauschen); Linienhöhe: 2 |X|^2 / (sum w)^2 (Leistung einer Linie, für Gipfel).
    """
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    n = len(x)
    p = np.abs(np.fft.rfft(x * w, n * pad)) ** 2
    if norm == "Leistungsdichte":
        return p / (fs * (w ** 2).sum())
    if norm == "Linienhöhe":
        return 2.0 * p / w.sum() ** 2
    raise ValueError(f"unbekannte Normierung: {norm}")


def expected_spectrum(f_hz, fs: float, w, r, s=None, norm: str = "Leistungsdichte") -> np.ndarray:
    """E[gefenstertes Periodogramm] exakt: Rauschen mit Autokovarianz r (Lags 0 .. n-1) über die Fensterautokorrelation c_w(tau) = sum w_m w_{m+tau},
    deterministische Folge s (Linien, feste Phase) direkt: E|X|^2 = |sum w s e^{-iwm}|^2 + sum_tau r(tau) c_w(tau) cos(w tau) (beide Vorzeichen von tau)."""
    w = np.asarray(w, dtype=float)
    r = np.asarray(r, dtype=float)
    n = len(w)
    f = np.atleast_1d(np.asarray(f_hz, dtype=float))
    om = 2.0 * np.pi * f / fs
    c = np.correlate(w, w, mode="full")[n - 1:]
    tau = np.arange(n)
    wts = r * c
    out = np.empty(len(f))
    for lo in range(0, len(f), 256):
        cs = np.cos(np.outer(om[lo:lo + 256], tau))
        out[lo:lo + 256] = wts[0] + 2.0 * (cs[:, 1:] @ wts[1:])
        if s is not None:
            m = np.arange(n)
            out[lo:lo + 256] += np.abs(np.exp(-1j * np.outer(om[lo:lo + 256], m)) @ (w * np.asarray(s, dtype=float))) ** 2
    if norm == "Leistungsdichte":
        return out / (fs * (w ** 2).sum())
    if norm == "Linienhöhe":
        return 2.0 * out / w.sum() ** 2
    raise ValueError(f"unbekannte Normierung: {norm}")


def count_peaks(y, rel_prominence: float, reference: float | None = None) -> int:
    """Zahl der Gipfel mit Prominenz >= rel_prominence * reference (Standard: Maximum von y). Gipfel sind innere Punkte, die höher als beide Nachbarn sind
    (bei Plateaus der erste Punkt); die Ränder zählen nie.

    Prominenz wie bei scipy.signal.find_peaks ohne Fensterlänge: Höhe über der höheren der beiden Basen; Basis = tiefster Wert bis zum nächsten höheren
    Punkt auf dieser Seite, ohne einen solchen bis zum Rand.
    """
    y = np.asarray(y, dtype=float)
    if len(y) < 3:
        return 0
    ref = float(y.max()) if reference is None else float(reference)
    count = 0
    for i in range(1, len(y) - 1):
        if not (y[i] > y[i - 1] and y[i] >= y[i + 1]):
            continue
        j = i - 1
        while j > 0 and y[j] <= y[i]:
            j -= 1
        left = y[j:i + 1].min() if y[j] > y[i] else y[0:i + 1].min()
        k = i + 1
        while k < len(y) - 1 and y[k] <= y[i]:
            k += 1
        right = y[i:k + 1].min() if y[k] > y[i] else y[i:].min()
        if y[i] - max(left, right) >= rel_prominence * ref:
            count += 1
    return count


def two_lines_resolved(spec, pad: int, f1_bins: float, f2_bins: float, rel_prominence: float = 0.1) -> bool:
    """Zwei Linien gelten als aufgelöst, wenn das (aufgefüllte) Spektrum zwischen f1 - d/2 und f2 + d/2 mindestens zwei Gipfel mit Prominenz >= 10 % des Maximums hat."""
    d = abs(f2_bins - f1_bins)
    lo = max(int(math.floor((min(f1_bins, f2_bins) - d / 2) * pad)), 0)
    hi = int(math.ceil((max(f1_bins, f2_bins) + d / 2) * pad)) + 1
    return count_peaks(np.asarray(spec)[lo:hi], rel_prominence, reference=float(np.max(spec))) >= 2


def gibbs_overshoot(m_max: int, fejer: bool = False) -> float:
    """Größtes Überschwingen der Teilsumme einer Rechteckschwingung ±1 bis zur Oberschwingung m_max über die ganze Halbperiode (0, pi), in Bruchteilen der
    Amplitude (> 0 heißt über 1, < 0 heißt, dass die Teilsumme nie bis 1 reicht).

    fejer: Dreieckgewichte (1 - k / (m_max + 1)); dann ist die Teilsumme ein gleitender Mittelwert der Rechteckschwingung und überschreitet 1 nie."""
    k = np.arange(1, m_max + 1, 2)
    weights = (1.0 - k / (m_max + 1.0)) if fejer else np.ones(len(k))
    coeff = 4.0 / (math.pi * k) * weights
    points = max(2001, 40 * (m_max + 1))
    theta = np.linspace(1e-9, math.pi - 1e-9, points)
    best = -np.inf
    for lo in range(0, points, 1000):
        best = max(best, float((coeff @ np.sin(np.outer(k, theta[lo:lo + 1000]))).max()))
    return best - 1.0
