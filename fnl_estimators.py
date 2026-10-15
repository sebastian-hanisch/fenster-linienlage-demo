"""Schätzer der Frequenz einer Linie aus den drei Bins um die Spitze (Rechteckfenster), Cramér-Rao-Schranke und feines Maximum als Referenz.

Alle Schätzer liefern die Lage der Linie relativ zum Spitzen-Bin k in Bins (delta, zwischen -0,5 und 0,5). Arbeiten auf Feldern: X ist komplex
(Zeilen = Läufe), k die Spitzen-Bins je Zeile.
"""

import math

import numpy as np

NAMES = ("Spitze", "Parabel", "Quinn 1", "Jacobsen", "Rife", "Candan")


def peak_bins(X, lo: int, hi: int) -> np.ndarray:
    """Spitzen-Bins je Zeile im Bereich [lo, hi) der letzten Achse."""
    X = np.atleast_2d(X)
    return np.argmax(np.abs(X[:, lo:hi]), axis=1) + lo


def three_bins(X, k):
    X = np.atleast_2d(X)
    rows = np.arange(X.shape[0])
    k = np.atleast_1d(k)
    return X[rows, k - 1], X[rows, k], X[rows, k + 1]


def est_peak(xm, x0, xp, n: int) -> np.ndarray:
    return np.zeros(np.shape(x0))


def est_parabola(xm, x0, xp, n: int) -> np.ndarray:
    """Parabel durch die logarithmierten Betragsquadrate der drei Bins."""
    la, lb, lc = np.log(np.abs(xm) ** 2), np.log(np.abs(x0) ** 2), np.log(np.abs(xp) ** 2)
    return 0.5 * (la - lc) / (la - 2.0 * lb + lc)


def est_quinn1(xm, x0, xp, n: int) -> np.ndarray:
    a1 = (xm / x0).real
    a2 = (xp / x0).real
    d1 = a1 / (1.0 - a1)
    d2 = -a2 / (1.0 - a2)
    return np.where((d1 > 0) & (d2 > 0), d2, d1)


def est_jacobsen(xm, x0, xp, n: int) -> np.ndarray:
    return ((xm - xp) / (2.0 * x0 - xm - xp)).real


def est_rife(xm, x0, xp, n: int) -> np.ndarray:
    am, a0, ap = np.abs(xm), np.abs(x0), np.abs(xp)
    return np.where(ap > am, ap / (a0 + ap), -am / (a0 + am))


def est_candan(xm, x0, xp, n: int) -> np.ndarray:
    """Jacobsen mit dem Faktor tan(pi/n) / (pi/n), der den Fehler endlicher Länge beseitigt."""
    return math.tan(math.pi / n) / (math.pi / n) * est_jacobsen(xm, x0, xp, n)


ESTIMATORS = {"Spitze": est_peak, "Parabel": est_parabola, "Quinn 1": est_quinn1, "Jacobsen": est_jacobsen, "Rife": est_rife, "Candan": est_candan}


def estimate_all(X, lo: int, hi: int) -> dict:
    """{Name: Schätzung der Linie in Bins (k + delta)} je Zeile, Spitze im Bereich [lo, hi)."""
    k = peak_bins(X, lo, hi)
    xm, x0, xp = three_bins(X, k)
    n = np.atleast_2d(X).shape[1]
    return {name: k + fn(xm, x0, xp, n) for name, fn in ESTIMATORS.items()}


def crb_bins(n: int, snr_db: float) -> float:
    """Cramér-Rao-Schranke der Standardabweichung der Frequenz einer reellen Linie in Bins: sqrt(3 / (pi^2 snr n (n^2 - 1))) * n,
    snr = A^2 / (2 sigma^2) je Abtastwert (Rife und Boorstyn 1974, reelle Linie fern von 0 und Nyquist)."""
    snr = 10.0 ** (snr_db / 10.0)
    return math.sqrt(3.0 / (math.pi ** 2 * snr * n * (n * n - 1.0))) * n


def refined_maximum(x, lo: int, hi: int, factor: int = 32) -> np.ndarray:
    """Frequenz des Periodogrammmaximums in Bins auf einem `factor`-fach aufgefüllten Gitter, nachgeschärft mit einer Parabel (ML-Näherung der Linie)."""
    x = np.atleast_2d(np.asarray(x, dtype=float))
    n = x.shape[1]
    Xf = np.fft.rfft(x, n * factor, axis=1)
    p = np.abs(Xf) ** 2
    k = np.argmax(p[:, lo * factor:hi * factor], axis=1) + lo * factor
    rows = np.arange(x.shape[0])
    la, lb, lc = np.log(p[rows, k - 1]), np.log(p[rows, k]), np.log(p[rows, k + 1])
    return (k + 0.5 * (la - lc) / (la - 2.0 * lb + lc)) / factor
