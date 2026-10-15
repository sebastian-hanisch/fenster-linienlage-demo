"""Gemeinsame Fixtures: die Mini-Instanz, an der jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def hann8():
    """Hann-Fenster der Länge 8 von Hand: w = 0,5 - 0,5 cos(2 pi m / 8) = (0; 0,1464; 0,5; 0,8536; 1; 0,8536; 0,5; 0,1464).

    Summe 4, Summe der Quadrate 3, Rauschbandbreite 8 * 3 / 16 = 1,5 Bins.
    """
    return np.array([0.0, 0.5 - 0.5 * np.cos(np.pi / 4), 0.5, 0.5 + 0.5 * np.cos(np.pi / 4), 1.0, 0.5 + 0.5 * np.cos(np.pi / 4), 0.5, 0.5 - 0.5 * np.cos(np.pi / 4)])


@pytest.fixture
def cosine8():
    """N = 8, fs = 1, x = cos(pi m / 2): X_2 = X_6 = 4, alle anderen 0 (Linie genau auf dem Bin 2)."""
    return np.cos(np.pi * np.arange(8) / 2.0)
