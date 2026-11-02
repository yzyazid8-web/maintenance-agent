"""Calibration de probabilités : score brut XGBoost -> probabilité"""

import numpy as np
from sklearn.linear_model import LogisticRegression


class PlattCalibrator:
    """Calibration sigmoide : ajuste une régression logistique score_brut -> P(panne)."""

    def __init__(self):
        self._lr = LogisticRegression(C=1e6)  # C élevé , ajustement non régularisé (2 paramètres)

    def fit(self, raw, y) -> "PlattCalibrator":
        self._lr.fit(np.asarray(raw, dtype=float).reshape(-1, 1), np.asarray(y))
        return self

    def predict(self, raw):
        """Probabilité calibrée pour un ou des scores bruts."""
        return self._lr.predict_proba(np.asarray(raw, dtype=float).reshape(-1, 1))[:, 1]
