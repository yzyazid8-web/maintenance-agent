"""Calibration de probabilités : score brut XGBoost -> probabilité."""

import numpy as np
from sklearn.linear_model import LogisticRegression

_EPS = 1e-6


def _logit(p):
    """logit d'une probabilité (bornée pour éviter ±inf)."""
    p = np.clip(np.asarray(p, dtype=float), _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


class PlattCalibrator:
    """Calibration de Platt : régression logistique sur le LOGIT du score brut -> P(panne)."""

    def __init__(self):
        self._lr = LogisticRegression(C=1e6)  # C élevé , ajustement non régularisé (2 paramètres)

    def fit(self, raw, y) -> "PlattCalibrator":
        self._lr.fit(_logit(raw).reshape(-1, 1), np.asarray(y))
        return self

    def predict(self, raw):
        """Probabilité calibrée pour un ou des scores bruts."""
        return self._lr.predict_proba(_logit(raw).reshape(-1, 1))[:, 1]
