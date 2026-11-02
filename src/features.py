"""Feature engineering + préprocesseur"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config

# 3 variables physiques
ENGINEERED = ["power_w", "delta_t_k", "wear_x_torque"]
NUMERIC = config.NUMERIC_FEATURES + ENGINEERED
CATEGORICAL = config.CATEGORICAL_FEATURES  # ["type"]


# Formules physiques 
def power_w(row):
    """Puissance mécanique (W) = couple x vitesse angulaire."""
    return row["torque_nm"] * row["rotational_speed_rpm"] * 2 * np.pi / 60


def delta_t_k(row):
    """Écart de température procédé - air (K)."""
    return row["process_temp_k"] - row["air_temp_k"]


def wear_x_torque(row):
    """Interaction usure x couple."""
    return row["tool_wear_min"] * row["torque_nm"]


class PhysicsFeatures(BaseEstimator, TransformerMixin):
    """Ajoute puissance, écart de température et usure x couple"""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["power_w"] = power_w(X)
        X["delta_t_k"] = delta_t_k(X)
        X["wear_x_torque"] = wear_x_torque(X)
        return X


def build_preprocessor(scale: bool) -> ColumnTransformer:
    """Encode `type` (one-hot) ; standardise le numérique seulement si `scale`"""
    num = StandardScaler() if scale else "passthrough"
    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("num", num, NUMERIC),
        ]
    )


def build_pipeline(model, scale: bool) -> Pipeline:
    """PhysicsFeatures -> préprocesseur -> modèle. Meme objet pour l'entrainement et le service."""
    return Pipeline(
        [
            ("physics", PhysicsFeatures()),
            ("prep", build_preprocessor(scale)),
            ("model", model),
        ]
    )
