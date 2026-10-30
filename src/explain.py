"""Explicabilité SHAP du pipeline XGBoost (importances globales + explication locale)"""

import numpy as np
import shap


def _feature_names(pipe) -> list[str]:
    """Noms des colonnes vues par le modèle"""
    return list(pipe.named_steps["prep"].get_feature_names_out())


def _transform(pipe, X):
    """Applique physics + préprocesseur (tout le pipeline sauf le modèle)."""
    return pipe[:-1].transform(X)


def global_importance(pipe, X) -> dict[str, float]:
    """Importance globale = moyenne des |valeurs SHAP| par feature, triée décroissante."""
    model = pipe.named_steps["model"]
    shap_values = shap.TreeExplainer(model).shap_values(_transform(pipe, X))
    importance = np.abs(shap_values).mean(axis=0)
    names = _feature_names(pipe)
    return dict(sorted(zip(names, importance), key=lambda kv: kv[1], reverse=True))


def explain_row(pipe, X_row) -> dict[str, float]:
    """Contribution SHAP de chaque feature pour une seule observation"""
    model = pipe.named_steps["model"]
    shap_values = shap.TreeExplainer(model).shap_values(_transform(pipe, X_row))
    return dict(zip(_feature_names(pipe), shap_values[0]))
