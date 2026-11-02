"""Détection de dérive (drift) des données : PSI, Wasserstein normalisé, domain classifier.

`simulate_drift` fabrique un batch de "production" dérivé  - c'est une
dérive covariable SIMULÉE, pour simuler la surveillance.
"""

import numpy as np
from scipy.stats import wasserstein_distance
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src import config

PSI_MODERATE = 0.1
PSI_SIGNIFICANT = 0.25


def psi(reference, current, bins: int = 10) -> float:
    """Population Stability Index : 0 = stable, >0,1 modéré, >0,25 significatif"""
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0
    # Bornes ouvertes : les valeurs hors plage de référence (= la dérive) tombent dans
    # les bacs extrêmes au lieu d'être ignorées par np.histogram.
    edges[0], edges[-1] = -np.inf, np.inf
    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)
    ref_pct = np.clip(ref_counts / ref_counts.sum(), 1e-6, None)
    cur_pct = np.clip(cur_counts / cur_counts.sum(), 1e-6, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def wasserstein_norm(reference, current) -> float:
    """Distance de Wasserstein normalisée par l'écart-type de la référence (seuil 0,1)"""
    scale = float(np.std(reference)) or 1.0
    return float(wasserstein_distance(reference, current) / scale)


def _status(psi_value: float) -> str:
    if psi_value > PSI_SIGNIFICANT:
        return "🔴"
    if psi_value > PSI_MODERATE:
        return "🟡"
    return "🟢"


def domain_classifier_auc(reference_df, current_df, features) -> float:
    """AUC d'un classifieur réf (0) vs courant (1). 0,5 = aucune dérive ; -> 1 = forte dérive"""
    X = np.vstack([reference_df[features].to_numpy(), current_df[features].to_numpy()])
    y = np.concatenate([np.zeros(len(reference_df)), np.ones(len(current_df))])
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=config.SEED, stratify=y
    )
    clf = HistGradientBoostingClassifier(max_depth=3, max_iter=100, random_state=config.SEED)
    clf.fit(X_tr, y_tr)
    return float(roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1]))


def simulate_drift(df, severity: float = 1.0, seed: int = 0):
    """Batch de production dérivé (covariate drift simulé)"""
    rng = np.random.default_rng(seed)
    out = df.copy()
    out["tool_wear_min"] = out["tool_wear_min"] + severity * 40
    out["torque_nm"] = out["torque_nm"] * (1 + severity * 0.15)
    out["process_temp_k"] = out["process_temp_k"] + severity * rng.normal(1.0, 0.3, len(out))
    return out


def drift_report(reference_df, current_df, features=None) -> dict:
    """Rapport de dérive : statut par variable (PSI, Wasserstein) + AUC multivariée globale"""
    features = features or config.NUMERIC_FEATURES
    per_feature = []
    for feat in features:
        ref = reference_df[feat].to_numpy()
        cur = current_df[feat].to_numpy()
        p = psi(ref, cur)
        per_feature.append(
            {
                "feature": feat,
                "psi": round(p, 3),
                "wasserstein": round(wasserstein_norm(ref, cur), 3),
                "status": _status(p),
            }
        )
    return {
        "per_feature": per_feature,
        "domain_classifier_auc": round(
            domain_classifier_auc(reference_df, current_df, features), 3
        ),
    }
