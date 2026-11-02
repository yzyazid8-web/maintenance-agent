"""Outils de l'agent : fonctions pures opérant sur le modèle et les données via Deps"""

import json
from dataclasses import dataclass

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from src import config
from src.agent.knowledge_graph import KnowledgeGraph
from src.agent.memory import EpisodicMemory
from src.data import load_data
from src.explain import explain_row, global_importance


@dataclass
class Deps:
    """État partagé injecté dans les outils de l'agent"""

    pipe: Pipeline
    X: pd.DataFrame  # features (sans la cible)
    scored: pd.DataFrame  # données + colonne `risk`
    metrics: dict
    kg: KnowledgeGraph  # graphe FMEA (modes -> causes / actions)
    memory: EpisodicMemory  # mémoire
    calibrator: object  # IsotonicRegression : score brut -> probabilité calibrée


def build_deps() -> Deps:
    """Charge le pipeline, les métriques et le risque out-of-fold (hors-échantillon)."""
    pipe = joblib.load(config.MODELS_DIR / "pipeline.joblib")
    calibrator = joblib.load(config.MODELS_DIR / "calibrator.joblib")
    metrics = json.loads((config.MODELS_DIR / "metrics.json").read_text())
    df = load_data()
    X = df.drop(columns=[config.TARGET])
    # Risque affiché = prédictions out-of-fold calibrées (hors-échantillon + probabilité honnête).
    oof = pd.read_csv(config.MODELS_DIR / "oof_scores.csv", index_col="machine_id")
    scored = df.assign(risk=oof["risk"].to_numpy())
    return Deps(
        pipe=pipe,
        X=X,
        scored=scored,
        metrics=metrics,
        kg=KnowledgeGraph(),
        memory=EpisodicMemory(),
        calibrator=calibrator,
    )


def _clean(name: str) -> str:
    """Retire le préfixe du préprocesseur ('num__torque_nm' -> 'torque_nm')"""
    return name.split("__", 1)[-1]


def get_metrics(deps: Deps) -> dict:
    """Performances du modèle XGBoost (PR-AUC, ROC-AUC)"""
    return deps.metrics["xgboost"]


def feature_importance(deps: Deps, top_n: int = 6) -> dict:
    """Importance globale (SHAP) des top_n features"""
    sample = deps.X.sample(min(500, len(deps.X)), random_state=0)
    imp = global_importance(deps.pipe, sample)
    return {_clean(k): round(float(v), 3) for k, v in list(imp.items())[:top_n]}


def predict_risk(deps: Deps, machine_id: int) -> float:
    """Probabilité de panne d'une machine (par son index)"""
    return round(float(deps.scored.loc[machine_id, "risk"]), 4)


def explain_prediction(deps: Deps, machine_id: int, top_n: int = 5) -> dict:
    """Contributions SHAP locales (top_n) expliquant le risque d'une machine"""
    contrib = explain_row(deps.pipe, deps.X.loc[[machine_id]])
    top = sorted(contrib.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_n]
    return {_clean(k): round(float(v), 3) for k, v in top}


def whatif(deps: Deps, machine_id: int, overrides: dict) -> dict:
    """Re-score une machine après modification de variables brutes (scénario what-if)

    Tolère les noms SHAP ('num__torque_nm') ; n'applique que les colonnes connues.
    """
    row = deps.X.loc[[machine_id]].copy()
    # before/after : meme modèle de service + meme calibrateur -> probabilités comparables.
    before = float(deps.calibrator.predict(deps.pipe.predict_proba(row)[:, 1])[0])
    modes_before = deps.kg.link_modes(row.iloc[0].to_dict())
    applied, out_of_domain = {}, []
    for key, value in overrides.items():
        col = _clean(key)
        if col not in deps.X.columns:
            continue
        row[col] = value
        applied[col] = value
        lo, hi = float(deps.X[col].min()), float(deps.X[col].max())
        if value < lo or value > hi:
            out_of_domain.append(
                {
                    "feature": col,
                    "value": value,
                    "train_min": round(lo, 2),
                    "train_max": round(hi, 2),
                }
            )
    after = float(deps.calibrator.predict(deps.pipe.predict_proba(row)[:, 1])[0])
    # Modes FMEA recalculés sur la ligne modifiée
    modes_after = deps.kg.link_modes(row.iloc[0].to_dict())
    result = {
        "risk_before": round(before, 4),
        "risk_after": round(after, 4),
        "applied": applied,
        "modes_before": modes_before,
        "modes_after": modes_after,
    }
    if out_of_domain:
        result["warning"] = (
            "Valeur(s) hors du domaine d'entrainement : prédiction extrapolée, peu fiable."
        )
        result["out_of_domain"] = out_of_domain
    return result


def top_risk_machines(deps: Deps, n: int = 5) -> list[dict]:
    """Les n machines les plus à risque (index + score)"""
    top = deps.scored.sort_values("risk", ascending=False).head(n)
    return [{"machine_id": int(i), "risk": round(float(r), 4)} for i, r in top["risk"].items()]


def dataset_summary(deps: Deps) -> dict:
    """Volumétrie et taux de panne observé"""
    return {
        "n_machines": int(len(deps.scored)),
        "failure_rate": round(float(deps.scored["machine_failure"].mean()), 4),
    }
