"""Outils de l'agent : fonctions pures opérant sur le modèle et les données via Deps"""

import json
from dataclasses import dataclass

import joblib
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline

from src import config
from src.agent.knowledge_graph import KnowledgeGraph
from src.agent.memory import EpisodicMemory
from src.calibration import PlattCalibrator
from src.data import load_data
from src.explain import explain_row, global_importance


@dataclass
class Deps:
    """État partagé injecté dans les outils de l'agent"""

    pipe: Pipeline  # modèle final (vue globale : importance SHAP, scoring nouvelles données)
    X: pd.DataFrame  # features (sans la cible)
    scored: pd.DataFrame  # données + colonnes `risk`, `risk_raw`, `fold`
    metrics: dict
    kg: KnowledgeGraph  # graphe FMEA (modes -> causes / actions)
    memory: EpisodicMemory  # mémoire
    calibrator: PlattCalibrator  # score brut -> probabilité calibrée
    fold_models: list[Pipeline]  # 5 modèles de fold (ligne i scorée par celui qui ne l'a pas vue)


def build_deps() -> Deps:
    """Charge le pipeline final, les modèles de fold, le calibrateur et le risque OOF."""
    pipe = joblib.load(config.MODELS_DIR / "pipeline.joblib")
    fold_models = joblib.load(config.MODELS_DIR / "fold_models.joblib")
    calibrator = joblib.load(config.MODELS_DIR / "calibrator.joblib")
    metrics = json.loads((config.MODELS_DIR / "metrics.json").read_text())
    df = load_data()
    X = df.drop(columns=[config.TARGET])
    # Risque affiché = prédictions out-of-fold calibrées (+ score brut + fold d'appartenance)
    oof = pd.read_csv(config.MODELS_DIR / "oof_scores.csv", index_col="machine_id")
    scored = df.assign(
        risk=oof["risk"].to_numpy(),
        risk_raw=oof["raw"].to_numpy(),
        fold=oof["fold"].astype(int).to_numpy(),
    )
    return Deps(
        pipe=pipe,
        X=X,
        scored=scored,
        metrics=metrics,
        kg=KnowledgeGraph(),
        memory=EpisodicMemory(),
        calibrator=calibrator,
        fold_models=fold_models,
    )


def _clean(name: str) -> str:
    """Retire le préfixe du préprocesseur ('num__torque_nm' -> 'torque_nm')"""
    return name.split("__", 1)[-1]


def _model_for(deps: Deps, machine_id: int) -> Pipeline:
    """Le modèle du fold qui n'a PAS vu cette ligne (risque/SHAP/what-if cohérents, hors-échantillon)."""
    return deps.fold_models[int(deps.scored.loc[machine_id, "fold"])]


def get_metrics(deps: Deps) -> dict:

    m = deps.metrics["xgboost"]
    served = float(average_precision_score(deps.scored[config.TARGET], deps.scored["risk"]))
    out: dict[str, object] = {
        "pr_auc_servi_oof": round(served, 3),  # ce que voit l'utilisateur (hors-échantillon)
        "pr_auc_test": round(float(m["pr_auc"]), 3),
        "roc_auc_test": round(float(m["roc_auc"]), 3),
    }
    rp = config.MODELS_DIR / "report_metrics.json"
    if rp.exists():
        mo = json.loads(rp.read_text())["model"]
        out["pr_auc_cv"] = f"{mo['pr_auc_cv_mean']} ± {mo['pr_auc_cv_std']}"
        out["pr_auc_ic95_test"] = mo["pr_auc_test_ci95"]
        out["pr_auc_ordre_arrivee"] = mo["pr_auc_temporal"]  # proxy déploiement temporel
    return out


def feature_importance(deps: Deps, top_n: int = 6) -> dict:
    """Importance globale (SHAP) des top_n features"""
    sample = deps.X.sample(min(500, len(deps.X)), random_state=0)
    imp = global_importance(deps.pipe, sample)
    return {_clean(k): round(float(v), 3) for k, v in list(imp.items())[:top_n]}


def predict_risk(deps: Deps, machine_id: int) -> float:
    """Probabilité de panne d'un relevé (par son index)"""
    return round(float(deps.scored.loc[machine_id, "risk"]), 4)


def explain_prediction(deps: Deps, machine_id: int, top_n: int = 5) -> dict:
    """Contributions SHAP locales (top_n) - sur le modèle de fold qui a produit le risque affiché"""
    contrib = explain_row(_model_for(deps, machine_id), deps.X.loc[[machine_id]])
    top = sorted(contrib.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_n]
    return {_clean(k): round(float(v), 3) for k, v in top}


def whatif(deps: Deps, machine_id: int, overrides: dict) -> dict:
    """Re-score une machine après modification de variables brutes (scénario what-if)"""
    model = _model_for(deps, machine_id)
    row = deps.X.loc[[machine_id]].copy()
    risk_before = float(deps.calibrator.predict(model.predict_proba(row)[:, 1])[0])
    modes_before = deps.kg.link_modes(row.iloc[0].to_dict())
    applied, out_of_domain = {}, []
    for key, value in overrides.items():
        col = _clean(key)
        if col not in config.NUMERIC_FEATURES:  # variables numériques seulement
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
    risk_after = float(deps.calibrator.predict(model.predict_proba(row)[:, 1])[0])
    # Modes FMEA recalculés sur la ligne modifiée
    modes_after = deps.kg.link_modes(row.iloc[0].to_dict())
    result = {
        "risk_before": round(risk_before, 4),
        "risk_after": round(risk_after, 4),
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
    # Tri par risque calibré, départage par score brut -> ordre déterministe même en cas d'ex-aequo.
    top = deps.scored.sort_values(["risk", "risk_raw"], ascending=False).head(n)
    return [{"machine_id": int(i), "risk": round(float(r), 4)} for i, r in top["risk"].items()]


def dataset_summary(deps: Deps) -> dict:
    """Volumétrie (relevés) et taux de panne observé"""
    return {
        "n_releves": int(len(deps.scored)),
        "failure_rate": round(float(deps.scored["machine_failure"].mean()), 4),
    }
