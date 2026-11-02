"""Entrainement : baseline logistique + XGBoost (Optuna + CV stratifiée), MLflow, artefacts.

Lancer un entraînement complet :  python -m src.model
"""

import json
from datetime import UTC, datetime

import joblib
import mlflow
import numpy as np
import optuna
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from xgboost import XGBClassifier

from src import config
from src.calibration import PlattCalibrator
from src.data import load_data, split_data
from src.features import build_pipeline

COST_FN, COST_FP = 10.0, 1.0
PROMOTE_EPS = 0.005  # on ne remplace pas le champion par un challenger nettement moins bon

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _champion_pr_auc() -> float | None:
    """PR-AUC du modèle actuellement servi (None si aucun artefact)."""
    path = config.MODELS_DIR / "metrics.json"
    if not path.exists():
        return None
    try:
        return float(json.loads(path.read_text())["xgboost"]["pr_auc"])
    except (KeyError, ValueError):
        return None


def _should_promote(new_pr: float, champion_pr: float | None, eps: float = PROMOTE_EPS) -> bool:
    """Promeut le challenger s'il n'est pas nettement moins bon que le champion (PR-AUC)."""
    return champion_pr is None or new_pr >= champion_pr - eps


def _log_history(n_trials: int, decision: dict) -> None:
    """Trace chaque entraînement et sa décision de promotion (journal append-only)."""
    rec = {"at": datetime.now(UTC).isoformat(timespec="seconds"), "n_trials": n_trials, **decision}
    with open(config.MODELS_DIR / "history.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def _scale_pos_weight(y) -> float:
    """Ratio négatifs / positifs, pour pondérer la classe rare dans XGBoost."""
    pos = int(y.sum())
    return (len(y) - pos) / pos


def _evaluate(pipe, X_test, y_test) -> dict:
    proba = pipe.predict_proba(X_test)[:, 1]
    return {
        "pr_auc": float(average_precision_score(y_test, proba)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
    }


def train_baseline(X_train, y_train, X_test, y_test):
    """Régression logistique (features standardisées, classes rééquilibrées) - référence faible."""
    pipe = build_pipeline(LogisticRegression(max_iter=1000, class_weight="balanced"), scale=True)
    pipe.fit(X_train, y_train)
    return pipe, _evaluate(pipe, X_test, y_test)


def _xgb(params, spw) -> XGBClassifier:
    return XGBClassifier(
        **params,
        scale_pos_weight=spw,
        eval_metric="aucpr",
        tree_method="hist",
        random_state=config.SEED,
    )


def _objective(trial, X_train, y_train, spw) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }
    pipe = build_pipeline(_xgb({**params, "n_jobs": 1}, spw), scale=False)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.SEED)
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="average_precision", n_jobs=-1)
    return float(scores.mean())


def train_xgboost(X_train, y_train, X_test, y_test, n_trials: int = 30):
    """Recherche Optuna évaluée par PR-AUC en CV stratifiée 5-fold, puis réentrainement final."""
    spw = _scale_pos_weight(y_train)
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=config.SEED)
    )
    study.optimize(lambda t: _objective(t, X_train, y_train, spw), n_trials=n_trials)

    pipe = build_pipeline(_xgb({**study.best_params, "n_jobs": -1}, spw), scale=False)
    pipe.fit(X_train, y_train)
    return pipe, _evaluate(pipe, X_test, y_test), study.best_params, float(study.best_value)


def _oof_scores(df, best_params) -> pd.DataFrame:
    """Risque out-of-fold sur tout le dataset"""
    X = df.drop(columns=[config.TARGET])
    y = df[config.TARGET]
    pipe = build_pipeline(_xgb({**best_params, "n_jobs": 1}, _scale_pos_weight(y)), scale=False)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.SEED)
    oof = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    return pd.DataFrame({"risk": oof}, index=df.index)


def _optimal_threshold(y, proba, cost_fn: float = COST_FN, cost_fp: float = COST_FP) -> float:
    """Seuil qui minimise le coût attendu (cost_fn * faux négatifs + cost_fp * faux positifs)"""
    y = np.asarray(y)
    best_t, best_cost = 0.5, float("inf")
    for t in np.linspace(0.01, 0.99, 99):
        pred = proba >= t
        fn = int(((y == 1) & ~pred).sum())
        fp = int(((y == 0) & pred).sum())
        cost = cost_fn * fn + cost_fp * fp
        if cost < best_cost:
            best_cost, best_t = cost, float(t)
    return best_t


def run(n_trials: int = 30) -> dict:
    """Entraîne baseline + XGBoost, logge dans MLflow, sauvegarde le pipeline et les métriques."""
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    mlflow.set_experiment("maintenance-agent")

    with mlflow.start_run(run_name="baseline-logreg"):
        _, base_metrics = train_baseline(X_train, y_train, X_test, y_test)
        mlflow.log_param("model", "logistic_regression")
        mlflow.log_metrics(base_metrics)

    with mlflow.start_run(run_name="xgboost-optuna"):
        pipe, xgb_metrics, best_params, cv_pr_auc = train_xgboost(
            X_train, y_train, X_test, y_test, n_trials
        )
        mlflow.log_param("model", "xgboost")
        mlflow.log_params(best_params)
        mlflow.log_metric("cv_pr_auc", cv_pr_auc)
        mlflow.log_metrics(xgb_metrics)

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Champion / challenger : on ne remplace les artefacts QUE si le nouveau modèle tient la route.
    champion_pr = _champion_pr_auc()
    new_pr = xgb_metrics["pr_auc"]
    promoted = _should_promote(new_pr, champion_pr)
    decision = {
        "promoted": promoted,
        "challenger_pr_auc": round(new_pr, 4),
        "champion_pr_auc": round(champion_pr, 4) if champion_pr is not None else None,
    }

    if promoted:
        joblib.dump(pipe, config.MODELS_DIR / "pipeline.joblib")

        # Risque out-of-fold brut, puis calibration sigmoïde (Platt).
        y_all = df[config.TARGET].to_numpy()
        raw_oof = _oof_scores(df, best_params)["risk"].to_numpy()

        # Affichage / seuil / courbe : calibration CROSS-FIT (non circulaire).
        cal_oof = np.zeros_like(raw_oof, dtype=float)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.SEED)
        for tr, te in skf.split(raw_oof.reshape(-1, 1), y_all):
            cal_oof[te] = PlattCalibrator().fit(raw_oof[tr], y_all[tr]).predict(raw_oof[te])

        calibrator = PlattCalibrator().fit(raw_oof, y_all)  # calibrateur servi (what-if)
        joblib.dump(calibrator, config.MODELS_DIR / "calibrator.joblib")
        pd.DataFrame({"risk": cal_oof, "raw": raw_oof}, index=df.index).to_csv(
            config.MODELS_DIR / "oof_scores.csv", index_label="machine_id"
        )
        results = {
            "baseline": base_metrics,
            "xgboost": xgb_metrics,
            "best_params": best_params,
            "threshold": round(_optimal_threshold(y_all, cal_oof), 3),
            "cost_fn": COST_FN,
            "cost_fp": COST_FP,
        }
        (config.MODELS_DIR / "metrics.json").write_text(json.dumps(results, indent=2))
    else:
        # Challenger moins bon : on CONSERVE le champion (pas d'écrasement) et on renvoie ses métriques.
        results = json.loads((config.MODELS_DIR / "metrics.json").read_text())

    _log_history(n_trials, decision)
    return {**results, "promotion": decision}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
