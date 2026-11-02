"""Génère TOUS les chiffres cités dans le README (reproductibilité)."""

import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src import config
from src.agent.knowledge_graph import KnowledgeGraph
from src.data import load_data, load_failure_modes, split_data
from src.features import build_pipeline
from src.model import _scale_pos_weight, _xgb

SEED = config.SEED


def _bootstrap_pr_auc(y, proba, n: int = 1000) -> tuple[float, float]:
    """IC 95 % de la PR-AUC par ré-échantillonnage (percentiles 2,5 et 97,5)."""
    rng = np.random.default_rng(SEED)
    y, proba = np.asarray(y), np.asarray(proba)
    idx = np.arange(len(y))
    scores = []
    for _ in range(n):
        s = rng.choice(idx, size=len(idx), replace=True)
        if y[s].sum() == 0:  # pas de positif dans le tirage -> PR-AUC indéfinie
            continue
        scores.append(average_precision_score(y[s], proba[s]))
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def model_metrics(df, best_params) -> dict:
    """PR-AUC test (+ IC bootstrap), CV 5-fold (moyenne/écart), split temporel par ordre."""
    X_train, X_test, y_train, y_test = split_data(df)
    pipe = build_pipeline(
        _xgb({**best_params, "n_jobs": -1}, _scale_pos_weight(y_train)), scale=False
    )
    pipe.fit(X_train, y_train)
    proba_test = pipe.predict_proba(X_test)[:, 1]
    pr_test = average_precision_score(y_test, proba_test)
    ci_lo, ci_hi = _bootstrap_pr_auc(y_test, proba_test)

    X, y = df.drop(columns=[config.TARGET]), df[config.TARGET].to_numpy()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = []
    for tr, te in cv.split(X, y):
        p = build_pipeline(
            _xgb({**best_params, "n_jobs": -1}, _scale_pos_weight(y[tr])), scale=False
        )
        p.fit(X.iloc[tr], y[tr])
        folds.append(float(average_precision_score(y[te], p.predict_proba(X.iloc[te])[:, 1])))

    cut = int(0.8 * len(X))  # split temporel 
    pt = build_pipeline(
        _xgb({**best_params, "n_jobs": -1}, _scale_pos_weight(y[:cut])), scale=False
    )
    pt.fit(X.iloc[:cut], y[:cut])
    proba_tmp = pt.predict_proba(X.iloc[cut:])[:, 1]

    return {
        "pr_auc_test": round(pr_test, 3),
        "pr_auc_test_ci95": [round(ci_lo, 3), round(ci_hi, 3)],
        "roc_auc_test": round(roc_auc_score(y_test, proba_test), 3),
        "n_pos_test": int(y_test.sum()),
        "pr_auc_cv_mean": round(float(np.mean(folds)), 3),
        "pr_auc_cv_std": round(float(np.std(folds)), 3),
        "pr_auc_cv_folds": [round(f, 3) for f in folds],
        "pr_auc_temporal": round(average_precision_score(y[cut:], proba_tmp), 3),
        "roc_auc_temporal": round(roc_auc_score(y[cut:], proba_tmp), 3),
        "n_pos_temporal": int(y[cut:].sum()),
    }


def _pr(y, pred) -> dict:
    return {
        "precision": round(precision_score(y, pred, zero_division=0), 3),
        "recall": round(recall_score(y, pred, zero_division=0), 3),
        "alerts": int(pred.sum()),
    }


def baseline_comparison(df, threshold: float) -> dict:
    """XGBoost (OOF, seuil coût) vs règles FMEA (toutes), vs règles fortes (sans TWF)."""
    X, y = df.drop(columns=[config.TARGET]), df[config.TARGET].to_numpy()
    kg = KnowledgeGraph()
    modes = X.apply(lambda r: kg.link_modes(r.to_dict()), axis=1)
    rules_all = modes.apply(lambda m: int(len(m) > 0)).to_numpy()
    rules_no_twf = modes.apply(lambda m: int(bool(set(m) - {"TWF"}))).to_numpy()

    oof = pd.read_csv(config.MODELS_DIR / "oof_scores.csv", index_col="machine_id")[
        "risk"
    ].to_numpy()
    xgb_pred = (oof >= threshold).astype(int)

    truth = load_failure_modes()
    per_mode = {}
    for m in config.FAILURE_MODE_COLS:
        mask = truth[m].to_numpy() == 1
        if mask.sum() == 0:
            continue
        per_mode[m] = {
            "n": int(mask.sum()),
            "rules": round(float(rules_all[mask].mean()), 2),
            "xgb": round(float(xgb_pred[mask].mean()), 2),
        }

    return {
        "xgboost_at_threshold": _pr(y, xgb_pred),
        "rules_all_modes": _pr(y, rules_all),
        "rules_no_twf": _pr(y, rules_no_twf),
        "recall_per_mode": per_mode,
    }


def run() -> dict:
    df = load_data()
    meta = json.loads((config.MODELS_DIR / "metrics.json").read_text())
    report = {
        "model": model_metrics(df, meta["best_params"]),
        "threshold": meta["threshold"],
        "baseline": baseline_comparison(df, meta["threshold"]),
    }
    (config.MODELS_DIR / "report_metrics.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, ensure_ascii=False))
    print("\nÉcrit : models/report_metrics.json")
