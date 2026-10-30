"""Tests du moteur ML. n_trials réduit pour rester rapide en CI"""

from src.data import load_data, split_data
from src.explain import explain_row, global_importance
from src.model import train_baseline, train_xgboost


def test_baseline_and_xgboost():
    X_tr, X_te, y_tr, y_te = split_data(load_data())
    _, base = train_baseline(X_tr, y_tr, X_te, y_te)
    pipe, xgb, params, cv_pr_auc = train_xgboost(X_tr, y_tr, X_te, y_te, n_trials=2)

    assert 0.0 <= base["pr_auc"] <= 1.0
    assert xgb["roc_auc"] > 0.9 

    assert xgb["pr_auc"] > base["pr_auc"]

    # SHAP : importances globales + explication locale 
    imp = global_importance(pipe, X_te.head(200))
    assert len(imp) > 0
    local = explain_row(pipe, X_te.head(1))
    assert set(local) == set(imp)
