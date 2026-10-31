"""Tests de la détection de dérive"""

import numpy as np

from src import config
from src.data import load_data
from src.drift import drift_report, psi, simulate_drift, wasserstein_norm


def test_psi_zero_on_identical():
    x = np.random.default_rng(0).normal(size=2000)
    assert psi(x, x) < 1e-6


def test_psi_and_wasserstein_high_on_shift():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 2000)
    cur = rng.normal(2, 1, 2000)
    assert psi(ref, cur) > 0.25
    assert wasserstein_norm(ref, cur) > 0.5


def test_simulate_drift_shifts_distributions():
    df = load_data().head(1000)
    drifted = simulate_drift(df, severity=1.0)
    assert drifted["tool_wear_min"].mean() > df["tool_wear_min"].mean()
    assert drifted["torque_nm"].mean() > df["torque_nm"].mean()


def test_drift_report_detects_drift():
    df = load_data()
    clean = drift_report(df, df, config.NUMERIC_FEATURES)
    drifted = drift_report(df, simulate_drift(df, severity=1.5), config.NUMERIC_FEATURES)

    # AUC proche de 0,5 sans dérive, nettement plus haute sinon
    assert clean["domain_classifier_auc"] < 0.6
    assert drifted["domain_classifier_auc"] > 0.7
    # au moins une variable passe en 🔴 quand ça dérive
    assert any(r["status"] == "🔴" for r in drifted["per_feature"])
