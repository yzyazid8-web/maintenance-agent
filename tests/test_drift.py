"""Tests de la détection de dérive"""

import numpy as np

from src import config
from src.data import load_data
from src.drift import (
    categorical_drift,
    drift_report,
    drift_verdict,
    psi,
    simulate_drift,
    wasserstein_norm,
)


def test_drift_verdict_thresholds():
    assert drift_verdict(0.50) == "🟢 faible"
    assert drift_verdict(0.70) == "🟡 modérée"
    assert drift_verdict(0.90) == "🔴 forte"


def test_categorical_drift():
    ref = np.array(["L"] * 50 + ["M"] * 30 + ["H"] * 20)
    assert not categorical_drift(ref, ref.copy())["drifted"]  # distributions identiques
    shifted = np.array(["H"] * 90 + ["L"] * 10)
    assert categorical_drift(ref, shifted)["drifted"]  # proportions très différentes


def test_psi_zero_on_identical():
    x = np.random.default_rng(0).normal(size=2000)
    assert psi(x, x) < 1e-6


def test_psi_and_wasserstein_high_on_shift():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 2000)
    cur = rng.normal(2, 1, 2000)
    assert psi(ref, cur) > 0.25
    assert wasserstein_norm(ref, cur) > 0.5


def test_psi_counts_values_outside_reference_range():
    # Production entièrement au-delà de la plage de référence : sans bornes ouvertes,
    # np.histogram ignorerait toutes ces valeurs et le PSI serait faussement bas.
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 2000)
    cur = rng.normal(0, 1, 2000) + 50  # décalage hors plage de ref
    assert psi(ref, cur) > 1.0


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
