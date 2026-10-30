"""Tests du feature engineering et du pipeline"""

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.data import load_data, split_data
from src.features import ENGINEERED, PhysicsFeatures, build_pipeline


def test_physics_features_added():
    df = load_data().head(100)
    out = PhysicsFeatures().fit_transform(df)
    assert all(col in out.columns for col in ENGINEERED)
    row = df.iloc[0]
    expected_power = row["torque_nm"] * row["rotational_speed_rpm"] * 2 * np.pi / 60
    assert abs(out.iloc[0]["power_w"] - expected_power) < 1e-6


def test_pipeline_fits_and_predicts():
    X_tr, X_te, y_tr, y_te = split_data(load_data())
    pipe = build_pipeline(LogisticRegression(max_iter=500, class_weight="balanced"), scale=True)
    pipe.fit(X_tr, y_tr)
    proba = pipe.predict_proba(X_te)[:, 1]
    assert proba.shape[0] == len(y_te)
    assert ((proba >= 0) & (proba <= 1)).all()
