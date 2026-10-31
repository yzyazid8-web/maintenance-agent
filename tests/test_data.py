"""Tests des données"""

from src import config
from src.data import load_data, load_failure_modes, split_data


def test_load_data_shape_and_columns():
    df = load_data()
    assert len(df) == 10_000
    assert config.TARGET in df.columns
    assert all(col in df.columns for col in config.FEATURES)
    # anti-fuite : les modes de panne ne doivent pas être des features
    assert all(col not in df.columns for col in config.FAILURE_MODE_COLS)


def test_target_binary_and_imbalanced():
    df = load_data()
    assert set(df[config.TARGET].unique()) <= {0, 1}
    assert 0.02 < df[config.TARGET].mean() < 0.05  # ~3,4 %


def test_failure_modes_aligned():
    modes = load_failure_modes()
    assert len(modes) == 10_000
    assert list(modes.columns) == config.FAILURE_MODE_COLS


def test_split_preserves_stratification():
    X_tr, X_te, y_tr, y_te = split_data(load_data())
    assert len(X_tr) + len(X_te) == 10_000
    assert abs(y_tr.mean() - y_te.mean()) < 0.01
