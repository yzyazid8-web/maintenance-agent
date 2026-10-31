"""Téléchargement, nettoyage et découpage du dataset AI4I 2020."""

import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src import config


def download_data(force: bool = False) -> Path:
    """Télécharge le dataset depuis UCI et le met en cache dans data/.
    Ne retélécharge pas si le fichier existe déjà.
    """
    if config.RAW_CSV.exists() and not force:
        return config.RAW_CSV

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(config.UCI_URL) as resp:
        archive = resp.read()
    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        config.RAW_CSV.write_bytes(zf.read(config.CSV_IN_ZIP))
    return config.RAW_CSV


def load_raw() -> pd.DataFrame:
    """Lit le CSV. Télécharge au besoin."""
    if not config.RAW_CSV.exists():
        download_data()
    return pd.read_csv(config.RAW_CSV, encoding="utf-8-sig")


def load_data() -> pd.DataFrame:
    """DataFrame de modélisation : features process + type + cible, colonnes nettoyées."""
    raw = load_raw()
    df = raw.drop(columns=config.ID_COLS + config.FAILURE_MODE_COLS)
    return df.rename(columns=config.RENAME_MAP)


def load_failure_modes() -> pd.DataFrame:
    """Les 5 modes de panne, mêmes index que load_data()."""
    return load_raw()[config.FAILURE_MODE_COLS]


def split_data(df: pd.DataFrame):
    """Découpe train/test stratifiée sur la cible.

    Retourne X_train, X_test, y_train, y_test.
    """
    X = df.drop(columns=[config.TARGET])
    y = df[config.TARGET]
    return train_test_split(X, y, test_size=config.TEST_SIZE, random_state=config.SEED, stratify=y)
