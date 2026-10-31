"""chemins, constantes de repro et schéma des colonnes"""

from pathlib import Path

# Chemins (calculés depuis l'emplacement du fichier)
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
RAW_CSV = DATA_DIR / "ai4i2020.csv"

# Source des données : AI4I 2020
UCI_URL = (
    "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip"
)
CSV_IN_ZIP = "ai4i2020.csv"

# Reproductibilité
SEED = 42
TEST_SIZE = 0.2

# Schéma des colonnes
TARGET = "machine_failure"
ID_COLS = ["UDI", "Product ID"]
FAILURE_MODE_COLS = ["TWF", "HDF", "PWF", "OSF", "RNF"]

# Renommage des colonnes
RENAME_MAP = {
    "Type": "type",
    "Air temperature [K]": "air_temp_k",
    "Process temperature [K]": "process_temp_k",
    "Rotational speed [rpm]": "rotational_speed_rpm",
    "Torque [Nm]": "torque_nm",
    "Tool wear [min]": "tool_wear_min",
    "Machine failure": "machine_failure",
}

NUMERIC_FEATURES = [
    "air_temp_k",
    "process_temp_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
]
CATEGORICAL_FEATURES = ["type"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
