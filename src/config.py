from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
TRAIN_PARQUET = DATA_DIR / "train.parquet"
VALID_PARQUET = DATA_DIR / "validation.parquet"
TEST_PARQUET = DATA_DIR / "test.parquet"
EXPERIMENTS_DIR = ROOT_DIR / "experiments"
OOF_DIR = EXPERIMENTS_DIR / "oof"
SUB_DIR = EXPERIMENTS_DIR / "sub"

# Ensure dirs exist
OOF_DIR.mkdir(parents=True, exist_ok=True)
SUB_DIR.mkdir(parents=True, exist_ok=True)

# CV Settings
N_FOLDS = 5
GROUP_COL = "atco"
TARGET_COL = "target_index"
SAMPLE_ID_COL = "sample_id"
ROW_INDEX_COL = "row_index"
TASK_INDEX_COL = "task_index"
TASK_NAME_COL = "task_name"

RANDOM_SEED = 42

# Hardware: di default tutti i core; sovrascrivibile con AWARE_THREADS. AWARE_GPU=1 prova CatBoost su GPU.
import os
N_THREADS = int(os.environ.get("AWARE_THREADS", os.cpu_count() or 4))
USE_GPU = os.environ.get("AWARE_GPU", "0") == "1"

# Columns prohibited from features
DROP_COLS = [
    "sample_id",
    "seq_len",
    "row_index",
    "target_index",
    "session",
    "atco",
    "scenario",
    "timestamp",
    "task_name",
]

