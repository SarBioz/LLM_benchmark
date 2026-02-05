# config.py
"""Central configuration for the benchmark pipeline.
Change a setting here and it applies to every model."""

import os

# --- Reproducibility ---
SEED = 42

# --- Data ---
# NEW: Path to folder containing Normal/ and MCI/ subfolders with .txt files
# For Mac:
DATA_DIR = "/Users/sajjadilab/Desktop/adrc/ADRC Audio+Text by 1st Language/Manually Checked Text (AD)"
# For Windows (update path as needed):
# DATA_DIR = r"D:\path\to\your\data"

# OLD (dummy data): People's Speech parquet files
# DATA_DIR = r"D:\hf_cache\hub\datasets--MLCommons--peoples_speech\snapshots\f10597c5d3d3a63f8b6827701297c3afdf178272\clean"

MAX_ROWS = None  # Set to None to use all data, or a number to limit

# --- Splits (participant-level) ---
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15

# --- Feature extraction ---
MAX_LENGTH = 512
OVERLAP_TOKENS = 64
POOLING = "mean"

# --- Classifier (same for all models) ---
CLASSIFIER = "logreg"  # "logreg" | "linear_svm" | "mlp"

# --- Output ---
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "results")
FEATURES_DIR = os.path.join(OUTPUT_DIR, "features")
SAVE_FEATURES = True
