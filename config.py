# config.py
"""Central configuration for the benchmark pipeline.
Change a setting here and it applies to every model."""

import os

# --- Reproducibility ---
SEED = 42

# --- Data ---
DATA_DIR = r"D:\hf_cache\hub\datasets--MLCommons--peoples_speech\snapshots\f10597c5d3d3a63f8b6827701297c3afdf178272\clean"
MAX_ROWS = 2000

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
