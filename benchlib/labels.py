# benchlib/labels.py
import numpy as np
import pandas as pd


# =============================================================================
# NEW: For real data (NC and MCI labels from folder structure)
# =============================================================================

def make_binary_label(
    df: pd.DataFrame,
    src_label_col: str,
    out_col: str = "y",
    normal_label: str = "NC",
    abnormal_labels=("MCI", "QCI", "Dementia"),  # All impaired categories
) -> pd.DataFrame:
    """
    Convert labels to binary (0/1).
    NC = 0 (Normal)
    MCI, QCI, Dementia = 1 (Impaired)
    """
    df = df.copy()

    def map_label(x):
        if x == normal_label:
            return 0
        if x in abnormal_labels:
            return 1
        raise ValueError(f"Unknown label: {x}")

    df[out_col] = df[src_label_col].apply(map_label)
    return df


# =============================================================================
# OLD CODE: For dummy data only (random label assignment)
# Used with People's Speech dataset for testing pipeline
# =============================================================================

# def assign_random_labels_per_participant(
#     df: pd.DataFrame,
#     participant_col: str,
#     label_col: str,
#     labels=("NC", "CIND", "AD"),
#     seed: int = 42,
# ) -> pd.DataFrame:
#     """[DUMMY DATA ONLY] Randomly assign NC/CIND/AD labels to participants."""
#     rng = np.random.default_rng(seed)
#
#     participants = df[participant_col].unique()
#     label_map = {
#         pid: rng.choice(labels)
#         for pid in participants
#     }
#
#     df = df.copy()
#     df[label_col] = df[participant_col].map(label_map)
#     return df
#
#
# def save_participant_label_map(
#     df: pd.DataFrame,
#     participant_col: str,
#     label_col: str,
#     out_path: str,
# ):
#     """[DUMMY DATA ONLY] Save participant-to-label mapping to CSV."""
#     (
#         df[[participant_col, label_col]]
#         .drop_duplicates()
#         .sort_values(participant_col)
#         .to_csv(out_path, index=False)
#     )
#
#
# def make_binary_label_dummy(
#     df: pd.DataFrame,
#     src_label_col: str,
#     out_col: str = "y",
#     normal_label: str = "NC",
#     abnormal_labels=("CIND", "AD"),
# ) -> pd.DataFrame:
#     """[DUMMY DATA ONLY] Convert NC/CIND/AD labels to binary (0/1)."""
#     df = df.copy()
#
#     def map_label(x):
#         if x == normal_label:
#             return 0
#         if x in abnormal_labels:
#             return 1
#         raise ValueError(f"Unknown label: {x}")
#
#     df[out_col] = df[src_label_col].apply(map_label)
#     return df
