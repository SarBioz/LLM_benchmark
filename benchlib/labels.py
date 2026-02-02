# benchlib/labels.py
import numpy as np
import pandas as pd


def assign_random_labels_per_participant(
    df: pd.DataFrame,
    participant_col: str,
    label_col: str,
    labels=("NC", "CIND", "AD"),
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    participants = df[participant_col].unique()
    label_map = {
        pid: rng.choice(labels)
        for pid in participants
    }

    df = df.copy()
    df[label_col] = df[participant_col].map(label_map)
    return df


def save_participant_label_map(
    df: pd.DataFrame,
    participant_col: str,
    label_col: str,
    out_path: str,
):
    (
        df[[participant_col, label_col]]
        .drop_duplicates()
        .sort_values(participant_col)
        .to_csv(out_path, index=False)
    )


def make_binary_label(
    df: pd.DataFrame,
    src_label_col: str,
    out_col: str = "y",
    normal_label: str = "NC",
    abnormal_labels=("CIND", "AD"),
) -> pd.DataFrame:
    df = df.copy()

    def map_label(x):
        if x == normal_label:
            return 0
        if x in abnormal_labels:
            return 1
        raise ValueError(f"Unknown label: {x}")

    df[out_col] = df[src_label_col].apply(map_label)
    return df
