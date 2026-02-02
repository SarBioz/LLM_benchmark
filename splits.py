# splits.py
import numpy as np
import pandas as pd


def make_participant_splits(
    df: pd.DataFrame,
    participant_col: str,
    split_col: str = "split",
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6

    rng = np.random.default_rng(seed)

    participants = df[participant_col].unique()
    rng.shuffle(participants)

    n = len(participants)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_ids = participants[:n_train]
    val_ids = participants[n_train:n_train + n_val]
    test_ids = participants[n_train + n_val:]

    split_map = {}
    for pid in train_ids:
        split_map[pid] = "train"
    for pid in val_ids:
        split_map[pid] = "val"
    for pid in test_ids:
        split_map[pid] = "test"

    return pd.DataFrame({
        participant_col: participants,
        split_col: [split_map[p] for p in participants],
    })


def attach_splits(
    df: pd.DataFrame,
    split_df: pd.DataFrame,
    participant_col: str,
    split_col: str = "split",
) -> pd.DataFrame:
    df = df.copy()
    return df.merge(
        split_df,
        on=participant_col,
        how="left",
        validate="many_to_one",
    )
