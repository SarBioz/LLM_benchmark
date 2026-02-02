# Label_binary.py
import pandas as pd


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
