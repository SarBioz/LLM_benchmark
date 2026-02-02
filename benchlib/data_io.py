# benchlib/data_io.py
import glob
import os
from datasets import load_dataset
import pandas as pd


def load_people_speech_parquet(root_clean_dir: str):
    root = os.path.normpath(root_clean_dir)
    train_files = sorted(glob.glob(os.path.join(root, "train-*.parquet")))
    if not train_files:
        raise FileNotFoundError(f"No train-*.parquet found in: {root}")
    print("Found train shards:", len(train_files))

    ds = load_dataset("parquet", data_files={"train": train_files})["train"]
    print(ds)
    print("Columns:", ds.column_names)
    return ds


def dataset_to_transcripts_df(ds, id_col="id", text_col="text", max_rows=None) -> pd.DataFrame:
    # Keep only what we need; NO preprocessing
    ds = ds.select_columns([id_col, text_col])

    if max_rows is not None:
        max_rows = min(max_rows, len(ds))
        ds = ds.select(range(max_rows))

    df = ds.to_pandas()
    df = df.rename(columns={id_col: "participant_id", text_col: "text"})
    df["chunk_id"] = 0
    return df
