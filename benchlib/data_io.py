# benchlib/data_io.py
import glob
import os
import pandas as pd


# =============================================================================
# NEW: Load text files from labeled folders (Normal/, MCI/)
# Use this for real data
# =============================================================================

def load_text_files_from_folders(data_dir: str, max_rows: int = None) -> pd.DataFrame:
    """
    Load text files from Normal/ and MCI/ subfolders.

    Expected folder structure:
        data_dir/
            Normal/
                file1.txt
                file2.txt
                ...
            MCI/
                file1.txt
                file2.txt
                ...

    Args:
        data_dir: Path to the folder containing Normal/ and MCI/ subfolders
        max_rows: Optional limit on total number of samples to load

    Returns:
        DataFrame with columns: participant_id, text, label, chunk_id
    """
    records = []

    # Load Normal folder → label "NC"
    normal_dir = os.path.join(data_dir, "Normal")
    if os.path.exists(normal_dir):
        for filename in os.listdir(normal_dir):
            if filename.endswith(".txt"):
                filepath = os.path.join(normal_dir, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    text = f.read()
                records.append({
                    "participant_id": filename.replace(".txt", ""),
                    "text": text,
                    "label": "NC"
                })
        print(f"Loaded {len([r for r in records if r['label'] == 'NC'])} files from Normal/")
    else:
        print(f"Warning: Normal folder not found at {normal_dir}")

    # Load MCI folder → label "MCI"
    mci_dir = os.path.join(data_dir, "MCI")
    if os.path.exists(mci_dir):
        mci_count_before = len(records)
        for filename in os.listdir(mci_dir):
            if filename.endswith(".txt"):
                filepath = os.path.join(mci_dir, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    text = f.read()
                records.append({
                    "participant_id": filename.replace(".txt", ""),
                    "text": text,
                    "label": "MCI"
                })
        print(f"Loaded {len(records) - mci_count_before} files from MCI/")
    else:
        print(f"Warning: MCI folder not found at {mci_dir}")

    if not records:
        raise FileNotFoundError(f"No .txt files found in {data_dir}/Normal or {data_dir}/MCI")

    df = pd.DataFrame(records)
    df["chunk_id"] = 0

    # Apply max_rows limit if specified
    if max_rows is not None and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=42).reset_index(drop=True)
        print(f"Limited to {max_rows} samples")

    print(f"Total samples: {len(df)} (NC={len(df[df['label']=='NC'])}, MCI={len(df[df['label']=='MCI'])})")
    return df


# =============================================================================
# OLD CODE: For dummy data only (parquet files with random labels)
# Keep for reference - used with People's Speech dataset for testing
# =============================================================================

from datasets import load_dataset

def load_people_speech_parquet(root_clean_dir: str):
    """
    [DUMMY DATA ONLY] Load People's Speech parquet files for testing pipeline.
    Labels are randomly assigned - not real cognitive labels.
    """
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
    """
    [DUMMY DATA ONLY] Convert HuggingFace dataset to DataFrame.
    Used with People's Speech dataset for testing pipeline.
    """
    # Keep only what we need; NO preprocessing
    ds = ds.select_columns([id_col, text_col])

    if max_rows is not None:
        max_rows = min(max_rows, len(ds))
        ds = ds.select(range(max_rows))

    df = ds.to_pandas()
    df = df.rename(columns={id_col: "participant_id", text_col: "text"})
    df["chunk_id"] = 0
    return df
