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
    Load text files from Normal/, MCI/, QCI/, and Dementia/ subfolders.

    Expected folder structure:
        data_dir/
            Normal/
                English 1st Language/
                    file1.txt
                    ...
            MCI/
                English 1st Language/
                    file1.txt
                    ...
            QCI/
                English 1st Language/
                    file1.txt
                    ...
            Dementia/
                English 1st Language/
                    file1.txt
                    ...

    Classification:
        - Normal → NC (0)
        - MCI, QCI, Dementia → Impaired (1)

    Args:
        data_dir: Path to the folder containing the subfolders
        max_rows: Optional limit on total number of samples to load

    Returns:
        DataFrame with columns: participant_id, text, label, chunk_id
    """
    records = []

    # Load Normal folder → label "NC"
    normal_dir = os.path.join(data_dir, "Normal", "English 1st Language")
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
        print(f"Loaded {len([r for r in records if r['label'] == 'NC'])} files from Normal/English 1st Language/")
    else:
        print(f"Warning: Normal folder not found at {normal_dir}")

    # Load MCI folder → label "MCI"
    mci_dir = os.path.join(data_dir, "MCI", "English 1st Language")
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
        print(f"Loaded {len(records) - mci_count_before} files from MCI/English 1st Language/")
    else:
        print(f"Warning: MCI folder not found at {mci_dir}")

    # Load QCI folder → label "QCI"
    qci_dir = os.path.join(data_dir, "QCI", "English 1st Language")
    if os.path.exists(qci_dir):
        qci_count_before = len(records)
        for filename in os.listdir(qci_dir):
            if filename.endswith(".txt"):
                filepath = os.path.join(qci_dir, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    text = f.read()
                records.append({
                    "participant_id": filename.replace(".txt", ""),
                    "text": text,
                    "label": "QCI"
                })
        print(f"Loaded {len(records) - qci_count_before} files from QCI/English 1st Language/")
    else:
        print(f"Warning: QCI folder not found at {qci_dir}")

    # Load Dementia folder → label "Dementia"
    dementia_dir = os.path.join(data_dir, "Dementia", "English 1st Language")
    if os.path.exists(dementia_dir):
        dementia_count_before = len(records)
        for filename in os.listdir(dementia_dir):
            if filename.endswith(".txt"):
                filepath = os.path.join(dementia_dir, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    text = f.read()
                records.append({
                    "participant_id": filename.replace(".txt", ""),
                    "text": text,
                    "label": "Dementia"
                })
        print(f"Loaded {len(records) - dementia_count_before} files from Dementia/English 1st Language/")
    else:
        print(f"Warning: Dementia folder not found at {dementia_dir}")

    if not records:
        raise FileNotFoundError(f"No .txt files found in {data_dir}")

    df = pd.DataFrame(records)
    df["chunk_id"] = 0

    # Apply max_rows limit if specified
    if max_rows is not None and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=42).reset_index(drop=True)
        print(f"Limited to {max_rows} samples")

    # Count by label
    nc_count = len(df[df['label'] == 'NC'])
    mci_count = len(df[df['label'] == 'MCI'])
    qci_count = len(df[df['label'] == 'QCI'])
    dementia_count = len(df[df['label'] == 'Dementia'])
    impaired_total = mci_count + qci_count + dementia_count

    print(f"Total samples: {len(df)}")
    print(f"  NC (Normal): {nc_count}")
    print(f"  Impaired: {impaired_total} (MCI={mci_count}, QCI={qci_count}, Dementia={dementia_count})")
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
