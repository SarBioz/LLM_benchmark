# main_encoder.py
import numpy as np
import pandas as pd

from data_io import load_people_speech_parquet, dataset_to_transcripts_df
from labels_debug import assign_random_labels_per_participant, save_participant_label_map
from splits import make_participant_splits, attach_splits
from Label_binary import make_binary_label

from hf_encoder import load_frozen_encoder
from embedder import embed_dataframe
from classifier_head import train_logreg  # (optional) train_linear_svm, train_mlp
from eval_utils import eval_binary


ENCODER_MODELS = [
    # use EXACT HF names you want:
    "prajjwal1/bert-tiny",
    "distilbert-base-uncased",
    "albert-base-v2",
    "distilroberta-base",
]


def main():
    root_clean_dir = r"D:\hf_cache\hub\datasets--MLCommons--peoples_speech\snapshots\f10597c5d3d3a63f8b6827701297c3afdf178272\clean"

    # ===== 1) LOAD =====
    ds = load_people_speech_parquet(root_clean_dir)

    # ===== 2) TRANSCRIPTS DF (NO WORD CHUNKING) =====
    df = dataset_to_transcripts_df(ds, id_col="id", text_col="text", max_rows=2000)
    print("\n=== Transcripts DF ===")
    print(df.head(5).to_string(index=False))
    print("Total transcripts:", len(df))

    # ===== 3) DEBUG LABELS (REMOVE when real labels) =====
    df = assign_random_labels_per_participant(
        df,
        participant_col="participant_id",
        label_col="label_debug",
        labels=("NC", "CIND", "AD"),
        seed=42,
    )
    save_participant_label_map(
        df,
        participant_col="participant_id",
        label_col="label_debug",
        out_path="participant_labels_debug.csv",
    )

    # ===== 3.5) BINARY LABEL =====
    df = make_binary_label(
        df,
        src_label_col="label_debug",
        out_col="y",
        normal_label="NC",
        abnormal_labels=("CIND", "AD"),
    )
    print("\nClass balance (transcripts):")
    print(df["y"].value_counts())

    # ===== 4) SPLIT BY PARTICIPANT =====
    split_df = make_participant_splits(
        df,
        participant_col="participant_id",
        split_col="split",
        train_frac=0.70,
        val_frac=0.15,
        test_frac=0.15,
        seed=42,
    )
    df = attach_splits(df, split_df, participant_col="participant_id", split_col="split")
    split_df.to_csv("splits_debug.csv", index=False)
    df.to_csv("transcripts_labeled_split_debug.csv", index=False)

    print("\nSaved: splits_debug.csv")
    print("Saved: transcripts_labeled_split_debug.csv")

    # ===== 5) PREP SPLITS =====
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df   = df[df["split"] == "val"].reset_index(drop=True)
    test_df  = df[df["split"] == "test"].reset_index(drop=True)

    y_train = train_df["y"].to_numpy().astype(int)
    y_val   = val_df["y"].to_numpy().astype(int)
    y_test  = test_df["y"].to_numpy().astype(int)

    results_rows = []

    # ===== 6) LOOP MODELS =====
    for model_name in ENCODER_MODELS:
        print("\n" + "=" * 80)
        print(f"MODEL: {model_name}")
        print("=" * 80)

        bundle = load_frozen_encoder(model_name)

        # --- embeddings (one per transcript) ---
        X_train = embed_dataframe(bundle, train_df["text"].tolist(), max_length=512, overlap_tokens=64, verbose_every=50)
        X_val   = embed_dataframe(bundle, val_df["text"].tolist(),   max_length=512, overlap_tokens=64, verbose_every=50)
        X_test  = embed_dataframe(bundle, test_df["text"].tolist(),  max_length=512, overlap_tokens=64, verbose_every=50)

        # save embeddings for this model (ALL transcripts in original order, not just train)
        X_all = embed_dataframe(bundle, df["text"].tolist(), max_length=512, overlap_tokens=64, verbose_every=50)
        safe_name = model_name.replace("/", "__")
        np.save(f"embeddings_{safe_name}.npy", X_all)
        print(f"Saved embeddings_{safe_name}.npy  shape={X_all.shape}")

        # --- classifier head ---
        clf = train_logreg(X_train, y_train)

        # --- eval ---
        val_metrics = eval_binary(clf, X_val, y_val)
        test_metrics = eval_binary(clf, X_test, y_test)

        print("\n[VAL]")
        print(val_metrics["report"])
        print("\n[TEST]")
        print(test_metrics["report"])

        results_rows.append({
            "model": model_name,
            "pooling": "mean_last_hidden + mean_over_chunks",
            "max_length": 512,
            "overlap_tokens": 64,
            "clf": "LogReg",
            "val_accuracy": val_metrics["accuracy"],
            "val_f1": val_metrics["f1"],
            "val_auc": val_metrics["auc"],
            "test_accuracy": test_metrics["accuracy"],
            "test_f1": test_metrics["f1"],
            "test_auc": test_metrics["auc"],
        })

    # ===== 7) SAVE RESULTS CSV =====
    results_df = pd.DataFrame(results_rows)
    results_df.to_csv("results_encoder.csv", index=False)
    print("\nSaved: results_encoder.csv")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
