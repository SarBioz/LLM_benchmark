# run_benchmark.py
"""Main entrypoint: runs all registered models through the shared
benchmark pipeline and saves a comparison table."""

import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

import config
from models import MODEL_REGISTRY
from benchlib.data_io import load_people_speech_parquet, dataset_to_transcripts_df
from benchlib.labels import (
    assign_random_labels_per_participant,
    save_participant_label_map,
    make_binary_label,
)
from benchlib.splits import make_participant_splits, attach_splits
from benchlib.feature_extraction import extract_features_batch, extract_combined_features_batch
from benchlib.classifier import train_classifier
from benchlib.eval_utils import eval_binary


def load_frozen_model(model_id: str, device: str):
    """Load a HuggingFace encoder, freeze it, return (model, tokenizer)."""
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModel.from_pretrained(model_id)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    model.to(device)
    return model, tokenizer


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── 1) Load data ──
    ds = load_people_speech_parquet(config.DATA_DIR)
    df = dataset_to_transcripts_df(ds, id_col="id", text_col="text", max_rows=config.MAX_ROWS)
    print(f"\nTranscripts: {len(df)}")

    # ── 2) Labels (debug) ──
    df = assign_random_labels_per_participant(
        df, participant_col="participant_id", label_col="label_debug",
        labels=("NC", "CIND", "AD"), seed=config.SEED,
    )
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    save_participant_label_map(
        df, participant_col="participant_id", label_col="label_debug",
        out_path=os.path.join(config.OUTPUT_DIR, "participant_labels_debug.csv"),
    )
    df = make_binary_label(df, src_label_col="label_debug", out_col="y")
    print("Class balance:", df["y"].value_counts().to_dict())

    # ── 3) Split by participant ──
    split_df = make_participant_splits(
        df, participant_col="participant_id", split_col="split",
        train_frac=config.TRAIN_FRAC, val_frac=config.VAL_FRAC,
        test_frac=config.TEST_FRAC, seed=config.SEED,
    )
    df = attach_splits(df, split_df, participant_col="participant_id", split_col="split")

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df   = df[df["split"] == "val"].reset_index(drop=True)
    test_df  = df[df["split"] == "test"].reset_index(drop=True)

    y_train = train_df["y"].to_numpy().astype(int)
    y_val   = val_df["y"].to_numpy().astype(int)
    y_test  = test_df["y"].to_numpy().astype(int)

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    print(f"Linguistic features: {'Enabled (Mobtahej et al.)' if config.USE_LINGUISTIC_FEATURES else 'Disabled'}")

    # ── 4) Benchmark loop ──
    results_rows = []

    for entry in MODEL_REGISTRY:
        model_id = entry["model_id"]
        model_name = entry["model_name"]
        print("\n" + "=" * 70)
        print(f"MODEL: {model_name}  ({model_id})")
        print("=" * 70)

        model, tokenizer = load_frozen_model(model_id, device)

        # a) Extract feature vectors (BERT + optional linguistic features)
        X_train = extract_combined_features_batch(
            train_df["text"].tolist(), model, tokenizer, device,
            max_length=config.MAX_LENGTH, overlap=config.OVERLAP_TOKENS,
            use_linguistic=config.USE_LINGUISTIC_FEATURES,
        )
        X_val = extract_combined_features_batch(
            val_df["text"].tolist(), model, tokenizer, device,
            max_length=config.MAX_LENGTH, overlap=config.OVERLAP_TOKENS,
            use_linguistic=config.USE_LINGUISTIC_FEATURES,
        )
        X_test = extract_combined_features_batch(
            test_df["text"].tolist(), model, tokenizer, device,
            max_length=config.MAX_LENGTH, overlap=config.OVERLAP_TOKENS,
            use_linguistic=config.USE_LINGUISTIC_FEATURES,
        )

        print(f"Feature vector dim: {X_train.shape[1]}")

        # b) Save feature vectors
        if config.SAVE_FEATURES:
            os.makedirs(config.FEATURES_DIR, exist_ok=True)
            safe_name = model_id.replace("/", "__")
            for split_name, X in [("train", X_train), ("val", X_val), ("test", X_test)]:
                path = os.path.join(config.FEATURES_DIR, f"{safe_name}_{split_name}.npy")
                np.save(path, X)
            print(f"Saved feature vectors to {config.FEATURES_DIR}/")

        # c) Train classifier
        clf = train_classifier(X_train, y_train, method=config.CLASSIFIER, seed=config.SEED)

        # d) Evaluate
        val_metrics = eval_binary(clf, X_val, y_val)
        test_metrics = eval_binary(clf, X_test, y_test)

        print(f"\n[VAL]  acc={val_metrics['accuracy']:.4f}  f1={val_metrics['f1']:.4f}  auc={val_metrics['auc']:.4f}")
        print(f"[TEST] acc={test_metrics['accuracy']:.4f}  f1={test_metrics['f1']:.4f}  auc={test_metrics['auc']:.4f}")

        results_rows.append({
            "model": model_id,
            "model_name": model_name,
            "feature_dim": X_train.shape[1],
            "linguistic_features": config.USE_LINGUISTIC_FEATURES,
            "pooling": config.POOLING,
            "max_length": config.MAX_LENGTH,
            "overlap_tokens": config.OVERLAP_TOKENS,
            "classifier": config.CLASSIFIER,
            "val_accuracy": val_metrics["accuracy"],
            "val_f1": val_metrics["f1"],
            "val_auc": val_metrics["auc"],
            "test_accuracy": test_metrics["accuracy"],
            "test_f1": test_metrics["f1"],
            "test_auc": test_metrics["auc"],
        })

        # Free GPU memory
        del model
        torch.cuda.empty_cache()

    # ── 5) Save results ──
    results_df = pd.DataFrame(results_rows)
    results_path = os.path.join(config.OUTPUT_DIR, "results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved: {results_path}")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
