# run_benchmark.py
"""Main entrypoint: runs all registered models through the shared
benchmark pipeline and saves a comparison table.

Updated to use combined features:
- BERT transcript embedding (H dimensions)
- Linguistic features (10 interpretable features)
- Total: H + 10 dimensions
"""

import os
import numpy as np
import pandas as pd
import torch
import joblib
from transformers import AutoTokenizer, AutoModel
from typing import List, Tuple

import config
from models import MODEL_REGISTRY

# NEW: Import for real data (text files from folders)
from benchlib.data_io import load_text_files_from_folders

# OLD (dummy data): Import for parquet files
# from benchlib.data_io import load_people_speech_parquet, dataset_to_transcripts_df

# NEW: Import only make_binary_label (labels come from folder structure)
from benchlib.labels import make_binary_label

# OLD (dummy data): Imports for random label assignment
# from benchlib.labels import (
#     assign_random_labels_per_participant,
#     save_participant_label_map,
#     make_binary_label,
# )

from benchlib.splits import make_participant_splits, attach_splits
from benchlib.feature_extraction import extract_features_batch
from benchlib.linguistic_features import (
    extract_all_linguistic_features,
    get_feature_vector,
)
from benchlib.classifier import train_classifier, get_impairment_scores
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


def extract_combined_features_single(
    text: str,
    model,
    tokenizer,
    device: str,
    max_length: int = 512,
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract combined features for a single transcript.

    Returns:
        feature_vector: (10 + H,) array where:
            - [0:10] = interpretable linguistic features
            - [10:] = BERT transcript embedding
        feature_names: List of feature names
    """
    # Extract all linguistic features (includes BERT embedding)
    features = extract_all_linguistic_features(
        text=text,
        model=model,
        tokenizer=tokenizer,
        device=device,
        use_language_tool=False,
        max_length=max_length,
    )

    # Convert to flat feature vector
    feature_vector, feature_names = get_feature_vector(features)

    return feature_vector, feature_names


def extract_combined_features_batch(
    texts: List[str],
    model,
    tokenizer,
    device: str,
    max_length: int = 512,
    verbose_every: int = 50,
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract combined features for a batch of transcripts.

    Returns:
        X: (N, 10 + H) feature matrix
        feature_names: List of feature names
    """
    features_list = []
    feature_names = None

    for i, text in enumerate(texts):
        fv, fn = extract_combined_features_single(
            text, model, tokenizer, device, max_length
        )
        features_list.append(fv)

        if feature_names is None:
            feature_names = fn

        if verbose_every and (i + 1) % verbose_every == 0:
            print(f"  Extracted features {i + 1}/{len(texts)}")

    X = np.vstack(features_list)
    return X, feature_names


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── 1) Load data ──
    # NEW: Load real data from Normal/ and MCI/ folders (labels from folder structure)
    df = load_text_files_from_folders(config.DATA_DIR, max_rows=config.MAX_ROWS)
    print(f"\nTranscripts: {len(df)}")

    # OLD (dummy data): Load parquet files and assign random labels
    # ds = load_people_speech_parquet(config.DATA_DIR)
    # df = dataset_to_transcripts_df(ds, id_col="id", text_col="text", max_rows=config.MAX_ROWS)
    # print(f"\nTranscripts: {len(df)}")

    # ── 2) Labels ──
    # NEW: Convert labels to binary (labels already in df from folder structure)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    df = make_binary_label(df, src_label_col="label", out_col="y")
    print("Class balance:", df["y"].value_counts().to_dict())

    # OLD (dummy data): Random label assignment
    # df = assign_random_labels_per_participant(
    #     df, participant_col="participant_id", label_col="label_debug",
    #     labels=("NC", "CIND", "AD"), seed=config.SEED,
    # )
    # os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    # save_participant_label_map(
    #     df, participant_col="participant_id", label_col="label_debug",
    #     out_path=os.path.join(config.OUTPUT_DIR, "participant_labels_debug.csv"),
    # )
    # df = make_binary_label(df, src_label_col="label_debug", out_col="y")
    # print("Class balance:", df["y"].value_counts().to_dict())

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

    # ── 4) Benchmark loop ──
    results_rows = []

    for entry in MODEL_REGISTRY:
        model_id = entry["model_id"]
        model_name = entry["model_name"]
        print("\n" + "=" * 70)
        print(f"MODEL: {model_name}  ({model_id})")
        print("=" * 70)

        model, tokenizer = load_frozen_model(model_id, device)
        hidden_size = model.config.hidden_size

        # a) Extract COMBINED feature vectors (BERT + linguistic)
        print("\nExtracting combined features (BERT + linguistic)...")

        X_train, feature_names = extract_combined_features_batch(
            train_df["text"].tolist(), model, tokenizer, device,
            max_length=config.MAX_LENGTH, verbose_every=50,
        )
        X_val, _ = extract_combined_features_batch(
            val_df["text"].tolist(), model, tokenizer, device,
            max_length=config.MAX_LENGTH, verbose_every=50,
        )
        X_test, _ = extract_combined_features_batch(
            test_df["text"].tolist(), model, tokenizer, device,
            max_length=config.MAX_LENGTH, verbose_every=50,
        )

        print(f"\nFeature vector dimensions:")
        print(f"  Interpretable features: 10")
        print(f"  BERT embedding: {hidden_size}")
        print(f"  Total: {X_train.shape[1]}")

        # b) Save feature vectors
        if config.SAVE_FEATURES:
            os.makedirs(config.FEATURES_DIR, exist_ok=True)
            safe_name = model_id.replace("/", "__")
            for split_name, X in [("train", X_train), ("val", X_val), ("test", X_test)]:
                path = os.path.join(config.FEATURES_DIR, f"{safe_name}_combined_{split_name}.npy")
                np.save(path, X)

            # Save feature names
            names_path = os.path.join(config.FEATURES_DIR, f"{safe_name}_feature_names.txt")
            with open(names_path, 'w') as f:
                for name in feature_names:
                    f.write(name + '\n')

            print(f"Saved feature vectors to {config.FEATURES_DIR}/")

        # c) Train classifier
        clf = train_classifier(X_train, y_train, method=config.CLASSIFIER, seed=config.SEED)

        # d) Evaluate
        val_metrics = eval_binary(clf, X_val, y_val)
        test_metrics = eval_binary(clf, X_test, y_test)

        print(f"\n[VAL]  acc={val_metrics['accuracy']:.4f}  f1={val_metrics['f1']:.4f}  auc={val_metrics['auc']:.4f}")
        print(f"[TEST] acc={test_metrics['accuracy']:.4f}  f1={test_metrics['f1']:.4f}  auc={test_metrics['auc']:.4f}")

        # e) Feature importance (for interpretable features)
        if hasattr(clf, 'coef_'):
            print("\nInterpretable feature weights (top 5):")
            coefs = clf.coef_[0][:10]  # First 10 are interpretable
            importance = list(zip(feature_names[:10], coefs))
            importance.sort(key=lambda x: abs(x[1]), reverse=True)
            for name, weight in importance[:5]:
                print(f"  {name}: {weight:+.4f}")

        # f) Save trained classifier
        os.makedirs(os.path.join(config.OUTPUT_DIR, "classifiers"), exist_ok=True)
        safe_name = model_id.replace("/", "__")
        clf_path = os.path.join(config.OUTPUT_DIR, "classifiers", f"{safe_name}_classifier.joblib")
        joblib.dump(clf, clf_path)
        print(f"Saved classifier to {clf_path}")

        # g) Get impairment scores (0-1 probability) and predictions
        scores_train = get_impairment_scores(clf, X_train)
        scores_val = get_impairment_scores(clf, X_val)
        scores_test = get_impairment_scores(clf, X_test)

        # Show sample scores
        print("\nSample impairment scores (0=NC, 1=MCI):")
        for i in range(min(5, len(scores_test))):
            label = "MCI" if y_test[i] == 1 else "NC"
            print(f"  Sample {i}: score={scores_test[i]:.3f}  (actual: {label})")

        # h) Save predictions and scores for later plotting
        os.makedirs(os.path.join(config.OUTPUT_DIR, "predictions"), exist_ok=True)
        predictions = {
            # True labels
            "y_train_true": y_train,
            "y_val_true": y_val,
            "y_test_true": y_test,
            # Binary predictions (threshold=0.5)
            "y_train_pred": clf.predict(X_train),
            "y_val_pred": clf.predict(X_val),
            "y_test_pred": clf.predict(X_test),
            # Impairment scores (0-1 probability)
            "scores_train": scores_train,
            "scores_val": scores_val,
            "scores_test": scores_test,
        }
        pred_path = os.path.join(config.OUTPUT_DIR, "predictions", f"{safe_name}_predictions.npz")
        np.savez(pred_path, **predictions)
        print(f"Saved predictions and scores to {pred_path}")

        results_rows.append({
            "model": model_id,
            "model_name": model_name,
            "feature_dim": X_train.shape[1],
            "interpretable_features": 10,
            "bert_embedding_dim": hidden_size,
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
    results_path = os.path.join(config.OUTPUT_DIR, "results_combined_features.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved: {results_path}")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
