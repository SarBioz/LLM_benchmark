# Refactoring Plan: Portable Benchmarking Pipeline (v2)

## Terminology (corrected)

| Term | Meaning in this project |
|------|------------------------|
| **Tokenization** | Converting raw text → token IDs (each pretrained model has its own tokenizer) |
| **Chunking** | Splitting long token sequences into fixed-length windows (512 tokens, 64 overlap) |
| **Feature extraction** | Running token IDs through a frozen pretrained model to get hidden states |
| **Pooling** | Reducing hidden states to a single vector per transcript (mean pool within chunk, mean across chunks) |
| **Feature vector** | The pooled output of a frozen model — the input to the classifier |
| **Classifier** | A trainable head (e.g. LogReg) that maps feature vectors → predictions |

---

## Core Design Principle

The benchmark holds everything **constant** except the pretrained model:

```
                     FIXED                    VARIABLE                FIXED
                 ┌───────────┐          ┌─────────────────┐     ┌────────────┐
Raw Text ──────►│ Chunking   │─── token IDs ──►│ Pretrained Model │──►│  Pooling    │──► Feature Vector
                 │ Strategy   │          │  (swappable)    │     │  Strategy   │     (H-dimensional)
                 └───────────┘          └─────────────────┘     └────────────┘
                                                                       │
                                                                       ▼
                                                                ┌────────────┐
                                                                │ Classifier │──► Prediction
                                                                │  (fixed)   │
                                                                └────────────┘
```

**What stays the same across all models:**
- Data loading, labels, participant-level splits
- Chunking strategy (max_length, overlap)
- Pooling strategy (mean pool)
- Classifier head (LogReg with same hyperparameters)
- Evaluation metrics

**What changes per model:**
- The pretrained model (and its paired tokenizer — each model requires its own)
- The resulting feature vector dimensionality (128 for bert-tiny, 768 for others)

**Note on tokenizers:** Each pretrained model has a paired tokenizer that cannot be swapped. The tokenizer is part of the model, not part of the shared pipeline. What *is* shared is the chunking strategy (how we handle long texts) and the pooling strategy (how we reduce to one vector).

---

## Current State Summary

### Pipeline Flow
1. **Load data** – People Speech parquet → DataFrame (participant_id, text)
2. **Label** – Random debug labels per participant (NC/CIND/AD → binary 0/1)
3. **Split** – Participant-level 70/15/15 train/val/test (seed=42, leakage-safe)
4. **Per model**: Load frozen model → tokenize+chunk → feature extraction → pooling → feature vector → classifier → evaluate

### Models
| # | Model ID | Hidden Size (feature vector dim) |
|---|----------|----------------------------------|
| 1 | prajjwal1/bert-tiny | 128 |
| 2 | distilbert-base-uncased | 768 |
| 3 | albert-base-v2 | 768 |
| 4 | distilroberta-base | 768 |

### Current Files
- `main.py` – orchestrator
- `hf_encoder.py` – load/freeze HF model
- `embedder.py` – tokenize + chunk + feature extraction + pooling (all bundled)
- `data_io.py` – load parquet data
- `classifier_head.py` – LogReg / SVM / MLP classifier heads
- `eval_utils.py` – accuracy, F1, AUC
- `splits.py` – participant-level splitting
- `labels_debug.py` – random debug labels
- `Label_binary.py` – NC→0, CIND/AD→1

---

## Proposed Architecture

```
Transformer_encder/
├── benchlib/                        # Shared benchmark library
│   ├── __init__.py
│   ├── data_io.py                   # Data loading
│   ├── labels.py                    # Label assignment + binary conversion
│   ├── splits.py                    # Participant-level splitting
│   ├── chunking.py                  # Text chunking strategy (shared across models)
│   ├── feature_extraction.py        # Run frozen model + pooling → feature vectors
│   ├── classifier.py                # Unified classifier head (NOT loss.py)
│   └── eval_utils.py                # Evaluation metrics
│
├── models/                          # One file per pretrained model
│   ├── __init__.py                  # MODEL_REGISTRY: auto-discovers all models
│   ├── bert_tiny.py
│   ├── distilbert.py
│   ├── albert.py
│   └── distilroberta.py
│
├── results/                         # Output directory
│   ├── features/                    # Saved feature vectors per model (.npy)
│   └── results.csv                  # Comparison table
│
├── config.py                        # Central configuration
├── run_benchmark.py                 # Main entrypoint
├── REFACTOR_PLAN.md                 # This file
├── PROGRESS.md                      # Progress tracker
└── Readme.md
```

---

## Detailed Design

### `benchlib/chunking.py`
Shared chunking logic extracted from current `embedder.py`. Each model's tokenizer produces different token IDs, but the chunking *strategy* is identical.

```python
def chunk_token_ids(input_ids, max_length, overlap) -> list[Tensor]:
    """Split a long token sequence into overlapping windows."""

def tokenize_and_chunk(text, tokenizer, max_length, overlap) -> list[dict]:
    """Tokenize text with a model's tokenizer, then chunk."""
```

### `benchlib/feature_extraction.py`
Runs the frozen model and pools the output. Returns the feature vector that feeds the classifier. Stores feature vectors so they can be inspected independently.

```python
def extract_features_single(text, model, tokenizer, device, max_length, overlap) -> np.ndarray:
    """Raw text → single feature vector (H,) for one transcript."""

def extract_features_batch(texts, model, tokenizer, device, max_length, overlap) -> np.ndarray:
    """Raw texts → feature matrix (N, H) for N transcripts."""
    # Returns the feature matrix that becomes classifier input
```

### `benchlib/classifier.py` (renamed from `loss.py`)
The trainable classification head. Same for all models.

```python
def train_classifier(X_train, y_train, method="logreg", seed=42) -> Pipeline:
    """Train a classifier on feature vectors. Identical for all models."""

def predict(clf, X) -> np.ndarray:
    """Get predictions from trained classifier."""
```

### `models/*.py` (per-model files)
Each file defines only what's unique to that model:

```python
MODEL_ID = "prajjwal1/bert-tiny"
MODEL_NAME = "BERT-Tiny"
# Optional: model-specific overrides (none expected for now)
```

### `config.py`
Single source of truth for all shared settings:

```python
SEED = 42
SPLIT_RATIOS = (0.70, 0.15, 0.15)
MAX_LENGTH = 512
OVERLAP_TOKENS = 64
POOLING = "mean"
CLASSIFIER = "logreg"          # change here → changes for ALL models
MAX_ROWS = 2000
DATA_DIR = r"D:\hf_cache\..."
OUTPUT_DIR = "results/"
SAVE_FEATURES = True           # save feature vectors to results/features/
```

### `run_benchmark.py`
```
1. Load data                              (benchlib.data_io)
2. Assign labels                          (benchlib.labels)
3. Split by participant                   (benchlib.splits)
4. For each model in MODEL_REGISTRY:
   a. Load frozen pretrained model        (models/*.py → transformers)
   b. Extract feature vectors             (benchlib.feature_extraction)
   c. Save feature vectors to disk        (results/features/<model>.npy)
   d. Train classifier on train features  (benchlib.classifier)
   e. Evaluate on val/test features       (benchlib.eval_utils)
   f. Append to results table
5. Save results table                     (results/results.csv)
```

---

## Detailed Plan

### Phase 1: Create `benchlib/` shared library
- [ ] **1.1** Create `benchlib/__init__.py`
- [ ] **1.2** Move `data_io.py` → `benchlib/data_io.py`
- [ ] **1.3** Merge `labels_debug.py` + `Label_binary.py` → `benchlib/labels.py`
- [ ] **1.4** Move `splits.py` → `benchlib/splits.py`
- [ ] **1.5** Create `benchlib/chunking.py` – extract chunking logic from `embedder.py`
- [ ] **1.6** Create `benchlib/feature_extraction.py` – frozen model forward pass + pooling → feature vectors
- [ ] **1.7** Create `benchlib/classifier.py` – unified classifier head (StandardScaler + LogReg/SVM/MLP)
- [ ] **1.8** Move `eval_utils.py` → `benchlib/eval_utils.py`

### Phase 2: Create `config.py`
- [ ] **2.1** Central config: seed, splits, max_length, overlap, pooling, classifier type, data dir, output dir, save_features flag

### Phase 3: Per-model files under `models/`
- [ ] **3.1** Create `models/__init__.py` with `MODEL_REGISTRY`
- [ ] **3.2** `models/bert_tiny.py`
- [ ] **3.3** `models/distilbert.py`
- [ ] **3.4** `models/albert.py`
- [ ] **3.5** `models/distilroberta.py`

### Phase 4: Create `run_benchmark.py`
- [ ] **4.1** New main entrypoint implementing the benchmark loop described above

### Phase 5: Cleanup & verify
- [ ] **5.1** Delete old top-level files now in `benchlib/` or `models/`
- [ ] **5.2** Update `Readme.md` with new architecture diagram
- [ ] **5.3** Verify pipeline produces equivalent results to current code

---

## How to Add a New Model

1. Create `models/new_model.py` with `MODEL_ID` and `MODEL_NAME`
2. Run `python run_benchmark.py`

Everything else (chunking, feature extraction, classification, evaluation) is handled by `benchlib/`.

---

## Feature Vector Access

Feature vectors are saved to `results/features/<model_name>.npy` for every model. This enables:
- Inspecting what the frozen model produces before classification
- Rerunning classifiers without re-extracting features
- Comparing feature distributions across models
- Dimensionality analysis (128-d vs 768-d)

---

## Progress Tracking

See [PROGRESS.md](PROGRESS.md) for real-time status of each task.
