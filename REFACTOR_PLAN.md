# Refactoring Plan: Portable Benchmarking Pipeline

## Current State Summary

### Pipeline Flow
1. **Load data** - People Speech parquet shards → DataFrame (participant_id, text)
2. **Label** - Random debug labels per participant (NC/CIND/AD → binary 0/1)
3. **Split** - Participant-level 70/15/15 train/val/test (seed=42, leakage-safe)
4. **Per model**: Load frozen encoder → tokenize+chunk (512 tokens, 64 overlap) → mean-pool embeddings → LogisticRegression → evaluate (accuracy, F1, AUC)

### Models
| # | Model ID | Type |
|---|----------|------|
| 1 | prajjwal1/bert-tiny | BERT (tiny) |
| 2 | distilbert-base-uncased | DistilBERT |
| 3 | albert-base-v2 | ALBERT |
| 4 | distilroberta-base | DistilRoBERTa |

### Current Files
- `main.py` – orchestrator
- `hf_encoder.py` – load/freeze HF encoder
- `embedder.py` – tokenize, chunk, embed
- `data_io.py` – load parquet data
- `classifier_head.py` – LogReg / SVM / MLP heads
- `eval_utils.py` – accuracy, F1, AUC
- `splits.py` – participant-level splitting
- `labels_debug.py` – random debug labels
- `Label_binary.py` – NC→0, CIND/AD→1

---

## Refactoring Goals

1. **Shared library** (`benchlib/`) for data splitting, tokenization, and embedding
2. **One Python file per model** under `models/`
3. **Unified loss function** across all models (configurable from one place)
4. **Portable & extensible** – adding a new model = adding one file
5. **Progress tracking** built in

---

## Proposed Directory Structure

```
Transformer_encder/
├── benchlib/                    # Shared library (importable package)
│   ├── __init__.py
│   ├── data_io.py               # Data loading (from current data_io.py)
│   ├── labels.py                # Label assignment + binary conversion (merge labels_debug + Label_binary)
│   ├── splits.py                # Participant-level splitting (from current splits.py)
│   ├── tokenizer.py             # Tokenization + chunking (extracted from embedder.py)
│   ├── embedder.py              # Embedding with pooling (extracted from embedder.py)
│   ├── loss.py                  # Single unified loss/classifier head
│   └── eval_utils.py            # Evaluation metrics (from current eval_utils.py)
│
├── models/                      # One file per model
│   ├── __init__.py
│   ├── bert_tiny.py
│   ├── distilbert.py
│   ├── albert.py
│   └── distilroberta.py
│
├── config.py                    # Central configuration (seeds, splits, max_length, model list, loss choice)
├── run_benchmark.py             # New main entrypoint: runs all models, collects results
├── REFACTOR_PLAN.md             # This file
├── PROGRESS.md                  # Progress tracker
└── Readme.md                    # Updated documentation
```

---

## Detailed Plan

### Phase 1: Create `benchlib/` shared library
- [ ] **1.1** Create `benchlib/__init__.py`
- [ ] **1.2** Move `data_io.py` → `benchlib/data_io.py` (no changes needed)
- [ ] **1.3** Merge `labels_debug.py` + `Label_binary.py` → `benchlib/labels.py`
- [ ] **1.4** Move `splits.py` → `benchlib/splits.py` (no changes needed)
- [ ] **1.5** Extract tokenization logic from `embedder.py` → `benchlib/tokenizer.py`
  - `tokenize_and_chunk(text, tokenizer, max_length=512, overlap=64) → list[dict]`
- [ ] **1.6** Refactor embedding logic → `benchlib/embedder.py`
  - `embed_texts(texts, encoder_bundle, max_length=512, overlap=64) → np.ndarray`
  - Depends on `benchlib/tokenizer.py`
- [ ] **1.7** Create `benchlib/loss.py` – unified classifier head
  - Single function: `train_classifier(X_train, y_train, method="logreg") → model`
  - All models will call this same function
  - Supported methods: logreg (default), linear_svm, mlp
- [ ] **1.8** Move `eval_utils.py` → `benchlib/eval_utils.py` (no changes needed)

### Phase 2: Create `config.py`
- [ ] **2.1** Central config with all hyperparameters:
  - `SEED = 42`
  - `SPLIT_RATIOS = (0.70, 0.15, 0.15)`
  - `MAX_LENGTH = 512`
  - `OVERLAP_TOKENS = 64`
  - `CLASSIFIER = "logreg"` (single place to change loss/classifier for all models)
  - `MAX_ROWS = 2000`
  - `DATA_DIR = ...` (parquet path)
  - `OUTPUT_DIR = "results/"`

### Phase 3: Create per-model files under `models/`
- [ ] **3.1** Create `models/__init__.py` with a `MODEL_REGISTRY` dict
- [ ] **3.2** `models/bert_tiny.py` – defines model_id, any model-specific overrides
- [ ] **3.3** `models/distilbert.py`
- [ ] **3.4** `models/albert.py`
- [ ] **3.5** `models/distilroberta.py`

Each model file will follow this template:
```python
MODEL_ID = "prajjwal1/bert-tiny"
MODEL_NAME = "BERT-Tiny"

def get_config():
    """Return model-specific config overrides (if any)."""
    return {}
```

Models register themselves via `models/__init__.py` which collects all entries.

### Phase 4: Create `run_benchmark.py`
- [ ] **4.1** New entrypoint that:
  1. Loads data via `benchlib.data_io`
  2. Assigns labels via `benchlib.labels`
  3. Splits via `benchlib.splits`
  4. Iterates over registered models
  5. For each: load encoder → embed → train classifier → evaluate
  6. Collects all results into a comparison table
  7. Saves results CSV + per-model embeddings

### Phase 5: Cleanup
- [ ] **5.1** Delete old top-level files that are now in `benchlib/` or `models/`
- [ ] **5.2** Update `Readme.md` with new structure
- [ ] **5.3** Verify pipeline produces equivalent results to current code

---

## Unified Loss Function Strategy

All models will use the **same** classifier head defined in `benchlib/loss.py`. The classifier type is set once in `config.py` (`CLASSIFIER = "logreg"`). Changing it changes it for every model. This ensures fair comparison.

Current default: `StandardScaler → LogisticRegression(max_iter=2000, class_weight="balanced")`

---

## How to Add a New Model

1. Create `models/new_model.py` with `MODEL_ID` and `MODEL_NAME`
2. Register it in `models/__init__.py`
3. Run `python run_benchmark.py`

That's it. The shared library handles everything else.

---

## Progress Tracking

See [PROGRESS.md](PROGRESS.md) for real-time status of each task.
