# Transformer Encoder Benchmark

Compare frozen pretrained transformer encoders as feature extractors
for binary cognitive-health classification.

## Architecture

```
Raw Text
  │
  ├─ Tokenizer (model-specific, paired with pretrained model)
  │
  ├─ Chunking (shared: 512 tokens, 64 overlap)
  │
  ├─ Frozen Pretrained Model ← VARIABLE (swapped per benchmark run)
  │
  ├─ Mean Pooling (shared: within-chunk + across-chunk)
  │
  ├─ Feature Vector (H-dimensional, saved to results/features/)
  │
  ├─ Classifier (shared: StandardScaler + LogReg)
  │
  └─ Evaluation (accuracy, F1, AUC)
```

## Project Structure

```
Transformer_encder/
├── benchlib/                    # Shared benchmark library
│   ├── __init__.py
│   ├── data_io.py               # Data loading (parquet)
│   ├── labels.py                # Label assignment + binary conversion
│   ├── splits.py                # Participant-level train/val/test splitting
│   ├── chunking.py              # Token-level chunking strategy
│   ├── feature_extraction.py    # Frozen model forward pass + pooling
│   ├── classifier.py            # Unified classifier head
│   └── eval_utils.py            # Evaluation metrics
│
├── models/                      # One file per pretrained model
│   ├── __init__.py              # MODEL_REGISTRY
│   ├── bert_tiny.py             # prajjwal1/bert-tiny (128-d)
│   ├── distilbert.py            # distilbert-base-uncased (768-d)
│   ├── albert.py                # albert-base-v2 (768-d)
│   └── distilroberta.py         # distilroberta-base (768-d)
│
├── results/                     # Output (auto-created)
│   ├── features/                # Saved feature vectors per model/split
│   └── results.csv              # Comparison table
│
├── config.py                    # Central configuration
├── run_benchmark.py             # Main entrypoint
├── REFACTOR_PLAN.md             # Refactoring plan
└── PROGRESS.md                  # Progress tracker
```

## Usage

```bash
conda activate nlp_gpu
python run_benchmark.py
```

## Adding a New Model

1. Create `models/new_model.py`:
   ```python
   MODEL_ID = "huggingface/model-name"
   MODEL_NAME = "Display Name"
   ```
2. Register it in `models/__init__.py`
3. Run `python run_benchmark.py`

## Configuration

Edit `config.py` to change settings that apply to **all** models:

| Setting | Default | Description |
|---------|---------|-------------|
| SEED | 42 | Random seed everywhere |
| MAX_ROWS | 2000 | Max transcripts to load |
| TRAIN/VAL/TEST_FRAC | 0.70/0.15/0.15 | Participant-level splits |
| MAX_LENGTH | 512 | Token chunk size |
| OVERLAP_TOKENS | 64 | Overlap between chunks |
| CLASSIFIER | "logreg" | Classifier head: logreg, linear_svm, mlp |
| SAVE_FEATURES | True | Save feature vectors to disk |
