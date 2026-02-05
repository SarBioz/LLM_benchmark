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

## Features Extracted

The pipeline extracts two types of features from each transcript:

### 1. Linguistic Features (directly from transcript text)

| # | Feature | Description | Formula |
|---|---------|-------------|---------|
| 1 | `semantic_coherence` | Avg cosine similarity between consecutive sentence embeddings | `(1/(N-1)) * Σ cos(e_i, e_{i+1})` |
| 2 | `sentiment_score` | Average sentiment score across sentences | `(1/N) * Σ p_i` |
| 3 | `num_sentences` | Number of sentences in transcript | Count |
| 4 | `grammar_error_rate` | Grammar errors divided by total words | `E / W` |
| 5 | `avg_word_length` | Average character length of words | `(1/W) * Σ len(w_j)` |
| 6 | `std_word_length` | Standard deviation of word lengths | Std dev |
| 7 | `ttr` | Type-Token Ratio (vocabulary richness) | `V / N` |
| 8 | `gttr` | Guiraud's Root TTR | `V / √N` |
| 9 | `total_tokens` | Total number of word tokens | `N` |
| 10 | `unique_types` | Number of unique words | `V` |

### 2. Transformer Embedding (from frozen pretrained model)

| Model | Embedding Dimensions |
|-------|---------------------|
| BERT-Tiny | 128 |
| DistilBERT | 768 |
| ALBERT | 768 |
| DistilRoBERTa | 768 |

The transformer embedding is computed by:
1. Split transcript into sentences
2. Get BERT embedding for each sentence (masked mean pooling)
3. Average all sentence embeddings → transcript embedding

### Total Feature Vector

**Total dimensions = 10 (linguistic) + H (transformer embedding)**

- BERT-Tiny: 10 + 128 = **138 features**
- Other models: 10 + 768 = **778 features**

---

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
