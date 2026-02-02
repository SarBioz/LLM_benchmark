# Refactoring Progress Tracker

> Last updated: 2026-02-01

## Phase 1: Create `benchlib/` shared library
| Task | Status | Notes |
|------|--------|-------|
| 1.1 Create `benchlib/__init__.py` | DONE | |
| 1.2 Move `data_io.py` → `benchlib/data_io.py` | DONE | |
| 1.3 Merge labels → `benchlib/labels.py` | DONE | |
| 1.4 Move `splits.py` → `benchlib/splits.py` | DONE | |
| 1.5 Create `benchlib/chunking.py` | DONE | Extracted from embedder.py |
| 1.6 Create `benchlib/feature_extraction.py` | DONE | Frozen model + pooling → feature vectors |
| 1.7 Create `benchlib/classifier.py` | DONE | Unified classifier head (not loss.py) |
| 1.8 Move `eval_utils.py` → `benchlib/eval_utils.py` | DONE | |

## Phase 2: Central config
| Task | Status | Notes |
|------|--------|-------|
| 2.1 Create `config.py` | DONE | |

## Phase 3: Per-model files
| Task | Status | Notes |
|------|--------|-------|
| 3.1 Create `models/__init__.py` with registry | DONE | |
| 3.2 `models/bert_tiny.py` | DONE | |
| 3.3 `models/distilbert.py` | DONE | |
| 3.4 `models/albert.py` | DONE | |
| 3.5 `models/distilroberta.py` | DONE | |

## Phase 4: New entrypoint
| Task | Status | Notes |
|------|--------|-------|
| 4.1 Create `run_benchmark.py` | DONE | |

## Phase 5: Cleanup & verify
| Task | Status | Notes |
|------|--------|-------|
| 5.1 Delete old top-level files | DONE | |
| 5.2 Update `Readme.md` | DONE | |
| 5.3 Verify equivalent results | PENDING | Run `python run_benchmark.py` with full data |
