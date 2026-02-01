# Refactoring Progress Tracker

> Last updated: 2026-02-01

## Phase 1: Create `benchlib/` shared library
| Task | Status | Notes |
|------|--------|-------|
| 1.1 Create `benchlib/__init__.py` | NOT STARTED | |
| 1.2 Move `data_io.py` → `benchlib/data_io.py` | NOT STARTED | |
| 1.3 Merge labels → `benchlib/labels.py` | NOT STARTED | |
| 1.4 Move `splits.py` → `benchlib/splits.py` | NOT STARTED | |
| 1.5 Extract tokenizer → `benchlib/tokenizer.py` | NOT STARTED | |
| 1.6 Refactor embedding → `benchlib/embedder.py` | NOT STARTED | |
| 1.7 Create `benchlib/loss.py` | NOT STARTED | |
| 1.8 Move `eval_utils.py` → `benchlib/eval_utils.py` | NOT STARTED | |

## Phase 2: Central config
| Task | Status | Notes |
|------|--------|-------|
| 2.1 Create `config.py` | NOT STARTED | |

## Phase 3: Per-model files
| Task | Status | Notes |
|------|--------|-------|
| 3.1 Create `models/__init__.py` with registry | NOT STARTED | |
| 3.2 `models/bert_tiny.py` | NOT STARTED | |
| 3.3 `models/distilbert.py` | NOT STARTED | |
| 3.4 `models/albert.py` | NOT STARTED | |
| 3.5 `models/distilroberta.py` | NOT STARTED | |

## Phase 4: New entrypoint
| Task | Status | Notes |
|------|--------|-------|
| 4.1 Create `run_benchmark.py` | NOT STARTED | |

## Phase 5: Cleanup
| Task | Status | Notes |
|------|--------|-------|
| 5.1 Delete old top-level files | NOT STARTED | |
| 5.2 Update `Readme.md` | NOT STARTED | |
| 5.3 Verify equivalent results | NOT STARTED | |
