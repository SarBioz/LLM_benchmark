# benchlib/feature_extraction.py
"""Run a frozen pretrained model on tokenized/chunked text and pool
the hidden states into a single feature vector per transcript."""

from typing import List
import numpy as np
import torch

from benchlib.chunking import tokenize_and_chunk


def _mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool last hidden states respecting attention mask.

    last_hidden:    (B, T, H)
    attention_mask: (B, T)
    returns:        (B, H)
    """
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden)  # (B, T, 1)
    summed = (last_hidden * mask).sum(dim=1)                   # (B, H)
    counts = mask.sum(dim=1).clamp(min=1e-9)                   # (B, 1)
    return summed / counts


@torch.no_grad()
def extract_features_single(
    text: str,
    model,
    tokenizer,
    device: str,
    max_length: int = 512,
    overlap: int = 64,
) -> np.ndarray:
    """Raw text -> single feature vector (H,) for one transcript.

    Steps: tokenize -> chunk -> frozen forward pass -> mean pool within
    each chunk -> mean pool across chunks.
    """
    chunks = tokenize_and_chunk(text, tokenizer, max_length=max_length, overlap=overlap)

    chunk_features = []
    for c in chunks:
        ids = c["input_ids"].to(device)
        attn = c["attention_mask"].to(device)
        out = model(input_ids=ids, attention_mask=attn)
        pooled = _mean_pool(out.last_hidden_state, attn)  # (1, H)
        chunk_features.append(pooled[0].cpu().numpy())

    # Average across chunks -> one feature vector per transcript
    return np.mean(np.stack(chunk_features, axis=0), axis=0)


def extract_features_batch(
    texts: List[str],
    model,
    tokenizer,
    device: str,
    max_length: int = 512,
    overlap: int = 64,
    verbose_every: int = 50,
) -> np.ndarray:
    """Raw texts -> feature matrix (N, H) for N transcripts."""
    features = []
    for i, t in enumerate(texts):
        features.append(
            extract_features_single(t, model, tokenizer, device, max_length, overlap)
        )
        if verbose_every and (i + 1) % verbose_every == 0:
            print(f"  Extracted features {i + 1}/{len(texts)}")
    return np.vstack(features)
