# benchlib/feature_extraction.py
"""Run a frozen pretrained model on tokenized/chunked text and pool
the hidden states into a single feature vector per transcript.

Extended to support linguistic features based on Mobtahej et al. (2024):
"Transformer-based Deep Learning Architecture Improves Detection of
Associations between Spontaneous Speech Language Markers and Cognition"
"""

from typing import List
import numpy as np
import torch

from benchlib.chunking import tokenize_and_chunk
from benchlib.linguistic_features import extract_linguistic_features_batch


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


def extract_combined_features_batch(
    texts: List[str],
    model,
    tokenizer,
    device: str,
    max_length: int = 512,
    overlap: int = 64,
    use_linguistic: bool = True,
    verbose_every: int = 50,
) -> np.ndarray:
    """Extract BERT embeddings and optionally concatenate linguistic features.

    Based on Mobtahej et al. (2024) methodology for cognitive impairment detection.

    When use_linguistic=True, the output includes 5 linguistic features:
        1. Vocabulary Richness (TTR)
        2. Average Word Length
        3. Semantic Coherence (BERT-enhanced)
        4. Syntactic Complexity (MLU)
        5. Content Density

    Args:
        texts: List of raw transcript texts
        model: Frozen BERT model
        tokenizer: Corresponding tokenizer
        device: 'cuda' or 'cpu'
        max_length: Max tokens per chunk
        overlap: Token overlap between chunks
        use_linguistic: If True, concatenate 5 linguistic features
        verbose_every: Print progress every N samples

    Returns:
        np.ndarray: Feature matrix of shape (N, H) if use_linguistic=False,
                    or (N, H+5) if use_linguistic=True
    """
    # Extract BERT embeddings
    print("  Extracting BERT embeddings...")
    bert_features = extract_features_batch(
        texts, model, tokenizer, device, max_length, overlap, verbose_every
    )

    if not use_linguistic:
        return bert_features

    # Extract linguistic features (including BERT-enhanced semantic coherence)
    print("  Extracting linguistic features (Mobtahej et al. methodology)...")
    ling_features = extract_linguistic_features_batch(
        texts, model, tokenizer, device, verbose_every
    )

    # Concatenate: [BERT embeddings (H,) | Linguistic features (5,)]
    combined = np.hstack([bert_features, ling_features])
    print(f"  Combined features: BERT ({bert_features.shape[1]}) + Linguistic (5) = {combined.shape[1]}")

    return combined
