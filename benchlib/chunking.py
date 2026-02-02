# benchlib/chunking.py
"""Shared chunking strategy for splitting long token sequences into
fixed-length overlapping windows.  The tokenizer is model-specific,
but the chunking logic is identical across all models."""

from typing import List
import torch


def chunk_token_ids(
    input_ids: torch.Tensor,
    max_length: int = 512,
    overlap: int = 64,
) -> List[torch.Tensor]:
    """Split a 1-D tensor of token IDs into overlapping chunks.

    Returns a list of 1-D tensors, each at most *max_length* long.
    The last chunk may be shorter (caller is responsible for padding).
    """
    L = int(input_ids.shape[0])
    if L <= max_length:
        return [input_ids]

    stride = max(1, min(overlap, max_length - 1))
    chunks: List[torch.Tensor] = []
    start = 0
    while start < L:
        end = min(start + max_length, L)
        chunks.append(input_ids[start:end])
        if end == L:
            break
        start = end - stride
        if start < 0:
            start = 0
    return chunks


def tokenize_and_chunk(
    text: str,
    tokenizer,
    max_length: int = 512,
    overlap: int = 64,
) -> List[dict]:
    """Tokenize *text* with a HF tokenizer, then chunk into windows.

    Returns a list of dicts, each with 'input_ids' and 'attention_mask'
    tensors of shape (1, max_length), padded as needed.
    """
    tok = tokenizer(
        text,
        add_special_tokens=True,
        truncation=False,
        padding=False,
        return_tensors="pt",
    )

    input_ids = tok["input_ids"][0]  # (L,)
    chunks = chunk_token_ids(input_ids, max_length=max_length, overlap=overlap)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    result = []
    for c in chunks:
        attn = torch.ones_like(c)
        if c.shape[0] < max_length:
            pad_len = max_length - c.shape[0]
            c = torch.cat([c, torch.full((pad_len,), pad_id, dtype=c.dtype)])
            attn = torch.cat([attn, torch.zeros((pad_len,), dtype=attn.dtype)])
        result.append({
            "input_ids": c.unsqueeze(0),       # (1, max_length)
            "attention_mask": attn.unsqueeze(0),  # (1, max_length)
        })
    return result
