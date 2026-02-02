# embedder.py
from typing import List, Tuple
import numpy as np
import torch
from hf_encoder import EncoderBundle


def _mean_pool_last_hidden(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    last_hidden: (B, T, H)
    attention_mask: (B, T)
    returns: (B, H)
    """
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden)  # (B, T, 1)
    summed = (last_hidden * mask).sum(dim=1)                  # (B, H)
    counts = mask.sum(dim=1).clamp(min=1e-9)                  # (B, 1)
    return summed / counts


def _chunk_input_ids(input_ids_1d: torch.Tensor, max_len: int, stride: int) -> List[torch.Tensor]:
    """
    input_ids_1d: (L,)
    Returns list of (max_len,) chunks (last chunk may be shorter; we will pad later).
    """
    L = int(input_ids_1d.shape[0])
    if L <= max_len:
        return [input_ids_1d]

    chunks = []
    start = 0
    while start < L:
        end = min(start + max_len, L)
        chunks.append(input_ids_1d[start:end])
        if end == L:
            break
        start = end - stride  # overlap
        if start < 0:
            start = 0
    return chunks


@torch.no_grad()
def embed_one_text(
    bundle: EncoderBundle,
    text: str,
    max_length: int = 512,
    overlap_tokens: int = 64,
) -> np.ndarray:
    """
    One transcript -> one embedding vector (H,).
    Minimal preprocessing: NONE.
    """
    tok = bundle.tokenizer(
        text,
        add_special_tokens=True,
        truncation=False,     # we handle chunking ourselves
        padding=False,
        return_tensors="pt",
    )

    input_ids = tok["input_ids"][0]  # (L,)
    # choose stride
    stride = max(1, min(overlap_tokens, max_length - 1))

    chunks = _chunk_input_ids(input_ids, max_len=max_length, stride=stride)

    chunk_embs = []
    for c in chunks:
        # pad to max_length for batching consistency
        attn = torch.ones_like(c)
        if c.shape[0] < max_length:
            pad_len = max_length - c.shape[0]
            pad_id = bundle.tokenizer.pad_token_id
            if pad_id is None:
                # some models (e.g., GPT-like) might not have pad; but you listed encoder models, so ok.
                # Still make it safe:
                pad_id = 0
            c = torch.cat([c, torch.full((pad_len,), pad_id, dtype=c.dtype)])
            attn = torch.cat([attn, torch.zeros((pad_len,), dtype=attn.dtype)])

        c = c.unsqueeze(0).to(bundle.device)        # (1, T)
        attn = attn.unsqueeze(0).to(bundle.device)  # (1, T)

        out = bundle.model(input_ids=c, attention_mask=attn)
        last_hidden = out.last_hidden_state  # (1, T, H)

        pooled = _mean_pool_last_hidden(last_hidden, attn)  # (1, H)
        chunk_embs.append(pooled[0].detach().cpu().numpy())

    # pool across chunks -> one transcript embedding
    emb = np.mean(np.stack(chunk_embs, axis=0), axis=0)
    return emb


def embed_dataframe(
    bundle: EncoderBundle,
    texts: List[str],
    max_length: int = 512,
    overlap_tokens: int = 64,
    verbose_every: int = 50,
) -> np.ndarray:
    embs = []
    for i, t in enumerate(texts):
        embs.append(embed_one_text(bundle, t, max_length=max_length, overlap_tokens=overlap_tokens))
        if verbose_every and (i + 1) % verbose_every == 0:
            print(f"Embedded {i+1}/{len(texts)}")
    return np.vstack(embs)
