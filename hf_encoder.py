# hf_encoder.py
from dataclasses import dataclass
from typing import Tuple
import torch
from transformers import AutoTokenizer, AutoModel


@dataclass
class EncoderBundle:
    model_name: str
    tokenizer: any
    model: any
    device: str


def load_frozen_encoder(model_name: str, device: str = None) -> EncoderBundle:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModel.from_pretrained(model_name)

    # Freeze
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    model.to(device)

    return EncoderBundle(model_name=model_name, tokenizer=tokenizer, model=model, device=device)
