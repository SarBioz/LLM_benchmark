# benchlib/linguistic_features.py
"""
Extract 5 linguistic features from text transcripts based on Mobtahej et al. (2024):
"Transformer-based Deep Learning Architecture Improves Detection of Associations
between Spontaneous Speech Language Markers and Cognition"

Features:
1. Vocabulary Richness (Type-Token Ratio) - p=0.009 in paper
2. Average Word Length - p=0.005 in paper
3. Semantic Coherence (BERT-enhanced) - p=0.047 in paper
4. Syntactic Complexity (Mean Length of Utterance)
5. Content Density

The key innovation is using BERT embeddings to compute semantic coherence,
which captures deeper semantic relationships between sentences.
"""

import re
from typing import List, Optional, Tuple
import numpy as np
import torch

# POS tagging for content density
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag


# Feature names for reference and logging
LINGUISTIC_FEATURE_NAMES = [
    "vocabulary_richness",      # Type-Token Ratio
    "avg_word_length",          # Average word length in characters
    "semantic_coherence",       # BERT-based sentence similarity
    "syntactic_complexity",     # Mean Length of Utterance
    "content_density",          # Ratio of content words
]


def _ensure_nltk_data():
    """Download required NLTK data if not present."""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab', quiet=True)
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('averaged_perceptron_tagger', quiet=True)
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger_eng')
    except LookupError:
        nltk.download('averaged_perceptron_tagger_eng', quiet=True)


def _tokenize_words(text: str) -> List[str]:
    """Tokenize text into lowercase alphabetic words."""
    words = word_tokenize(text.lower())
    return [w for w in words if w.isalpha()]


def _count_pos_categories(pos_tags: List[Tuple[str, str]]) -> dict:
    """Count words by POS category using Penn Treebank tags.

    Content words: nouns, verbs, adjectives, adverbs
    """
    counts = {
        'nouns': 0,      # NN, NNS, NNP, NNPS
        'verbs': 0,      # VB, VBD, VBG, VBN, VBP, VBZ
        'adjectives': 0, # JJ, JJR, JJS
        'adverbs': 0,    # RB, RBR, RBS
    }

    for word, tag in pos_tags:
        if tag.startswith('NN'):
            counts['nouns'] += 1
        elif tag.startswith('VB'):
            counts['verbs'] += 1
        elif tag.startswith('JJ'):
            counts['adjectives'] += 1
        elif tag.startswith('RB'):
            counts['adverbs'] += 1

    return counts


def compute_vocabulary_richness(words: List[str]) -> float:
    """Feature 1: Type-Token Ratio (TTR).

    Measures lexical diversity: unique_words / total_words.
    Lower TTR in AD patients indicates reduced vocabulary usage.
    """
    if len(words) == 0:
        return 0.0
    unique_words = len(set(words))
    return unique_words / len(words)


def compute_avg_word_length(words: List[str]) -> float:
    """Feature 2: Average Word Length.

    Measures average number of characters per word.
    Significant predictor (p=0.005) in Mobtahej et al.
    """
    if len(words) == 0:
        return 0.0
    total_chars = sum(len(w) for w in words)
    return total_chars / len(words)


def compute_syntactic_complexity(text: str, words: List[str]) -> float:
    """Feature 4: Mean Length of Utterance (MLU).

    Measures syntactic complexity: total_words / num_sentences.
    Shorter MLU in AD patients indicates simpler sentence structures.
    """
    sentences = sent_tokenize(text)
    num_sentences = max(len(sentences), 1)
    return len(words) / num_sentences


def compute_content_density(words: List[str]) -> float:
    """Feature 5: Content Density.

    Ratio of content words (nouns, verbs, adjectives, adverbs) to total words.
    Measures semantic richness of speech.
    """
    if len(words) == 0:
        return 0.0

    pos_tags = pos_tag(words)
    pos_counts = _count_pos_categories(pos_tags)

    content_words = (
        pos_counts['nouns'] +
        pos_counts['verbs'] +
        pos_counts['adjectives'] +
        pos_counts['adverbs']
    )
    return content_words / len(words)


@torch.no_grad()
def compute_semantic_coherence(
    text: str,
    model,
    tokenizer,
    device: str,
    max_length: int = 128,
) -> float:
    """Feature 3: Semantic Coherence (BERT-enhanced).

    Computes average cosine similarity between consecutive sentence embeddings.
    This is the key BERT-enhanced feature from Mobtahej et al. that captures
    how well sentences connect semantically.

    Higher coherence = more logically connected discourse.
    Lower coherence in AD patients indicates fragmented thought patterns.
    """
    sentences = sent_tokenize(text)

    # Need at least 2 sentences to compute coherence
    if len(sentences) < 2:
        return 1.0  # Single sentence is maximally coherent with itself

    # Get BERT embeddings for each sentence
    sentence_embeddings = []
    for sent in sentences:
        # Tokenize with truncation for individual sentences
        inputs = tokenizer(
            sent,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = model(**inputs)
        # Mean pool the last hidden state
        last_hidden = outputs.last_hidden_state  # (1, T, H)
        attention_mask = inputs["attention_mask"]  # (1, T)

        mask = attention_mask.unsqueeze(-1).float()  # (1, T, 1)
        summed = (last_hidden * mask).sum(dim=1)     # (1, H)
        counts = mask.sum(dim=1).clamp(min=1e-9)     # (1, 1)
        pooled = summed / counts                      # (1, H)

        sentence_embeddings.append(pooled[0].cpu().numpy())

    # Compute cosine similarity between consecutive sentences
    similarities = []
    for i in range(len(sentence_embeddings) - 1):
        emb1 = sentence_embeddings[i]
        emb2 = sentence_embeddings[i + 1]

        # Cosine similarity
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)

        if norm1 > 0 and norm2 > 0:
            similarity = dot_product / (norm1 * norm2)
            similarities.append(similarity)

    if len(similarities) == 0:
        return 1.0

    return float(np.mean(similarities))


def extract_basic_linguistic_features(text: str) -> np.ndarray:
    """Extract 4 basic linguistic features (without BERT).

    Returns: np.ndarray of shape (4,) containing:
        [0] Vocabulary Richness (TTR)
        [1] Average Word Length
        [2] Syntactic Complexity (MLU)
        [3] Content Density

    Note: Semantic coherence requires BERT and is computed separately.
    """
    _ensure_nltk_data()

    words = _tokenize_words(text)

    if len(words) == 0:
        return np.zeros(4, dtype=np.float32)

    features = np.array([
        compute_vocabulary_richness(words),
        compute_avg_word_length(words),
        compute_syntactic_complexity(text, words),
        compute_content_density(words),
    ], dtype=np.float32)

    return features


@torch.no_grad()
def extract_linguistic_features(
    text: str,
    model,
    tokenizer,
    device: str,
) -> np.ndarray:
    """Extract all 5 linguistic features including BERT-enhanced semantic coherence.

    Based on Mobtahej et al. (2024) methodology.

    Returns: np.ndarray of shape (5,) containing:
        [0] Vocabulary Richness (TTR) - p=0.009
        [1] Average Word Length - p=0.005
        [2] Semantic Coherence (BERT) - p=0.047
        [3] Syntactic Complexity (MLU)
        [4] Content Density
    """
    _ensure_nltk_data()

    words = _tokenize_words(text)

    if len(words) == 0:
        return np.zeros(5, dtype=np.float32)

    # Basic features (no BERT needed)
    vocab_richness = compute_vocabulary_richness(words)
    avg_word_len = compute_avg_word_length(words)
    syntactic_complexity = compute_syntactic_complexity(text, words)
    content_density = compute_content_density(words)

    # BERT-enhanced semantic coherence
    semantic_coherence = compute_semantic_coherence(text, model, tokenizer, device)

    features = np.array([
        vocab_richness,
        avg_word_len,
        semantic_coherence,
        syntactic_complexity,
        content_density,
    ], dtype=np.float32)

    return features


def extract_linguistic_features_batch(
    texts: List[str],
    model,
    tokenizer,
    device: str,
    verbose_every: int = 50,
) -> np.ndarray:
    """Extract all 5 linguistic features for a batch of texts.

    Returns: np.ndarray of shape (N, 5)
    """
    _ensure_nltk_data()

    features = []
    for i, text in enumerate(texts):
        feat = extract_linguistic_features(text, model, tokenizer, device)
        features.append(feat)

        if verbose_every and (i + 1) % verbose_every == 0:
            print(f"    Linguistic features: {i + 1}/{len(texts)}")

    return np.vstack(features)
