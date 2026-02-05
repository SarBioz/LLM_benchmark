# benchlib/linguistic_features.py
"""
Linguistic Feature Extraction Using BERT (bert-tiny)

This module extracts both BERT-based and traditional linguistic features
from speech transcripts for cognitive impairment detection.

Features extracted:
    BERT-based:
        - Sentence embeddings (masked mean pooling)
        - Semantic coherence (cosine similarity between consecutive sentences)
        - Sentiment score (requires sentiment head)
        - Transcript-level embedding

    Traditional (text-based):
        - Grammar error rate
        - Average word length
        - Vocabulary richness (TTR, GTTR)
"""

import re
from typing import List, Dict, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity


# =============================================================================
# 1. SENTENCE EMBEDDING (BERT POOLING)
# =============================================================================

def masked_mean_pool(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute masked mean pooling over token embeddings.

    Formula:
        e_i = sum(m_it * h_it) / sum(m_it)

    Args:
        hidden_states: (B, T, H) - Token embeddings from BERT
        attention_mask: (B, T) - 1 for valid tokens, 0 for padding

    Returns:
        (B, H) - Sentence embeddings
    """
    # Expand mask for broadcasting: (B, T) -> (B, T, 1)
    mask = attention_mask.unsqueeze(-1).float()

    # Masked sum of embeddings: (B, H)
    masked_sum = (hidden_states * mask).sum(dim=1)

    # Count of valid tokens: (B, 1)
    token_counts = mask.sum(dim=1).clamp(min=1e-9)

    # Mean pooled embedding: (B, H)
    return masked_sum / token_counts


def get_sentence_embedding(
    sentence: str,
    model,
    tokenizer,
    device: str,
    max_length: int = 512
) -> np.ndarray:
    """
    Get a single sentence embedding using BERT + masked mean pooling.

    Args:
        sentence: Input sentence text
        model: BERT model
        tokenizer: BERT tokenizer
        device: 'cuda' or 'cpu'
        max_length: Maximum sequence length

    Returns:
        (H,) numpy array - Sentence embedding
    """
    # Tokenize
    inputs = tokenizer(
        sentence,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding="max_length"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)

    # Mean pool
    embedding = masked_mean_pool(
        outputs.last_hidden_state,
        inputs["attention_mask"]
    )

    return embedding[0].cpu().numpy()


def get_all_sentence_embeddings(
    sentences: List[str],
    model,
    tokenizer,
    device: str,
    max_length: int = 512
) -> np.ndarray:
    """
    Get embeddings for all sentences in a transcript.

    Args:
        sentences: List of sentences
        model: BERT model
        tokenizer: BERT tokenizer
        device: 'cuda' or 'cpu'
        max_length: Maximum sequence length

    Returns:
        (N, H) numpy array - N sentence embeddings
    """
    embeddings = []
    for sent in sentences:
        if sent.strip():  # Skip empty sentences
            emb = get_sentence_embedding(sent, model, tokenizer, device, max_length)
            embeddings.append(emb)

    if not embeddings:
        # Return zero embedding if no valid sentences
        hidden_size = model.config.hidden_size
        return np.zeros((1, hidden_size))

    return np.stack(embeddings, axis=0)


# =============================================================================
# 2. SEMANTIC COHERENCE (BERT-BASED)
# =============================================================================

def compute_semantic_coherence(sentence_embeddings: np.ndarray) -> float:
    """
    Compute semantic coherence as average cosine similarity between
    consecutive sentence embeddings.

    Formula:
        Coherence = (1 / (N-1)) * sum_{i=1}^{N-1} cos(e_i, e_{i+1})

    Args:
        sentence_embeddings: (N, H) array of sentence embeddings

    Returns:
        Coherence score (float between -1 and 1, typically 0 to 1)
    """
    N = len(sentence_embeddings)

    if N < 2:
        return 1.0  # Single sentence is perfectly coherent with itself

    coherence_scores = []
    for i in range(N - 1):
        # Reshape for sklearn cosine_similarity
        e_i = sentence_embeddings[i].reshape(1, -1)
        e_i_plus_1 = sentence_embeddings[i + 1].reshape(1, -1)

        # Cosine similarity
        sim = cosine_similarity(e_i, e_i_plus_1)[0, 0]
        coherence_scores.append(sim)

    return float(np.mean(coherence_scores))


# =============================================================================
# 3. SENTIMENT SCORE
# =============================================================================

class SentimentHead(nn.Module):
    """
    Simple linear sentiment classifier head.

    Formula:
        z = w^T * e + b
        p = sigmoid(z)
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: (B, H) sentence embeddings

        Returns:
            (B,) sentiment probabilities
        """
        logits = self.linear(embeddings).squeeze(-1)
        return torch.sigmoid(logits)


def compute_sentiment_score(
    sentence_embeddings: np.ndarray,
    sentiment_head: Optional[SentimentHead] = None,
    device: str = "cpu"
) -> float:
    """
    Compute transcript-level sentiment score.

    Formula:
        Sentiment = (1/N) * sum_{i=1}^{N} p_i

    Args:
        sentence_embeddings: (N, H) array of sentence embeddings
        sentiment_head: Trained sentiment classifier (optional)
        device: 'cuda' or 'cpu'

    Returns:
        Sentiment score (float between 0 and 1)

    Note:
        If no sentiment_head is provided, returns a placeholder based on
        embedding statistics (for demonstration purposes).
    """
    if sentiment_head is None:
        # Placeholder: use normalized L2 norm as proxy
        # In practice, you would train a sentiment classifier
        norms = np.linalg.norm(sentence_embeddings, axis=1)
        normalized = (norms - norms.min()) / (norms.max() - norms.min() + 1e-9)
        return float(np.mean(normalized))

    # Use trained sentiment head
    embeddings_tensor = torch.tensor(sentence_embeddings, dtype=torch.float32).to(device)

    with torch.no_grad():
        sentiment_probs = sentiment_head(embeddings_tensor)

    return float(sentiment_probs.mean().cpu().numpy())


# =============================================================================
# 4. GRAMMAR AND LANGUAGE ERROR RATE
# =============================================================================

def compute_grammar_error_rate(
    text: str,
    use_language_tool: bool = False
) -> Tuple[float, int, int]:
    """
    Compute grammar error rate.

    Formula:
        GrammarErrorRate = E / W

    Args:
        text: Full transcript text
        use_language_tool: If True, use language_tool_python (requires installation)
                          If False, use simple heuristics

    Returns:
        Tuple of (error_rate, num_errors, num_words)
    """
    words = text.split()
    W = len(words)

    if W == 0:
        return 0.0, 0, 0

    if use_language_tool:
        try:
            import language_tool_python
            tool = language_tool_python.LanguageTool('en-US')
            matches = tool.check(text)
            E = len(matches)
            tool.close()
        except ImportError:
            print("Warning: language_tool_python not installed. Using heuristics.")
            E = _count_errors_heuristic(text)
    else:
        E = _count_errors_heuristic(text)

    error_rate = E / W
    return error_rate, E, W


def _count_errors_heuristic(text: str) -> int:
    """
    Simple heuristic-based error detection.
    Counts potential grammar/spelling issues using regex patterns.

    This is a simplified placeholder - use language_tool_python for accuracy.
    """
    errors = 0

    # Pattern 1: Double spaces
    errors += len(re.findall(r'  +', text))

    # Pattern 2: Missing capitalization after period
    errors += len(re.findall(r'\. [a-z]', text))

    # Pattern 3: Double words (e.g., "the the")
    errors += len(re.findall(r'\b(\w+)\s+\1\b', text.lower()))

    # Pattern 4: Missing space after punctuation
    errors += len(re.findall(r'[.!?][A-Za-z]', text))

    # Pattern 5: Multiple punctuation
    errors += len(re.findall(r'[.!?]{2,}', text))

    return errors


# =============================================================================
# 5. AVERAGE WORD LENGTH
# =============================================================================

def compute_avg_word_length(text: str) -> Tuple[float, float]:
    """
    Compute average word length.

    Formula:
        AvgWordLength = (1/W) * sum_{j=1}^{W} len(w_j)

    Args:
        text: Full transcript text

    Returns:
        Tuple of (mean_length, std_length)
    """
    # Extract words (alphabetic characters only)
    words = re.findall(r'\b[a-zA-Z]+\b', text)

    if not words:
        return 0.0, 0.0

    lengths = [len(w) for w in words]

    return float(np.mean(lengths)), float(np.std(lengths))


# =============================================================================
# 6. TRANSCRIPT-LEVEL EMBEDDING (BERT-BASED)
# =============================================================================

def compute_transcript_embedding(sentence_embeddings: np.ndarray) -> np.ndarray:
    """
    Compute transcript-level embedding by averaging sentence embeddings.

    Formula:
        E = (1/N) * sum_{i=1}^{N} e_i

    Args:
        sentence_embeddings: (N, H) array of sentence embeddings

    Returns:
        (H,) transcript embedding
    """
    return np.mean(sentence_embeddings, axis=0)


# =============================================================================
# 7. VOCABULARY RICHNESS (LEXICAL FEATURES)
# =============================================================================

def compute_vocabulary_richness(text: str) -> Dict[str, float]:
    """
    Compute vocabulary richness metrics.

    Metrics:
        TTR (Type-Token Ratio) = V / N
        GTTR (Guiraud's Root TTR) = V / sqrt(N)

    Where:
        N = total number of word tokens
        V = number of unique word types

    Args:
        text: Full transcript text

    Returns:
        Dictionary with vocabulary richness metrics
    """
    # Extract words (lowercase for type counting)
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())

    N = len(words)  # Total tokens
    V = len(set(words))  # Unique types

    if N == 0:
        return {
            "total_tokens": 0,
            "unique_types": 0,
            "ttr": 0.0,
            "gttr": 0.0
        }

    ttr = V / N
    gttr = V / np.sqrt(N)

    return {
        "total_tokens": N,
        "unique_types": V,
        "ttr": ttr,
        "gttr": gttr
    }


# =============================================================================
# MAIN FEATURE EXTRACTION FUNCTION
# =============================================================================

def extract_all_linguistic_features(
    text: str,
    model,
    tokenizer,
    device: str,
    sentiment_head: Optional[SentimentHead] = None,
    use_language_tool: bool = False,
    max_length: int = 512
) -> Dict[str, any]:
    """
    Extract all linguistic features from a transcript.

    Args:
        text: Full transcript text
        model: BERT model
        tokenizer: BERT tokenizer
        device: 'cuda' or 'cpu'
        sentiment_head: Optional trained sentiment classifier
        use_language_tool: Whether to use language_tool_python for grammar checking
        max_length: Maximum sequence length for BERT

    Returns:
        Dictionary containing all extracted features:
            BERT-based:
                - sentence_embeddings: (N, H) array
                - transcript_embedding: (H,) array
                - semantic_coherence: float
                - sentiment_score: float

            Traditional:
                - grammar_error_rate: float
                - num_grammar_errors: int
                - num_words: int
                - avg_word_length: float
                - std_word_length: float
                - ttr: float (Type-Token Ratio)
                - gttr: float (Guiraud's Root TTR)
                - total_tokens: int
                - unique_types: int
    """
    # Split into sentences
    sentences = _split_into_sentences(text)

    # -------------------------------------------------------------------------
    # BERT-BASED FEATURES
    # -------------------------------------------------------------------------

    # 1. Sentence embeddings
    sentence_embeddings = get_all_sentence_embeddings(
        sentences, model, tokenizer, device, max_length
    )

    # 2. Transcript-level embedding
    transcript_embedding = compute_transcript_embedding(sentence_embeddings)

    # 3. Semantic coherence
    semantic_coherence = compute_semantic_coherence(sentence_embeddings)

    # 4. Sentiment score
    sentiment_score = compute_sentiment_score(
        sentence_embeddings, sentiment_head, device
    )

    # -------------------------------------------------------------------------
    # TRADITIONAL (TEXT-BASED) FEATURES
    # -------------------------------------------------------------------------

    # 5. Grammar error rate
    grammar_error_rate, num_errors, num_words = compute_grammar_error_rate(
        text, use_language_tool
    )

    # 6. Average word length
    avg_word_length, std_word_length = compute_avg_word_length(text)

    # 7. Vocabulary richness
    vocab_richness = compute_vocabulary_richness(text)

    # -------------------------------------------------------------------------
    # COMPILE ALL FEATURES
    # -------------------------------------------------------------------------

    features = {
        # BERT-based features
        "sentence_embeddings": sentence_embeddings,
        "transcript_embedding": transcript_embedding,
        "semantic_coherence": semantic_coherence,
        "sentiment_score": sentiment_score,
        "num_sentences": len(sentences),

        # Traditional features
        "grammar_error_rate": grammar_error_rate,
        "num_grammar_errors": num_errors,
        "num_words": num_words,
        "avg_word_length": avg_word_length,
        "std_word_length": std_word_length,

        # Vocabulary richness
        "ttr": vocab_richness["ttr"],
        "gttr": vocab_richness["gttr"],
        "total_tokens": vocab_richness["total_tokens"],
        "unique_types": vocab_richness["unique_types"],
    }

    return features


def get_feature_vector(features: Dict[str, any]) -> Tuple[np.ndarray, List[str]]:
    """
    Convert feature dictionary to a flat feature vector for classification.

    Args:
        features: Dictionary from extract_all_linguistic_features()

    Returns:
        Tuple of (feature_vector, feature_names)
    """
    # Scalar features
    scalar_features = [
        ("semantic_coherence", features["semantic_coherence"]),
        ("sentiment_score", features["sentiment_score"]),
        ("num_sentences", features["num_sentences"]),
        ("grammar_error_rate", features["grammar_error_rate"]),
        ("avg_word_length", features["avg_word_length"]),
        ("std_word_length", features["std_word_length"]),
        ("ttr", features["ttr"]),
        ("gttr", features["gttr"]),
        ("total_tokens", features["total_tokens"]),
        ("unique_types", features["unique_types"]),
    ]

    names = [name for name, _ in scalar_features]
    values = [value for _, value in scalar_features]

    # Add transcript embedding
    transcript_emb = features["transcript_embedding"]
    emb_names = [f"emb_{i}" for i in range(len(transcript_emb))]

    names.extend(emb_names)
    values.extend(transcript_emb.tolist())

    return np.array(values), names


def _split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences using simple rules.

    For better results, consider using nltk.sent_tokenize().
    """
    # Try to use NLTK if available
    try:
        import nltk
        return nltk.sent_tokenize(text)
    except ImportError:
        pass

    # Fallback: simple regex-based splitting
    # Split on .!? followed by space and capital letter or end of string
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])$', text)

    # Filter empty strings and strip whitespace
    sentences = [s.strip() for s in sentences if s.strip()]

    # If no sentences found, treat entire text as one sentence
    if not sentences:
        sentences = [text]

    return sentences
