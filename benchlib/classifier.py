# benchlib/classifier.py
"""Unified classifier head for the benchmark.  All pretrained models
feed their feature vectors into the *same* classifier, ensuring a
fair comparison.  The classifier type is selected once (in config)
and applied identically to every model."""

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier


_CLASSIFIERS = {
    "logreg": lambda seed: Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("logreg", LogisticRegression(
            max_iter=2000,
            random_state=seed,
            class_weight="balanced",
        )),
    ]),
    "linear_svm": lambda seed: Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("svm", LinearSVC(
            random_state=seed,
            class_weight="balanced",
        )),
    ]),
    "mlp": lambda seed: Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(256,),
            activation="relu",
            alpha=1e-4,
            max_iter=200,
            random_state=seed,
        )),
    ]),
}


def train_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    method: str = "logreg",
    seed: int = 42,
) -> Pipeline:
    """Train a classifier on feature vectors.

    Parameters
    ----------
    X_train : (N, H) feature matrix from a frozen pretrained model.
    y_train : (N,) binary labels.
    method  : one of "logreg", "linear_svm", "mlp".
    seed    : random seed for reproducibility.

    Returns
    -------
    Fitted sklearn Pipeline (scaler + classifier).

    Usage
    -----
    clf = train_classifier(X_train, y_train)

    # Get probability scores (0-1):
    scores = get_impairment_scores(clf, X_test)

    # Get binary predictions:
    predictions = clf.predict(X_test)
    """
    if method not in _CLASSIFIERS:
        raise ValueError(f"Unknown classifier '{method}'. Choose from: {list(_CLASSIFIERS)}")
    clf = _CLASSIFIERS[method](seed)
    clf.fit(X_train, y_train)
    return clf


def get_impairment_scores(clf, X: np.ndarray) -> np.ndarray:
    """
    Get cognitive impairment probability scores (0-1) for each sample.

    Parameters
    ----------
    clf : Trained classifier (from train_classifier)
    X   : (N, H) feature matrix

    Returns
    -------
    scores : (N,) array of probability scores
             0.0 = definitely NC (Normal Control)
             1.0 = definitely Impaired (MCI/AD)
             0.5 = uncertain

    Example
    -------
    >>> scores = get_impairment_scores(clf, X_test)
    >>> print(scores)
    [0.12, 0.87, 0.45, 0.93, ...]  # Probability of impairment

    >>> # Apply custom threshold
    >>> threshold = 0.5
    >>> predictions = (scores >= threshold).astype(int)
    """
    if hasattr(clf, 'predict_proba'):
        # Logistic Regression, MLP - have probability output
        return clf.predict_proba(X)[:, 1]
    elif hasattr(clf, 'decision_function'):
        # LinearSVC - use decision function, convert to pseudo-probability
        from scipy.special import expit  # sigmoid
        decision = clf.decision_function(X)
        return expit(decision)
    else:
        # Fallback: just return predictions as 0/1
        return clf.predict(X).astype(float)


def predict_with_threshold(clf, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Get binary predictions using a custom probability threshold.

    Parameters
    ----------
    clf       : Trained classifier
    X         : (N, H) feature matrix
    threshold : Probability threshold (default 0.5)
                Lower threshold = more sensitive (catches more impaired)
                Higher threshold = more specific (fewer false positives)

    Returns
    -------
    predictions : (N,) array of binary predictions (0 or 1)

    Example
    -------
    >>> # High sensitivity (catch more impaired, even if some false positives)
    >>> preds = predict_with_threshold(clf, X_test, threshold=0.3)

    >>> # High specificity (fewer false positives, may miss some impaired)
    >>> preds = predict_with_threshold(clf, X_test, threshold=0.7)
    """
    scores = get_impairment_scores(clf, X)
    return (scores >= threshold).astype(int)
