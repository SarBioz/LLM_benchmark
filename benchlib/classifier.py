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
    """
    if method not in _CLASSIFIERS:
        raise ValueError(f"Unknown classifier '{method}'. Choose from: {list(_CLASSIFIERS)}")
    clf = _CLASSIFIERS[method](seed)
    clf.fit(X_train, y_train)
    return clf
