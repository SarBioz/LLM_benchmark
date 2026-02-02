# classifier_head.py
from typing import Dict, Any
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier


def train_logreg(X_train: np.ndarray, y_train: np.ndarray, seed: int = 42) -> Pipeline:
    # scaler helps a lot for linear models on embeddings
    clf = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("logreg", LogisticRegression(
            max_iter=2000,
            random_state=seed,
            class_weight="balanced",
            n_jobs=None,
        )),
    ])
    clf.fit(X_train, y_train)
    return clf


def train_linear_svm(X_train: np.ndarray, y_train: np.ndarray, seed: int = 42) -> Pipeline:
    clf = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("svm", LinearSVC(
            random_state=seed,
            class_weight="balanced",
        )),
    ])
    clf.fit(X_train, y_train)
    return clf


def train_mlp(X_train: np.ndarray, y_train: np.ndarray, seed: int = 42) -> Pipeline:
    clf = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(256,),
            activation="relu",
            alpha=1e-4,
            max_iter=200,
            random_state=seed,
        )),
    ])
    clf.fit(X_train, y_train)
    return clf
