# eval_utils.py
from typing import Dict, Any
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report


def eval_binary(clf, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    y_pred = clf.predict(X)

    out = {
        "accuracy": float(accuracy_score(y, y_pred)),
        "f1": float(f1_score(y, y_pred)),
    }

    # AUC only if model can produce scores
    if hasattr(clf, "predict_proba"):
        p1 = clf.predict_proba(X)[:, 1]
        out["auc"] = float(roc_auc_score(y, p1))
    elif hasattr(clf, "decision_function"):
        s = clf.decision_function(X)
        out["auc"] = float(roc_auc_score(y, s))
    else:
        out["auc"] = None

    out["report"] = classification_report(y, y_pred, digits=4)
    return out
