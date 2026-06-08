import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


def recall_at_k(y_true, y_score, k=100):
    """
    Calcula Recall@K para priorización de alertas AML.

    Se ordenan las transacciones por score de riesgo descendente
    y se calcula cuántos ilícitos reales aparecen dentro del top K.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    k = min(k, len(y_true))

    top_k_idx = np.argsort(y_score)[::-1][:k]
    positives_total = np.sum(y_true == 1)

    if positives_total == 0:
        return 0.0

    positives_at_k = np.sum(y_true[top_k_idx] == 1)

    return positives_at_k / positives_total


def evaluate_binary_classifier(y_true, y_score, threshold=0.5, k_values=None):
    """
    Evalúa un clasificador binario para escenarios de fraude/AML.
    """
    if k_values is None:
        k_values = [50, 100, 500, 1000]

    y_pred = (y_score >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "pr_auc": average_precision_score(y_true, y_score),
        "roc_auc": roc_auc_score(y_true, y_score),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            zero_division=0
        ),
    }

    for k in k_values:
        metrics[f"recall_at_{k}"] = recall_at_k(y_true, y_score, k=k)

    return metrics
