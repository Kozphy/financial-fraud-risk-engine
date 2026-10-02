"""Imbalance-aware metrics. Accuracy is reported only to show why it is misleading."""

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


@dataclass(frozen=True)
class BinaryMetrics:
    """Metrics when scores >= threshold are flagged as fraud (REVIEW or BLOCK)."""

    threshold: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    accuracy: float

    def to_dict(self) -> dict:
        return asdict(self)


def binary_metrics(y_true, scores, threshold: float) -> BinaryMetrics:
    y = np.asarray(y_true).astype(bool)
    flagged = np.asarray(scores) >= threshold
    tp = int(np.sum(flagged & y))
    fp = int(np.sum(flagged & ~y))
    fn = int(np.sum(~flagged & y))
    tn = int(np.sum(~flagged & ~y))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return BinaryMetrics(threshold, tp, fp, tn, fn, precision, recall, f1, (tp + tn) / len(y))


def ranking_metrics(y_true, scores) -> dict[str, float]:
    """Threshold-free quality. PR-AUC (average precision) is the primary metric under imbalance."""
    return {
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "fraud_rate": float(np.mean(y_true)),
    }
