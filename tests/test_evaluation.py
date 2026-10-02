import numpy as np
import pytest

from fraud_risk.evaluation import binary_metrics, ranking_metrics

Y = [1, 1, 0, 0, 0, 0, 0, 0, 0, 1]
P = [0.9, 0.4, 0.8, 0.1, 0.1, 0.2, 0.05, 0.3, 0.1, 0.7]


def test_confusion_matrix_and_rates():
    m = binary_metrics(Y, P, 0.5)
    assert (m.true_positives, m.false_positives, m.true_negatives, m.false_negatives) == (2, 1, 6, 1)
    assert m.precision == pytest.approx(2 / 3)
    assert m.recall == pytest.approx(2 / 3)
    assert m.f1 == pytest.approx(2 / 3)


def test_threshold_is_inclusive():
    assert binary_metrics([1], [0.5], 0.5).true_positives == 1


def test_accuracy_rewards_a_useless_model_under_imbalance():
    y = np.array([1] * 2 + [0] * 98)
    never_flag = binary_metrics(y, np.zeros(100), 0.5)
    assert never_flag.accuracy == 0.98
    assert never_flag.recall == 0.0 and never_flag.precision == 0.0


def test_ranking_metrics_perfect_and_random():
    perfect = ranking_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert perfect["pr_auc"] == 1.0 and perfect["roc_auc"] == 1.0
    constant = ranking_metrics([0, 0, 0, 1], [0.5, 0.5, 0.5, 0.5])
    assert constant["roc_auc"] == 0.5
    assert constant["pr_auc"] == pytest.approx(0.25)
