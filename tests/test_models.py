import pytest

from fraud_risk.data import generate_transactions
from fraud_risk.evaluation import ranking_metrics
from fraud_risk.features import FEATURE_COLUMNS, build_features
from fraud_risk.models import BASELINE, ModelName, build_model, choose_model
from fraud_risk.split import chronological_split
from fraud_risk.validation import validate_transactions

LR, RF = ModelName.LOGISTIC_REGRESSION, ModelName.RANDOM_FOREST


@pytest.mark.parametrize("name", list(ModelName))
def test_model_learns_signal_out_of_time(small_config, name):
    valid = validate_transactions(generate_transactions(small_config.data, small_config.seed).astype(str)).valid
    split = chronological_split(build_features(valid), 0.70, 0.15)
    X, y = list(FEATURE_COLUMNS), "is_fraud"

    model = build_model(name, small_config.seed).fit(split.train[X], split.train[y])
    scores = model.predict_proba(split.validation[X])[:, 1]

    assert ((scores >= 0) & (scores <= 1)).all()
    metrics = ranking_metrics(split.validation[y], scores)
    assert metrics["roc_auc"] > 0.8
    assert metrics["pr_auc"] > 3 * metrics["fraud_rate"]


@pytest.mark.parametrize(
    ("candidate_cost", "expected"),
    [
        (800.0, RF),  # 20% cheaper: clear improvement, promote
        (950.0, LR),  # 5% cheaper: not worth the extra complexity
        (1200.0, LR),  # worse
    ],
)
def test_candidate_is_promoted_only_on_clear_validation_improvement(candidate_cost, expected):
    selected, reason = choose_model({LR: 1000.0, RF: candidate_cost}, min_relative_improvement=0.10)
    assert selected is expected
    assert ("kept simpler baseline" in reason) == (selected is BASELINE)
