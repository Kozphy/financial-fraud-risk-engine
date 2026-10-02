"""Model construction and baseline-vs-candidate selection.

Every model is a scikit-learn pipeline with fit / predict_proba. No class re-weighting: outputs stay
usable as probabilities, and the imbalance is handled where it belongs, in the cost-sensitive
threshold chosen on validation data.
"""

from enum import StrEnum
from typing import assert_never

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler


class ModelName(StrEnum):
    LOGISTIC_REGRESSION = "logistic_regression"
    RANDOM_FOREST = "random_forest"


BASELINE = ModelName.LOGISTIC_REGRESSION


def build_model(name: ModelName, seed: int) -> Pipeline:
    match name:
        case ModelName.LOGISTIC_REGRESSION:
            return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        case ModelName.RANDOM_FOREST:
            return make_pipeline(
                RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=seed, n_jobs=1)
            )
        case _ as unreachable:
            assert_never(unreachable)


def choose_model(validation_costs: dict[ModelName, float], min_relative_improvement: float) -> tuple[ModelName, str]:
    """Keep the simpler baseline unless a candidate cuts validation expected cost by a clear margin.

    Uses validation results only. The test set never influences which model is selected.
    """
    baseline_cost = validation_costs[BASELINE]
    candidates = {name: cost for name, cost in validation_costs.items() if name is not BASELINE}
    if not candidates:
        return BASELINE, "baseline is the only model"
    best = min(candidates, key=candidates.__getitem__)
    if baseline_cost <= 0:
        return BASELINE, "baseline already has zero validation cost"
    improvement = (baseline_cost - candidates[best]) / baseline_cost
    if improvement >= min_relative_improvement:
        return best, f"{best} cuts validation expected cost by {improvement:.0%} (required: {min_relative_improvement:.0%})"
    return BASELINE, (
        f"kept simpler baseline: {best} changes validation expected cost by {-improvement:+.0%} "
        f"(promotion requires a {min_relative_improvement:.0%} reduction)"
    )
