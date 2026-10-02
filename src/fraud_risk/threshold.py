"""Cost-sensitive threshold selection on the validation set, then frozen for the test set.

Flagging means "do not auto-approve" (REVIEW or BLOCK). A false positive is a legitimate payment
that was flagged; a false negative is a fraud that was approved.
"""

from dataclasses import dataclass

from fraud_risk.config import CostConfig, ThresholdConfig
from fraud_risk.evaluation import BinaryMetrics, binary_metrics


def expected_cost(false_positives: int, false_negatives: int, costs: CostConfig) -> float:
    return false_positives * costs.false_positive + false_negatives * costs.false_negative


def threshold_grid(config: ThresholdConfig) -> list[float]:
    steps = round((config.grid_stop - config.grid_start) / config.grid_step)
    return [round(config.grid_start + i * config.grid_step, 6) for i in range(steps + 1)]


@dataclass(frozen=True)
class ThresholdCandidate:
    metrics: BinaryMetrics
    expected_cost: float

    @property
    def threshold(self) -> float:
        return self.metrics.threshold


@dataclass(frozen=True)
class FrozenThresholds:
    """Chosen once on validation data; never re-tuned on test data."""

    review_threshold: float
    block_threshold: float | None  # None: no score is confident enough to block automatically
    validation_expected_cost: float
    validation_metrics: BinaryMetrics
    review_at_grid_edge: bool


def sweep_thresholds(y_true, scores, grid: list[float], costs: CostConfig) -> list[ThresholdCandidate]:
    candidates = []
    for threshold in grid:
        metrics = binary_metrics(y_true, scores, threshold)
        candidates.append(ThresholdCandidate(metrics, expected_cost(metrics.false_positives, metrics.false_negatives, costs)))
    return candidates


def select_min_cost(candidates: list[ThresholdCandidate]) -> ThresholdCandidate:
    """Lowest expected cost; ties go to the higher threshold (fewer customers interrupted)."""
    if not candidates:
        raise ValueError("no threshold candidates")
    return min(candidates, key=lambda c: (c.expected_cost, -c.threshold))


def select_block_threshold(
    candidates: list[ThresholdCandidate], review_threshold: float, min_precision: float, min_support: int
) -> float | None:
    """Lowest threshold at or above the review threshold whose validation precision meets the target."""
    for c in sorted(candidates, key=lambda c: c.threshold):
        flagged = c.metrics.true_positives + c.metrics.false_positives
        if c.threshold >= review_threshold and flagged >= min_support and c.metrics.precision >= min_precision:
            return c.threshold
    return None


def freeze_thresholds(y_validation, validation_scores, config: ThresholdConfig, costs: CostConfig) -> FrozenThresholds:
    grid = threshold_grid(config)
    candidates = sweep_thresholds(y_validation, validation_scores, grid, costs)
    best = select_min_cost(candidates)
    return FrozenThresholds(
        review_threshold=best.threshold,
        block_threshold=select_block_threshold(
            candidates, best.threshold, config.block_min_precision, config.block_min_support
        ),
        validation_expected_cost=best.expected_cost,
        validation_metrics=best.metrics,
        review_at_grid_edge=best.threshold in (grid[0], grid[-1]),
    )
