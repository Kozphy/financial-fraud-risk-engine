from fraud_risk.config import CostConfig, ThresholdConfig
from fraud_risk.threshold import (
    expected_cost,
    freeze_thresholds,
    select_block_threshold,
    select_min_cost,
    sweep_thresholds,
    threshold_grid,
)

COSTS = CostConfig(false_positive=5, false_negative=500)
Y = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
P = [0.95, 0.60, 0.15, 0.70, 0.30, 0.10, 0.08, 0.05, 0.02, 0.01]


def grid_config(start=0.05, stop=0.95, step=0.05, min_precision=0.9, min_support=1) -> ThresholdConfig:
    return ThresholdConfig(start, stop, step, min_precision, min_support)


def test_expected_cost_formula():
    assert expected_cost(false_positives=10, false_negatives=3, costs=COSTS) == 10 * 5 + 3 * 500


def test_grid_is_inclusive_and_free_of_float_drift():
    grid = threshold_grid(grid_config())
    assert grid[0] == 0.05 and grid[-1] == 0.95 and len(grid) == 19
    assert 0.3 in grid


def test_lower_threshold_raises_recall():
    candidates = {c.threshold: c for c in sweep_thresholds(Y, P, [0.1, 0.5, 0.9], COSTS)}
    assert candidates[0.1].metrics.recall > candidates[0.5].metrics.recall > candidates[0.9].metrics.recall


def test_optimizer_selects_minimum_cost_candidate():
    candidates = sweep_thresholds(Y, P, threshold_grid(grid_config()), COSTS)
    best = select_min_cost(candidates)
    assert best.expected_cost == min(c.expected_cost for c in candidates)
    # Catching the 0.15 fraud costs 2 extra false positives (10) instead of a missed fraud (500).
    assert best.threshold == 0.15 and best.metrics.false_negatives == 0


def test_optimizer_follows_the_cost_ratio():
    cheap_misses = CostConfig(false_positive=500, false_negative=5)
    best = select_min_cost(sweep_thresholds(Y, P, threshold_grid(grid_config()), cheap_misses))
    assert best.metrics.false_positives == 0


def test_ties_prefer_higher_threshold():
    candidates = sweep_thresholds([1, 0], [0.9, 0.1], [0.2, 0.5, 0.8], COSTS)
    assert select_min_cost(candidates).threshold == 0.8


def test_block_threshold_requires_precision_and_support():
    candidates = sweep_thresholds(Y, P, threshold_grid(grid_config()), COSTS)
    assert select_block_threshold(candidates, 0.15, min_precision=0.9, min_support=1) == 0.75
    assert select_block_threshold(candidates, 0.15, min_precision=0.9, min_support=2) is None


def test_block_threshold_never_below_review_threshold():
    candidates = sweep_thresholds([1, 1], [0.9, 0.9], [0.1, 0.5], COSTS)
    assert select_block_threshold(candidates, 0.5, 0.9, 1) == 0.5


def test_freeze_reports_grid_edge():
    frozen = freeze_thresholds(Y, P, grid_config(), COSTS)
    assert (frozen.review_threshold, frozen.block_threshold, frozen.review_at_grid_edge) == (0.15, 0.75, False)
    edge = freeze_thresholds([1, 0], [0.06, 0.0], grid_config(), COSTS)
    assert (edge.review_threshold, edge.review_at_grid_edge) == (0.05, True)
