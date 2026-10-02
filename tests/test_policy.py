import pytest

from fraud_risk.policy import Decision, Policy, Reason, decide

POLICY = Policy(version="policy-test", review_threshold=0.35, block_threshold=0.75)


@pytest.mark.parametrize(
    ("probability", "decision", "reason"),
    [
        (0.10, Decision.APPROVE, Reason.BELOW_REVIEW),
        (0.61, Decision.REVIEW, Reason.BETWEEN_THRESHOLDS),
        (0.90, Decision.BLOCK, Reason.AT_OR_ABOVE_BLOCK),
    ],
)
def test_score_bands(probability, decision, reason):
    outcome = decide(probability, POLICY)
    assert (outcome.decision, outcome.reason) == (decision, reason)


def test_boundaries_take_the_stricter_decision():
    assert decide(0.35, POLICY).decision is Decision.REVIEW
    assert decide(0.3499999, POLICY).decision is Decision.APPROVE
    assert decide(0.75, POLICY).decision is Decision.BLOCK
    assert decide(0.7499999, POLICY).decision is Decision.REVIEW


def test_without_block_threshold_high_scores_go_to_humans():
    policy = Policy("p", review_threshold=0.2, block_threshold=None)
    outcome = decide(0.99, policy)
    assert (outcome.decision, outcome.reason) == (Decision.REVIEW, Reason.BLOCK_NOT_AUTOMATED)


@pytest.mark.parametrize("bad", [float("nan"), 1.5, None])
def test_invalid_score_never_auto_approves_or_blocks(bad):
    outcome = decide(bad, POLICY)
    assert (outcome.decision, outcome.reason) == (Decision.REVIEW, Reason.INVALID_SCORE)


@pytest.mark.parametrize(("review", "block"), [(0.8, 0.5), (0.5, 1.2)])
def test_inconsistent_thresholds_rejected(review, block):
    with pytest.raises(ValueError):
        Policy("p", review, block)
