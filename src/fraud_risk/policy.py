"""Deterministic policy gate: turns a model probability into a business decision.

The model only estimates risk. Whether a payment is approved, reviewed, or blocked is decided here,
from versioned thresholds, so the business rule can be audited and changed without retraining.
"""

import math
from dataclasses import dataclass
from enum import StrEnum


class Decision(StrEnum):
    APPROVE = "APPROVE"
    REVIEW = "REVIEW"  # human escalation queue
    BLOCK = "BLOCK"


class Reason(StrEnum):
    BELOW_REVIEW = "score_below_review_threshold"
    BETWEEN_THRESHOLDS = "score_between_review_and_block_threshold"
    AT_OR_ABOVE_BLOCK = "score_at_or_above_block_threshold"
    BLOCK_NOT_AUTOMATED = "score_at_or_above_review_threshold_block_not_automated"
    INVALID_SCORE = "invalid_model_score_routed_to_review"


@dataclass(frozen=True)
class Policy:
    version: str
    review_threshold: float
    block_threshold: float | None

    def __post_init__(self) -> None:
        if not 0.0 <= self.review_threshold <= 1.0:
            raise ValueError("review_threshold must be within [0, 1]")
        if self.block_threshold is not None and not self.review_threshold <= self.block_threshold <= 1.0:
            raise ValueError("block_threshold must be within [review_threshold, 1]")


@dataclass(frozen=True)
class PolicyDecision:
    decision: Decision
    reason: Reason


def decide(probability: float, policy: Policy) -> PolicyDecision:
    """Thresholds are inclusive: a score equal to a threshold takes the stricter decision.

    A missing or out-of-range score never auto-approves and never auto-blocks: a human decides.
    """
    if not isinstance(probability, (int, float)) or math.isnan(probability) or not 0.0 <= probability <= 1.0:
        return PolicyDecision(Decision.REVIEW, Reason.INVALID_SCORE)
    if policy.block_threshold is not None and probability >= policy.block_threshold:
        return PolicyDecision(Decision.BLOCK, Reason.AT_OR_ABOVE_BLOCK)
    if probability >= policy.review_threshold:
        reason = Reason.BETWEEN_THRESHOLDS if policy.block_threshold is not None else Reason.BLOCK_NOT_AUTOMATED
        return PolicyDecision(Decision.REVIEW, reason)
    return PolicyDecision(Decision.APPROVE, Reason.BELOW_REVIEW)
