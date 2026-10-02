"""Append-only JSONL evidence. Files are only ever opened in append mode; nothing is rewritten."""

import json
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fraud_risk.policy import Policy, PolicyDecision


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def decision_record(
    *,
    run_id: str,
    transaction_id: str,
    probability: float,
    outcome: PolicyDecision,
    model_name: str,
    model_version: str,
    policy: Policy,
    decided_at: str,
) -> dict[str, Any]:
    """One record per decision. Contains synthetic IDs and scores only, no customer attributes or labels."""
    return {
        "timestamp": decided_at,
        "run_id": run_id,
        "transaction_id": transaction_id,
        "fraud_probability": round(float(probability), 6),
        "decision": outcome.decision.value,
        "reason": outcome.reason.value,
        "model_name": model_name,
        "model_version": model_version,
        "policy_version": policy.version,
        "review_threshold": policy.review_threshold,
        "block_threshold": policy.block_threshold,
    }
