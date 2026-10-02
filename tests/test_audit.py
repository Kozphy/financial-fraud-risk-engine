from fraud_risk.audit import append_jsonl, decision_record, read_jsonl
from fraud_risk.policy import Policy, decide

POLICY = Policy("policy-v1", review_threshold=0.35, block_threshold=0.75)
REQUIRED = {
    "timestamp",
    "run_id",
    "transaction_id",
    "fraud_probability",
    "decision",
    "reason",
    "model_name",
    "model_version",
    "policy_version",
    "review_threshold",
    "block_threshold",
}


def record(tx_id="TX-1", p=0.61):
    return decision_record(
        run_id="run-1",
        transaction_id=tx_id,
        probability=p,
        outcome=decide(p, POLICY),
        model_name="logistic_regression",
        model_version="logistic-v1",
        policy=POLICY,
        decided_at="2026-10-02T00:00:00+00:00",
    )


def test_one_decision_creates_one_jsonl_line(tmp_path):
    path = tmp_path / "evidence" / "decisions.jsonl"
    assert append_jsonl(path, [record()]) == 1
    assert len(path.read_text(encoding="utf-8").splitlines()) == 1


def test_record_has_required_metadata_and_no_customer_data():
    r = record()
    assert set(r) == REQUIRED
    assert (r["decision"], r["reason"]) == ("REVIEW", "score_between_review_and_block_threshold")
    assert not {"customer_id", "device_id", "amount", "country", "is_fraud"} & set(r)


def test_appends_never_rewrite_existing_records(tmp_path):
    path = tmp_path / "decisions.jsonl"
    append_jsonl(path, [record("TX-1", 0.1)])
    first_bytes = path.read_bytes()
    append_jsonl(path, [record("TX-2", 0.9), record("TX-3", 0.5)])
    assert path.read_bytes().startswith(first_bytes)
    assert [r["transaction_id"] for r in read_jsonl(path)] == ["TX-1", "TX-2", "TX-3"]
    assert [r["decision"] for r in read_jsonl(path)] == ["APPROVE", "BLOCK", "REVIEW"]
