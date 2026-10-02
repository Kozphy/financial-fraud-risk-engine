import pandas as pd
import pytest

from fraud_risk.audit import read_jsonl
from fraud_risk.data import generate_transactions
from fraud_risk.pipeline import format_summary, main, run_pipeline


@pytest.fixture(scope="module")
def data_path(small_config, tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "transactions.csv"
    df = generate_transactions(small_config.data, small_config.seed)
    bad = df.iloc[[0]].assign(transaction_id="TX-BAD", amount="-1")
    pd.concat([df, bad]).to_csv(path, index=False)
    return path


@pytest.fixture(scope="module")
def run(small_config, data_path, tmp_path_factory):
    out = tmp_path_factory.mktemp("run")
    result = run_pipeline(small_config, data_path, out / "evidence", out / "output")
    return result, out


def test_end_to_end_demo_completes(run, small_config):
    result, _ = run
    assert result.rows["input"] == small_config.data.n_transactions + 1
    assert result.rejected == {"non_positive_amount": 1}
    assert result.rows["train"] + result.rows["validation"] + result.rows["test"] == result.rows["valid"]
    assert sum(result.decision_counts.values()) == result.rows["test"]
    assert result.test_ranking["pr_auc"] > result.test_ranking["fraud_rate"]


def test_one_decision_record_per_test_transaction_with_frozen_thresholds(run):
    result, out = run
    records = read_jsonl(out / "evidence" / "decisions.jsonl")
    assert len(records) == result.rows["test"]
    assert len({r["transaction_id"] for r in records}) == len(records)
    assert {r["review_threshold"] for r in records} == {result.selected.thresholds.review_threshold}
    assert {r["run_id"] for r in records} == {result.run_id}


def test_review_queue_contains_exactly_the_review_decisions(run):
    result, out = run
    queue = pd.read_csv(out / "output" / "review_queue.csv")
    assert len(queue) == result.decision_counts["REVIEW"]
    assert list(queue.columns) == ["transaction_id", "fraud_probability", "reason"]
    assert (out / "output" / "rejected_rows.csv").is_file()


def test_experiment_records_keep_test_results_for_selected_model_only(run):
    result, out = run
    experiments = read_jsonl(out / "evidence" / "experiments.jsonl")
    assert len(experiments) == len(result.runs)
    for e in experiments:
        assert (e["test"] is not None) == e["selected"]
        assert e["dataset_version"] == result.dataset_version
        assert e["validation"]["review_threshold"] > 0
    assert sum(e["selected"] for e in experiments) == 1


def test_rerun_is_reproducible_and_appends_evidence(run, small_config, data_path):
    first, out = run
    second = run_pipeline(small_config, data_path, out / "evidence", out / "output")
    assert second.run_id != first.run_id
    assert second.test_metrics == first.test_metrics
    assert second.decision_counts == first.decision_counts
    assert second.policy == first.policy
    assert len(read_jsonl(out / "evidence" / "decisions.jsonl")) == 2 * first.rows["test"]


def test_summary_and_cli(run, small_config, data_path, tmp_path, capsys):
    result, _ = run
    summary = format_summary(result)
    for text in ("PR-AUC", "Frozen review threshold", "APPROVE", "REVIEW", "BLOCK", "always-APPROVE"):
        assert text in summary
    assert main(["--data", str(data_path), "--evidence-dir", str(tmp_path / "ev"), "--output-dir", str(tmp_path / "out")]) == 0
    assert "Audit evidence appended" in capsys.readouterr().out


def test_missing_data_file_explains_how_to_generate(small_config, tmp_path):
    with pytest.raises(FileNotFoundError, match="generate_demo_data.py"):
        run_pipeline(small_config, tmp_path / "missing.csv", tmp_path / "ev", tmp_path / "out")
