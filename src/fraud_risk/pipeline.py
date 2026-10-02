"""End-to-end demo: data -> validation -> features -> model -> threshold -> policy -> evidence.

Order of use of the data partitions:
  train       -> fit models
  validation  -> compare models, choose and freeze thresholds
  test        -> scored once, by the selected model with the frozen thresholds
"""

import argparse
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.pipeline import Pipeline

from fraud_risk.audit import append_jsonl, decision_record, utc_now
from fraud_risk.config import DEFAULT_CONFIG_PATH, Config, load_config
from fraud_risk.data import dataset_version, load_transactions
from fraud_risk.evaluation import BinaryMetrics, binary_metrics, ranking_metrics
from fraud_risk.features import FEATURE_COLUMNS, build_features
from fraud_risk.models import ModelName, build_model, choose_model
from fraud_risk.policy import Decision, Policy, decide
from fraud_risk.split import chronological_split
from fraud_risk.threshold import FrozenThresholds, expected_cost, freeze_thresholds
from fraud_risk.validation import validate_transactions

LABEL = "is_fraud"
DECISIONS_FILE = "decisions.jsonl"
EXPERIMENTS_FILE = "experiments.jsonl"


@dataclass(frozen=True)
class ModelRun:
    name: ModelName
    version: str
    model: Pipeline
    validation_ranking: dict[str, float]
    thresholds: FrozenThresholds


@dataclass(frozen=True)
class PipelineResult:
    run_id: str
    dataset_version: str
    rows: dict[str, int]
    rejected: dict[str, int]
    runs: list[ModelRun]
    selected: ModelRun
    selection_reason: str
    policy: Policy
    test_ranking: dict[str, float]
    test_metrics: BinaryMetrics
    test_expected_cost: float
    always_approve_accuracy: float
    decision_counts: dict[str, int]
    fraud_by_decision: dict[str, int]
    review_threshold_at_grid_edge: bool
    evidence_paths: list[Path]


def run_pipeline(config: Config, data_path: Path, evidence_dir: Path, output_dir: Path) -> PipelineResult:
    run_id = uuid.uuid4().hex
    raw = load_transactions(data_path)
    checked = validate_transactions(raw)
    if checked.valid.empty:
        raise ValueError("no valid transactions after validation")
    output_dir.mkdir(parents=True, exist_ok=True)
    checked.rejected.to_csv(output_dir / "rejected_rows.csv", index=False, lineterminator="\n")

    split = chronological_split(build_features(checked.valid), config.split.train_fraction, config.split.validation_fraction)
    X = list(FEATURE_COLUMNS)

    runs = []
    for name in ModelName:
        model = build_model(name, config.seed).fit(split.train[X], split.train[LABEL])
        scores = model.predict_proba(split.validation[X])[:, 1]
        runs.append(
            ModelRun(
                name=name,
                version=config.model_versions[name],
                model=model,
                validation_ranking=ranking_metrics(split.validation[LABEL], scores),
                thresholds=freeze_thresholds(split.validation[LABEL], scores, config.thresholds, config.costs),
            )
        )
    selected_name, selection_reason = choose_model(
        {run.name: run.thresholds.validation_expected_cost for run in runs}, config.min_relative_cost_improvement
    )
    selected = next(run for run in runs if run.name is selected_name)

    # The test partition is used exactly here, once, with everything already frozen.
    test = split.test
    test_scores = selected.model.predict_proba(test[X])[:, 1]
    frozen = selected.thresholds
    test_metrics = binary_metrics(test[LABEL], test_scores, frozen.review_threshold)
    policy = Policy(config.policy_version, frozen.review_threshold, frozen.block_threshold)

    decided_at = utc_now()
    outcomes = [decide(float(p), policy) for p in test_scores]
    records = [
        decision_record(
            run_id=run_id,
            transaction_id=tx_id,
            probability=p,
            outcome=outcome,
            model_name=selected.name.value,
            model_version=selected.version,
            policy=policy,
            decided_at=decided_at,
        )
        for tx_id, p, outcome in zip(test["transaction_id"], test_scores, outcomes, strict=True)
    ]
    decisions_path = evidence_dir / DECISIONS_FILE
    append_jsonl(decisions_path, records)

    review_queue = pd.DataFrame([r for r in records if r["decision"] == Decision.REVIEW])
    review_queue.reindex(columns=["transaction_id", "fraud_probability", "reason"]).sort_values(
        "fraud_probability", ascending=False
    ).to_csv(output_dir / "review_queue.csv", index=False, lineterminator="\n")

    decisions = [o.decision.value for o in outcomes]
    test_ranking = ranking_metrics(test[LABEL], test_scores)
    test_cost = expected_cost(test_metrics.false_positives, test_metrics.false_negatives, config.costs)
    rows = {"input": len(raw), "valid": len(checked.valid), "train": len(split.train), "validation": len(split.validation), "test": len(test)}
    version = dataset_version(data_path)

    experiments_path = evidence_dir / EXPERIMENTS_FILE
    append_jsonl(
        experiments_path,
        [
            _experiment_record(
                run_id, run, version, rows, config, selected_run=run is selected, selection_reason=selection_reason,
                test=(test_ranking, test_metrics, test_cost) if run is selected else None,
            )
            for run in runs
        ],
    )

    return PipelineResult(
        run_id=run_id,
        dataset_version=version,
        rows=rows,
        rejected=checked.reject_counts,
        runs=runs,
        selected=selected,
        selection_reason=selection_reason,
        policy=policy,
        test_ranking=test_ranking,
        test_metrics=test_metrics,
        test_expected_cost=test_cost,
        always_approve_accuracy=1.0 - test_ranking["fraud_rate"],
        decision_counts={d.value: decisions.count(d.value) for d in Decision},
        fraud_by_decision={d.value: int(Counter(zip(decisions, test[LABEL]))[(d.value, 1)]) for d in Decision},
        review_threshold_at_grid_edge=frozen.review_at_grid_edge,
        evidence_paths=[decisions_path, experiments_path],
    )


def _experiment_record(run_id, run, version, rows, config, *, selected_run, selection_reason, test) -> dict[str, Any]:
    t = run.thresholds
    record: dict[str, Any] = {
        "timestamp": utc_now(),
        "experiment_id": f"{run_id}-{run.name.value}",
        "run_id": run_id,
        "model": run.name.value,
        "model_version": run.version,
        "dataset_version": version,
        "train_rows": rows["train"],
        "validation_rows": rows["validation"],
        "test_rows": rows["test"],
        "seed": config.seed,
        "costs": {"false_positive": config.costs.false_positive, "false_negative": config.costs.false_negative},
        "validation": {
            **run.validation_ranking,
            "review_threshold": t.review_threshold,
            "block_threshold": t.block_threshold,
            "expected_cost": t.validation_expected_cost,
            **{k: v for k, v in t.validation_metrics.to_dict().items() if k != "threshold"},
        },
        "selected": selected_run,
        "selection_reason": selection_reason,
        "policy_version": config.policy_version,
        "test": None,
    }
    if test is not None:
        ranking, metrics, cost = test
        record["test"] = {**ranking, "expected_cost": cost, **{k: v for k, v in metrics.to_dict().items() if k != "threshold"}}
    return record


def format_summary(result: PipelineResult) -> str:
    s, t, m = result.selected, result.selected.thresholds, result.test_metrics
    block = f"{t.block_threshold:.3f}" if t.block_threshold is not None else "not automated (no threshold met the precision target)"
    lines = [
        f"Run: {result.run_id}   Dataset: {result.dataset_version}",
        f"Rows: input {result.rows['input']:,}, valid {result.rows['valid']:,}, rejected {sum(result.rejected.values()):,}",
        f"Train rows: {result.rows['train']:,}   Validation rows: {result.rows['validation']:,}   Test rows: {result.rows['test']:,}",
        "",
        "Validation (model selection and threshold tuning only)",
        f"  {'model':<22}{'PR-AUC':>8}{'ROC-AUC':>9}{'threshold':>11}{'exp. cost':>11}",
    ]
    for run in result.runs:
        lines.append(
            f"  {run.name.value:<22}{run.validation_ranking['pr_auc']:>8.3f}{run.validation_ranking['roc_auc']:>9.3f}"
            f"{run.thresholds.review_threshold:>11.3f}{run.thresholds.validation_expected_cost:>11,.0f}"
        )
    lines += [
        f"  Selected: {s.name.value} ({s.version}) - {result.selection_reason}",
        f"  Frozen review threshold: {t.review_threshold:.3f}   Block threshold: {block}",
    ]
    if result.review_threshold_at_grid_edge:
        lines.append("  WARNING: review threshold is at the edge of the grid; widen the grid in config.toml")
    lines += [
        "",
        "Test (untouched until now; frozen model and thresholds)",
        f"  Precision: {m.precision:.3f}   Recall: {m.recall:.3f}   F1: {m.f1:.3f}",
        f"  PR-AUC: {result.test_ranking['pr_auc']:.3f}   ROC-AUC: {result.test_ranking['roc_auc']:.3f}",
        f"  Confusion: TP {m.true_positives}  FP {m.false_positives}  TN {m.true_negatives}  FN {m.false_negatives}",
        f"  Expected cost: {result.test_expected_cost:,.0f}",
        f"  Accuracy: {m.accuracy:.1%} (an always-APPROVE model scores {result.always_approve_accuracy:.1%} and catches no fraud)",
        "",
        f"Decisions (policy {result.policy.version})",
    ]
    for decision, count in result.decision_counts.items():
        lines.append(f"  {decision:<8}{count:>6,}   (of which fraud in hindsight: {result.fraud_by_decision[decision]})")
    lines += ["", "Audit evidence appended:", *(f"  {p.as_posix()}" for p in result.evidence_paths)]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m fraud_risk.pipeline", description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--data", type=Path, default=Path("data/generated/transactions.csv"))
    parser.add_argument("--evidence-dir", type=Path, default=Path("evidence"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/generated"))
    args = parser.parse_args(argv)
    result = run_pipeline(load_config(args.config), args.data, args.evidence_dir, args.output_dir)
    print(format_summary(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
