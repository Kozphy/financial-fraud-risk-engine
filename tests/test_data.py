import pandas as pd

from fraud_risk.data import COLUMNS, dataset_version, generate_transactions, load_transactions
from fraud_risk.validation import validate_transactions


def test_generator_is_deterministic(small_config):
    first = generate_transactions(small_config.data, small_config.seed)
    second = generate_transactions(small_config.data, small_config.seed)
    pd.testing.assert_frame_equal(first, second)


def test_different_seed_changes_data(small_config):
    first = generate_transactions(small_config.data, small_config.seed)
    other = generate_transactions(small_config.data, small_config.seed + 1)
    assert not first.equals(other)


def test_generator_matches_schema_size_and_prevalence(small_config):
    df = generate_transactions(small_config.data, small_config.seed)
    assert tuple(df.columns) == COLUMNS
    assert len(df) == small_config.data.n_transactions
    assert df["is_fraud"].sum() == round(small_config.data.n_transactions * small_config.data.fraud_rate)
    assert df["transaction_id"].is_unique


def test_generated_data_passes_validation_unchanged(small_config, tmp_path):
    path = tmp_path / "tx.csv"
    generate_transactions(small_config.data, small_config.seed).to_csv(path, index=False)
    result = validate_transactions(load_transactions(path))
    assert result.rejected.empty
    assert len(result.valid) == small_config.data.n_transactions


def test_dataset_version_tracks_content(tmp_path):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    a.write_text("x\n1\n", encoding="utf-8")
    b.write_text("x\n2\n", encoding="utf-8")
    assert dataset_version(a).startswith("sha256:")
    assert dataset_version(a) != dataset_version(b)
