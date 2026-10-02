import pandas as pd
import pytest

from fraud_risk.validation import SchemaError, validate_transactions

GOOD = {
    "transaction_id": "TX-1",
    "customer_id": "C-1",
    "timestamp": "2026-01-01T10:00:00Z",
    "amount": "25.50",
    "merchant_category": "grocery",
    "country": "XA",
    "device_id": "D-1",
    "payment_method": "card_present",
    "is_fraud": "0",
}


def rows(*overrides: dict) -> pd.DataFrame:
    return pd.DataFrame([{**GOOD, "transaction_id": f"TX-{i}", **o} for i, o in enumerate(overrides, start=1)])


def test_missing_column_fails_clearly():
    with pytest.raises(SchemaError, match="missing required columns: device_id, is_fraud"):
        validate_transactions(rows({}).drop(columns=["device_id", "is_fraud"]))


def test_valid_rows_are_typed():
    result = validate_transactions(rows({}))
    assert result.rejected.empty
    row = result.valid.iloc[0]
    assert row["timestamp"] == pd.Timestamp("2026-01-01T10:00:00Z")
    assert row["amount"] == 25.5 and row["is_fraud"] == 0


def test_duplicate_transaction_id_keeps_first_and_rejects_rest():
    df = rows({"amount": "1"}, {"amount": "2"}, {"amount": "3"})
    df["transaction_id"] = ["TX-1", "TX-1", "TX-2"]
    result = validate_transactions(df)
    assert list(result.valid["transaction_id"]) == ["TX-1", "TX-2"]
    assert result.valid.iloc[0]["amount"] == 1.0
    assert result.reject_counts == {"duplicate_transaction_id": 1}


@pytest.mark.parametrize(
    ("override", "reason"),
    [
        ({"timestamp": "not-a-date"}, "invalid_timestamp"),
        ({"timestamp": "2026-02-30T10:00:00Z"}, "invalid_timestamp"),
        ({"timestamp": "2026-01-01T10:00:00"}, "timestamp_without_timezone"),
        ({"amount": "abc"}, "invalid_amount"),
        ({"amount": "-5"}, "non_positive_amount"),
        ({"amount": "0"}, "non_positive_amount"),
        ({"is_fraud": "yes"}, "invalid_label"),
        ({"customer_id": "  "}, "missing_customer_id"),
        ({"transaction_id": ""}, "missing_transaction_id"),
    ],
)
def test_invalid_rows_are_quarantined_with_reason(override, reason):
    result = validate_transactions(rows({}, override))
    assert len(result.valid) == 1
    assert result.reject_counts == {reason: 1}


def test_every_input_row_is_accounted_for_once():
    df = rows({}, {"amount": "x"}, {"timestamp": ""}, {}, {"is_fraud": "2"})
    result = validate_transactions(df)
    assert len(result.valid) + len(result.rejected) == len(df)


def test_offset_timestamps_are_normalized_to_utc_and_sorted():
    result = validate_transactions(
        rows({"timestamp": "2026-01-01T20:00:00+08:00"}, {"timestamp": "2026-01-01T11:00:00Z"})
    )
    assert list(result.valid["timestamp"]) == [pd.Timestamp("2026-01-01T11:00:00Z"), pd.Timestamp("2026-01-01T12:00:00Z")]
