import pandas as pd

from conftest import frame, tx
from fraud_risk.data import generate_transactions
from fraud_risk.features import FEATURE_COLUMNS, build_features
from fraud_risk.validation import validate_transactions


def by_id(features: pd.DataFrame) -> pd.DataFrame:
    return features.set_index("transaction_id")


def test_features_do_not_change_when_future_rows_change(small_config):
    valid = validate_transactions(generate_transactions(small_config.data, small_config.seed).astype(str)).valid
    cutoff = valid["timestamp"].iloc[len(valid) // 2]
    past = valid[valid["timestamp"] < cutoff]

    future = valid[valid["timestamp"] >= cutoff].copy()
    future["amount"] = future["amount"] * 100
    future["device_id"] = "D-NEW"
    future["country"] = "XH"

    full = build_features(pd.concat([past, future])).iloc[: len(past)]
    only_past = build_features(past)
    pd.testing.assert_frame_equal(full, only_past)


def test_new_device_is_relative_to_customer_history():
    f = by_id(
        build_features(
            frame(
                tx("T1", "C1", "2026-01-01T10:00Z", device="D-A"),
                tx("T2", "C1", "2026-01-02T10:00Z", device="D-A"),
                tx("T3", "C1", "2026-01-03T10:00Z", device="D-B"),
                tx("T4", "C2", "2026-01-04T10:00Z", device="D-A"),
            )
        )
    )
    assert f["new_device"].to_dict() == {"T1": 1, "T2": 0, "T3": 1, "T4": 1}


def test_transactions_last_1h_counts_only_earlier_events_in_window():
    f = by_id(
        build_features(
            frame(
                tx("T3", "C1", "2026-01-01T10:59:59Z", merchant="gift_cards"),
                tx("T1", "C1", "2026-01-01T10:00:00Z"),
                tx("T2", "C1", "2026-01-01T10:30:00Z", merchant="electronics"),
                tx("T4", "C1", "2026-01-01T11:00:00Z"),
                tx("T5", "C2", "2026-01-01T10:45:00Z"),
            )
        )
    )
    assert f["transactions_last_1h"].to_dict() == {"T1": 0, "T2": 1, "T3": 2, "T4": 2, "T5": 0}
    assert f.loc["T3", "merchant_switches_last_1h"] == 2
    assert f.loc["T4", "merchant_switches_last_1h"] == 2


def test_amount_ratio_uses_prior_mean_only():
    f = by_id(
        build_features(
            frame(
                tx("T1", "C1", "2026-01-01T10:00Z", amount=10),
                tx("T2", "C1", "2026-01-02T10:00Z", amount=30),
                tx("T3", "C1", "2026-01-03T10:00Z", amount=200),
            )
        )
    )
    assert f["amount_vs_customer_mean"].to_dict() == {"T1": 1.0, "T2": 3.0, "T3": 10.0}


def test_country_change_and_night_flags():
    f = by_id(
        build_features(
            frame(
                tx("T1", "C1", "2026-01-01T03:00Z", country="XA"),
                tx("T2", "C1", "2026-01-01T12:00Z", country="XD"),
                tx("T3", "C1", "2026-01-01T23:00Z", country="XD"),
            )
        )
    )
    assert f["country_changed"].to_dict() == {"T1": 0, "T2": 1, "T3": 0}
    assert f["is_night"].to_dict() == {"T1": 1, "T2": 0, "T3": 0}


def test_output_has_expected_columns_and_no_label_in_features():
    out = build_features(frame(tx("T1", "C1", "2026-01-01T10:00Z", is_fraud=1)))
    assert list(out.columns) == ["transaction_id", "timestamp", "is_fraud", *FEATURE_COLUMNS]
    assert "is_fraud" not in FEATURE_COLUMNS
