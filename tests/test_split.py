import pandas as pd
import pytest

from fraud_risk.split import chronological_split


def timeline(n: int, duplicate_at: tuple[int, ...] = ()) -> pd.DataFrame:
    stamps = list(pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC"))
    for i in duplicate_at:
        stamps[i] = stamps[i - 1]
    # Shuffled input: the split must not depend on incoming row order.
    return pd.DataFrame({"transaction_id": [f"TX-{i:03d}" for i in range(n)], "timestamp": stamps}).sample(frac=1, random_state=0)


def test_partitions_are_strictly_ordered_in_time():
    s = chronological_split(timeline(100), 0.70, 0.15)
    assert s.train["timestamp"].max() < s.validation["timestamp"].min()
    assert s.validation["timestamp"].max() < s.test["timestamp"].min()
    assert (len(s.train), len(s.validation), len(s.test)) == (70, 15, 15)


def test_no_transaction_in_more_than_one_split():
    s = chronological_split(timeline(100), 0.70, 0.15)
    ids = [set(p["transaction_id"]) for p in (s.train, s.validation, s.test)]
    assert not (ids[0] & ids[1]) and not (ids[1] & ids[2]) and not (ids[0] & ids[2])
    assert sum(map(len, ids)) == 100


def test_equal_timestamps_at_boundary_stay_together():
    s = chronological_split(timeline(100, duplicate_at=(70, 71)), 0.70, 0.15)
    assert s.train["timestamp"].max() < s.validation["timestamp"].min()
    assert len(s.train) == 72


@pytest.mark.parametrize(("train", "validation"), [(0.0, 0.15), (0.7, 0.3), (0.9, 0.2)])
def test_invalid_fractions_rejected(train, validation):
    with pytest.raises(ValueError):
        chronological_split(timeline(100), train, validation)


def test_too_little_data_fails_instead_of_returning_empty_split():
    with pytest.raises(ValueError, match="empty partition"):
        chronological_split(timeline(3), 0.70, 0.15)
