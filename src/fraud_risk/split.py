"""Chronological train / validation / test split.

Fraud models score *future* transactions, so they are trained on the past and judged on later
periods. A random split would let the model learn from transactions that happen after the ones it
is tested on. Split points move forward past equal timestamps so no instant straddles two splits.
"""

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ChronologicalSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def chronological_split(df: pd.DataFrame, train_fraction: float, validation_fraction: float) -> ChronologicalSplit:
    if not (0 < train_fraction and 0 < validation_fraction and train_fraction + validation_fraction < 1):
        raise ValueError("fractions must be positive and leave a non-empty test share")

    ordered = df.sort_values(["timestamp", "transaction_id"], kind="mergesort").reset_index(drop=True)
    timestamps = ordered["timestamp"]
    first_cut = _advance_past_ties(timestamps, round(len(ordered) * train_fraction))
    second_cut = _advance_past_ties(timestamps, max(first_cut, round(len(ordered) * (train_fraction + validation_fraction))))

    parts = (ordered.iloc[:first_cut], ordered.iloc[first_cut:second_cut], ordered.iloc[second_cut:])
    if any(part.empty for part in parts):
        raise ValueError(f"split produced an empty partition (rows={len(ordered)})")
    return ChronologicalSplit(*(part.reset_index(drop=True) for part in parts))


def _advance_past_ties(timestamps: pd.Series, cut: int) -> int:
    while 0 < cut < len(timestamps) and timestamps.iloc[cut] == timestamps.iloc[cut - 1]:
        cut += 1
    return cut
