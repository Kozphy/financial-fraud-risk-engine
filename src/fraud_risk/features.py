"""Causal per-customer features.

Rows are processed in event order (timestamp, then transaction_id). Each row's features are
computed from the customer's *earlier* events only, and the history is updated afterwards, so a
row can never see itself or anything that happens later. Features use no labels.
"""

from collections import deque
from dataclasses import dataclass, field

import pandas as pd

FEATURE_COLUMNS = (
    "amount",
    "hour",
    "is_night",
    "amount_vs_customer_mean",
    "transactions_last_1h",
    "merchant_switches_last_1h",
    "new_device",
    "country_changed",
)
WINDOW = pd.Timedelta(hours=1)
NIGHT_HOURS = range(0, 6)


@dataclass
class _History:
    recent: deque = field(default_factory=deque)  # (timestamp, merchant_category) within WINDOW
    devices: set = field(default_factory=set)
    last_country: str | None = None
    amount_sum: float = 0.0
    count: int = 0


def build_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """Return transaction_id, timestamp, is_fraud plus FEATURE_COLUMNS, in event order."""
    df = transactions.sort_values(["timestamp", "transaction_id"], kind="mergesort").reset_index(drop=True)
    histories: dict[str, _History] = {}
    rows = []
    for tx in df.itertuples(index=False):
        h = histories.setdefault(tx.customer_id, _History())
        while h.recent and h.recent[0][0] <= tx.timestamp - WINDOW:
            h.recent.popleft()

        rows.append(
            {
                "amount": tx.amount,
                "hour": tx.timestamp.hour,
                "is_night": int(tx.timestamp.hour in NIGHT_HOURS),
                "amount_vs_customer_mean": tx.amount / (h.amount_sum / h.count) if h.count else 1.0,
                "transactions_last_1h": len(h.recent),
                "merchant_switches_last_1h": sum(m != tx.merchant_category for _, m in h.recent),
                "new_device": int(tx.device_id not in h.devices),
                "country_changed": int(h.last_country is not None and tx.country != h.last_country),
            }
        )

        h.recent.append((tx.timestamp, tx.merchant_category))
        h.devices.add(tx.device_id)
        h.last_country = tx.country
        h.amount_sum += tx.amount
        h.count += 1

    features = pd.DataFrame(rows, columns=list(FEATURE_COLUMNS))
    return pd.concat([df[["transaction_id", "timestamp", "is_fraud"]], features], axis=1)
