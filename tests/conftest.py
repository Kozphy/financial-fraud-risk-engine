from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from fraud_risk.config import Config, load_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def config() -> Config:
    return load_config(ROOT / "config.toml")


@pytest.fixture(scope="session")
def small_config(config: Config) -> Config:
    """Same generator behaviour, fewer rows, so end-to-end tests stay fast."""
    return replace(config, data=replace(config.data, n_transactions=3000, n_customers=150))


def tx(transaction_id, customer_id, timestamp, amount=50.0, merchant="grocery", country="XA", device="D-1", is_fraud=0):
    return {
        "transaction_id": transaction_id,
        "customer_id": customer_id,
        "timestamp": pd.Timestamp(timestamp),
        "amount": amount,
        "merchant_category": merchant,
        "country": country,
        "device_id": device,
        "payment_method": "card_present",
        "is_fraud": is_fraud,
    }


def frame(*rows) -> pd.DataFrame:
    return pd.DataFrame(list(rows))
