"""Deterministic synthetic transaction generator and CSV loading.

The fraud patterns are deliberately simplified so the decision workflow can be demonstrated.
They do not represent real fraud prevalence, behaviour, or any real customer.
"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from fraud_risk.config import DataConfig

COLUMNS = (
    "transaction_id",
    "customer_id",
    "timestamp",
    "amount",
    "merchant_category",
    "country",
    "device_id",
    "payment_method",
    "is_fraud",
)
EVERYDAY_MERCHANTS = ("grocery", "restaurant", "transport", "utilities", "retail", "travel")
HIGH_RISK_MERCHANTS = ("electronics", "gift_cards", "digital_goods")
# ISO 3166 user-assigned codes, so no real country is implied to be a fraud source.
COUNTRIES = ("XA", "XB", "XC", "XD", "XE", "XF", "XG", "XH")
PAYMENT_METHODS = ("card_present", "card_not_present", "wallet", "bank_transfer")
TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def generate_transactions(config: DataConfig, seed: int) -> pd.DataFrame:
    """Return a chronologically ordered synthetic dataset; identical for the same inputs."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp(config.start)
    customers = _make_customers(rng, config.n_customers)

    n_fraud = round(config.n_transactions * config.fraud_rate)
    fraud = _fraud_rows(rng, customers, n_fraud, start, config.days)
    legit = _legit_rows(rng, customers, config.n_transactions - len(fraud), start, config.days)

    df = pd.DataFrame(legit + fraud)
    df = df.sort_values(["timestamp", "customer_id", "amount"], kind="mergesort").reset_index(drop=True)
    width = len(str(len(df)))
    df.insert(0, "transaction_id", [f"TX-{i:0{width}d}" for i in range(1, len(df) + 1)])
    df["timestamp"] = df["timestamp"].dt.strftime(TIMESTAMP_FORMAT)
    return df.loc[:, list(COLUMNS)]


def _make_customers(rng: np.random.Generator, n: int) -> list[dict]:
    customers = []
    for i in range(1, n + 1):
        devices = [f"D-{i:04d}-1"] + ([f"D-{i:04d}-2"] if rng.random() < 0.4 else [])
        customers.append(
            {
                "customer_id": f"C-{i:04d}",
                "home_country": "XA" if rng.random() < 0.85 else str(rng.choice(COUNTRIES[1:])),
                "log_amount": rng.normal(np.log(40), 0.7),
                "devices": devices,
                "merchants": list(rng.choice(EVERYDAY_MERCHANTS, size=3, replace=False)),
                "activity": rng.gamma(2.0, 1.0),
                "new_devices": 0,
            }
        )
    return customers


def _legit_rows(rng, customers, n, start, days) -> list[dict]:
    weights = np.array([c["activity"] for c in customers])
    weights /= weights.sum()
    rows: list[dict] = []
    while len(rows) < n:
        customer = customers[rng.choice(len(customers), p=weights)]
        if rows and rng.random() < 0.04:
            # Legitimate quick follow-up purchase: makes short-window counts noisy on purpose.
            previous = rows[-1]
            customer = next(c for c in customers if c["customer_id"] == previous["customer_id"])
            when = previous["timestamp"] + pd.Timedelta(minutes=float(rng.uniform(3, 50)))
        else:
            day = int(rng.integers(0, days))
            hour = rng.uniform(0, 24) if rng.random() < 0.08 else float(np.clip(rng.normal(14, 3.5), 6, 23.9))
            when = start + pd.Timedelta(days=day, hours=hour)
        rows.append(_legit_transaction(rng, customer, when.floor("s")))
    return rows


def _legit_transaction(rng, customer, when) -> dict:
    # Legitimate look-alikes (new phone, big purchase, travel) keep the problem from being trivial.
    roll = rng.random()
    if roll < 0.01:
        customer["new_devices"] += 1
        device = f"D-{customer['customer_id'][2:]}-N{customer['new_devices']}"
    elif roll < 0.09 and len(customer["devices"]) > 1:
        device = customer["devices"][1]
    else:
        device = customer["devices"][0]
    merchant = str(rng.choice(HIGH_RISK_MERCHANTS)) if rng.random() < 0.10 else str(rng.choice(customer["merchants"]))
    amount = float(np.exp(rng.normal(customer["log_amount"], 0.5)))
    if rng.random() < 0.04:
        amount *= rng.uniform(3, 10)
    return {
        "customer_id": customer["customer_id"],
        "timestamp": when,
        "amount": round(amount, 2),
        "merchant_category": merchant,
        "country": customer["home_country"] if rng.random() < 0.98 else str(rng.choice(COUNTRIES)),
        "device_id": device,
        "payment_method": str(rng.choice(PAYMENT_METHODS, p=[0.45, 0.30, 0.20, 0.05])),
        "is_fraud": 0,
    }


def _fraud_rows(rng, customers, n, start, days) -> list[dict]:
    """Fraud arrives in short bursts on a victim's account with a mix of risk signals.

    Each signal is applied with a probability, and some bursts are "quiet" (few or no signals),
    so not every fraud is detectable.
    """
    rows: list[dict] = []
    burst = 0
    while len(rows) < n:
        burst += 1
        customer = customers[int(rng.integers(len(customers)))]
        quiet = rng.random() < 0.30
        signal = 0.25 if quiet else 1.0
        new_device = rng.random() < 0.65 * signal
        device = f"D-F{burst:04d}" if new_device else customer["devices"][0]
        foreign = rng.random() < 0.45 * signal
        others = [c for c in COUNTRIES if c != customer["home_country"]]
        country = str(rng.choice(others)) if foreign else customer["home_country"]
        hour = rng.uniform(0, 5) if rng.random() < 0.40 * signal else rng.uniform(8, 23)
        when = start + pd.Timedelta(days=int(rng.integers(3, days)), hours=float(hour))
        high_amount = rng.random() < 0.55 * signal
        size = 1 if quiet else int(rng.choice([1, 2, 3, 4, 5], p=[0.35, 0.25, 0.2, 0.12, 0.08]))
        for _ in range(min(size, n - len(rows))):
            multiplier = rng.uniform(2.5, 12) if high_amount else rng.uniform(0.6, 1.8)
            merchant = str(rng.choice(HIGH_RISK_MERCHANTS)) if rng.random() < 0.6 * signal else str(rng.choice(EVERYDAY_MERCHANTS))
            rows.append(
                {
                    "customer_id": customer["customer_id"],
                    "timestamp": when.floor("s"),
                    "amount": round(float(np.exp(customer["log_amount"]) * multiplier), 2),
                    "merchant_category": merchant,
                    "country": country,
                    "device_id": device,
                    "payment_method": str(rng.choice(PAYMENT_METHODS, p=[0.05, 0.70, 0.20, 0.05])),
                    "is_fraud": 1,
                }
            )
            when += pd.Timedelta(minutes=float(rng.exponential(8)) + 1)
    return rows


def load_transactions(path: Path) -> pd.DataFrame:
    """Read every column as text; typing and checks happen in validation."""
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found. Run: python scripts/generate_demo_data.py")
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def dataset_version(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()[:16]
