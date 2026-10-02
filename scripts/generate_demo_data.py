"""Write the deterministic synthetic dataset and a small committed sample."""

import argparse
from pathlib import Path

from fraud_risk.config import DEFAULT_CONFIG_PATH, load_config
from fraud_risk.data import dataset_version, generate_transactions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--out", type=Path, default=Path("data/generated/transactions.csv"))
    parser.add_argument("--sample", type=Path, default=Path("data/sample/transactions_sample.csv"))
    args = parser.parse_args()

    config = load_config(args.config)
    df = generate_transactions(config.data, config.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, lineterminator="\n")
    args.sample.parent.mkdir(parents=True, exist_ok=True)
    df.head(200).to_csv(args.sample, index=False, lineterminator="\n")

    print(f"Wrote {len(df):,} synthetic transactions ({int(df['is_fraud'].sum())} labelled fraud) to {args.out}")
    print(f"Dataset version: {dataset_version(args.out)}")


if __name__ == "__main__":
    main()
