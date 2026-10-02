"""Typed view of config.toml. The file is the single source of tunable values."""

import tomllib
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CONFIG_PATH = Path("config.toml")


@dataclass(frozen=True)
class DataConfig:
    n_transactions: int
    n_customers: int
    fraud_rate: float
    days: int
    start: str


@dataclass(frozen=True)
class SplitConfig:
    train_fraction: float
    validation_fraction: float


@dataclass(frozen=True)
class CostConfig:
    false_positive: float
    false_negative: float


@dataclass(frozen=True)
class ThresholdConfig:
    grid_start: float
    grid_stop: float
    grid_step: float
    block_min_precision: float
    block_min_support: int


@dataclass(frozen=True)
class Config:
    seed: int
    data: DataConfig
    split: SplitConfig
    costs: CostConfig
    thresholds: ThresholdConfig
    model_versions: dict[str, str]
    min_relative_cost_improvement: float
    policy_version: str


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> Config:
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    return Config(
        seed=int(raw["seed"]),
        data=DataConfig(**raw["data"]),
        split=SplitConfig(**raw["split"]),
        costs=CostConfig(**raw["costs"]),
        thresholds=ThresholdConfig(**raw["thresholds"]),
        model_versions={name: section["version"] for name, section in raw["models"].items()},
        min_relative_cost_improvement=float(raw["promotion"]["min_relative_cost_improvement"]),
        policy_version=raw["policy"]["version"],
    )
