"""Schema and row validation with quarantined (rejected) rows.

A missing column is a contract failure and raises. Row-level problems never raise: the row is
moved to ``rejected`` with a single ``reject_reason`` (first failing check wins), so every input
row is accounted for in exactly one of ``valid`` or ``rejected``.
"""

from dataclasses import dataclass

import pandas as pd

from fraud_risk.data import COLUMNS

TEXT_COLUMNS = ("customer_id", "merchant_category", "country", "device_id", "payment_method")
TIMEZONE_SUFFIX = r"(?:Z|[+-]\d{2}:\d{2})$"


class SchemaError(ValueError):
    """The input does not have the required columns."""


@dataclass(frozen=True)
class ValidationResult:
    valid: pd.DataFrame
    rejected: pd.DataFrame

    @property
    def reject_counts(self) -> dict[str, int]:
        return {str(k): int(v) for k, v in self.rejected["reject_reason"].value_counts().sort_index().items()}


def validate_transactions(raw: pd.DataFrame) -> ValidationResult:
    missing = [c for c in COLUMNS if c not in raw.columns]
    if missing:
        raise SchemaError(f"missing required columns: {', '.join(missing)}")

    text = raw.loc[:, list(COLUMNS)].fillna("").astype(str).apply(lambda col: col.str.strip())
    timestamps = pd.to_datetime(text["timestamp"], utc=True, errors="coerce", format="ISO8601")
    amounts = pd.to_numeric(text["amount"], errors="coerce")

    checks: list[tuple[str, pd.Series]] = [("missing_transaction_id", text["transaction_id"] == "")]
    checks += [(f"missing_{col}", text[col] == "") for col in TEXT_COLUMNS]
    checks += [
        ("invalid_timestamp", timestamps.isna()),
        ("timestamp_without_timezone", ~text["timestamp"].str.contains(TIMEZONE_SUFFIX, regex=True)),
        ("invalid_amount", amounts.isna() | ~amounts.abs().lt(float("inf"))),
        ("non_positive_amount", amounts.le(0)),
        ("invalid_label", ~text["is_fraud"].isin(["0", "1"])),
        ("duplicate_transaction_id", text["transaction_id"].ne("") & text["transaction_id"].duplicated(keep="first")),
    ]

    reason = pd.Series(pd.NA, index=raw.index, dtype="object")
    for name, failed in checks:
        reason = reason.mask(reason.isna() & failed.fillna(True), name)

    is_rejected = reason.notna()
    rejected = raw.loc[is_rejected, list(COLUMNS)].assign(reject_reason=reason[is_rejected])

    valid = text.loc[~is_rejected].copy()
    valid["timestamp"] = timestamps[~is_rejected]
    valid["amount"] = amounts[~is_rejected].astype(float)
    valid["is_fraud"] = valid["is_fraud"].astype(int)
    valid = valid.sort_values(["timestamp", "transaction_id"], kind="mergesort").reset_index(drop=True)
    return ValidationResult(valid=valid, rejected=rejected.reset_index(drop=True))
