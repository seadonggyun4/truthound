"""Data masking utilities for anonymizing sensitive data."""

import hashlib
import random
import string

import polars as pl

from truthound.scanners import scan_pii


def mask_data(
    lf: pl.LazyFrame,
    columns: list[str] | None = None,
    strategy: str = "redact",
) -> pl.DataFrame:
    """Mask sensitive data in a LazyFrame.

    Args:
        lf: Polars LazyFrame to mask.
        columns: Optional list of columns to mask. If None, auto-detects PII.
        strategy: Masking strategy - "redact", "hash", or "fake".

    Returns:
        Polars DataFrame with masked values.

    Raises:
        ValueError: If an invalid strategy is provided.
    """
    if strategy not in ("redact", "hash", "fake"):
        raise ValueError(f"Invalid strategy: {strategy}. Use 'redact', 'hash', or 'fake'.")

    df = lf.collect()

    # Auto-detect PII columns if not specified
    if columns is None:
        pii_findings = scan_pii(lf)
        columns = [f["column"] for f in pii_findings]

    if not columns:
        return df

    # Apply masking to each column
    for col in columns:
        if col not in df.columns:
            continue

        if strategy == "redact":
            df = _apply_redact(df, col)
        elif strategy == "hash":
            df = _apply_hash(df, col)
        elif strategy == "fake":
            df = _apply_fake(df, col)

    return df


def _apply_redact(df: pl.DataFrame, col: str) -> pl.DataFrame:
    """Apply redaction masking - replaces characters with asterisks."""

    def redact_value(val: str | None) -> str | None:
        if val is None:
            return None
        # Keep structure hints for common formats
        if "@" in val:  # Email
            parts = val.split("@")
            if len(parts) == 2:
                local = "*" * len(parts[0])
                domain_parts = parts[1].split(".")
                domain = ".".join("*" * len(p) for p in domain_parts)
                return f"{local}@{domain}"
        if "-" in val and len(val.replace("-", "").replace(" ", "")) in (9, 10, 16):  # SSN or phone or CC
            return "".join("*" if c.isalnum() else c for c in val)
        # Default: replace all characters
        return "*" * len(val)

    return df.with_columns(pl.col(col).map_elements(redact_value, return_dtype=pl.String).alias(col))


def _apply_hash(df: pl.DataFrame, col: str) -> pl.DataFrame:
    """Apply hash masking - replaces values with SHA256 hash."""

    def hash_value(val: str | None) -> str | None:
        if val is None:
            return None
        return hashlib.sha256(val.encode()).hexdigest()[:16]

    return df.with_columns(pl.col(col).map_elements(hash_value, return_dtype=pl.String).alias(col))


def _apply_fake(df: pl.DataFrame, col: str) -> pl.DataFrame:
    """Apply fake data masking - replaces with realistic-looking fake data."""
    col_lower = col.lower()

    def generate_fake(val: str | None) -> str | None:
        if val is None:
            return None

        # Generate appropriate fake data based on column name or value pattern
        if "email" in col_lower or ("@" in val and "." in val):
            random_str = "".join(random.choices(string.ascii_lowercase, k=8))
            return f"user{random_str}@masked.com"

        if "phone" in col_lower or (val.replace("-", "").replace(" ", "").replace("+", "").isdigit() and len(val) >= 7):
            return f"+1-555-{random.randint(100, 999)}-{random.randint(1000, 9999)}"

        if "ssn" in col_lower or (len(val) == 11 and val.count("-") == 2):
            return f"{random.randint(100, 999)}-{random.randint(10, 99)}-{random.randint(1000, 9999)}"

        # Default: generate random string of same length
        return "".join(random.choices(string.ascii_letters + string.digits, k=len(val)))

    return df.with_columns(pl.col(col).map_elements(generate_fake, return_dtype=pl.String).alias(col))
