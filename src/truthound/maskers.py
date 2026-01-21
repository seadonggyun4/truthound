"""Data masking utilities for anonymizing sensitive data."""

import logging
import warnings
from typing import Literal

import polars as pl

from truthound.scanners import scan_pii

logger = logging.getLogger(__name__)

# Type alias for masking strategies
MaskingStrategy = Literal["redact", "hash", "fake"]


class MaskingWarning(UserWarning):
    """Warning raised for masking-related issues."""

    pass


def mask_data(
    lf: pl.LazyFrame,
    columns: list[str] | None = None,
    strategy: MaskingStrategy = "redact",
    *,
    strict: bool = False,
) -> pl.DataFrame:
    """Mask sensitive data in a LazyFrame.

    Args:
        lf: Polars LazyFrame to mask.
        columns: Optional list of columns to mask. If None, auto-detects PII.
        strategy: Masking strategy - "redact", "hash", or "fake".
        strict: If True, raise ValueError for non-existent columns.
                If False (default), emit a warning and skip.

    Returns:
        Polars DataFrame with masked values.

    Raises:
        ValueError: If an invalid strategy is provided, or if strict=True
                    and a column is not found.

    Warnings:
        MaskingWarning: When a specified column does not exist in the data
                        (only if strict=False).

    Example:
        >>> import polars as pl
        >>> from truthound import mask
        >>> df = pl.DataFrame({"email": ["test@example.com"], "name": ["John"]})
        >>> masked = mask(df, columns=["email", "nonexistent"])
        # Warning: Column 'nonexistent' not found in data. Skipping.
    """
    valid_strategies = ("redact", "hash", "fake")
    if strategy not in valid_strategies:
        raise ValueError(
            f"Unknown masking strategy: '{strategy}'\n\n"
            f"Available strategies:\n"
            f"  • redact  - Replace values with '***' (default)\n"
            f"  • hash    - Replace values with deterministic hash\n"
            f"  • fake    - Replace values with realistic fake data\n\n"
            f"Example: truthound mask data.csv --strategy hash"
        )

    # Use streaming for large datasets (>1M rows)
    df = lf.collect(engine="streaming")

    # Auto-detect PII columns if not specified
    if columns is None:
        pii_findings = scan_pii(lf)
        columns = [f["column"] for f in pii_findings]

    if not columns:
        return df

    # Validate columns and collect warnings
    available_columns = set(df.columns)
    missing_columns: list[str] = []
    valid_columns: list[str] = []

    for col in columns:
        if col not in available_columns:
            missing_columns.append(col)
        else:
            valid_columns.append(col)

    # Handle missing columns
    if missing_columns:
        if strict:
            raise ValueError(
                f"Column(s) not found in data: {missing_columns}. "
                f"Available columns: {sorted(available_columns)}"
            )
        else:
            for col in missing_columns:
                warning_msg = (
                    f"Column '{col}' not found in data. Skipping. "
                    f"Available columns: {sorted(available_columns)}"
                )
                warnings.warn(warning_msg, MaskingWarning, stacklevel=2)
                logger.warning(warning_msg)

    # Apply masking only to valid columns
    for col in valid_columns:
        if strategy == "redact":
            df = _apply_redact(df, col)
        elif strategy == "hash":
            df = _apply_hash(df, col)
        elif strategy == "fake":
            df = _apply_fake(df, col)

    return df


def _apply_redact(df: pl.DataFrame, col: str) -> pl.DataFrame:
    """Apply redaction masking - replaces characters with asterisks.

    Uses native Polars expressions for better performance (no Python callbacks).
    """
    c = pl.col(col)

    # SSN/Phone/CC pattern: contains '-' and alphanumeric length is 9, 10, or 16
    # Replace alphanumeric characters with '*', keep separators
    stripped_len = c.str.replace_all(r"[-\s]", "").str.len_chars()
    is_structured = c.str.contains("-") & stripped_len.is_in([9, 10, 16])
    masked_structured = c.str.replace_all(r"[a-zA-Z0-9]", "*")

    # Email pattern: local@domain.tld -> ****@****.**
    is_email = c.str.contains("@")
    # For email masking, we use replace_all to avoid list index issues
    # Replace all characters in local part (before @) with *
    # Replace all non-dot characters in domain with *
    masked_email = pl.concat_str([
        c.str.extract(r"^([^@]+)@", 1).str.replace_all(r".", "*"),
        pl.lit("@"),
        c.str.extract(r"@(.+)$", 1).str.replace_all(r"[^.]", "*")
    ])

    # Default: replace all characters with '*'
    masked_default = c.str.replace_all(r".", "*")

    return df.with_columns(
        pl.when(c.is_null())
        .then(pl.lit(None))
        .when(is_structured)
        .then(masked_structured)
        .when(is_email)
        .then(masked_email)
        .otherwise(masked_default)
        .alias(col)
    )


def _apply_hash(df: pl.DataFrame, col: str) -> pl.DataFrame:
    """Apply hash masking - replaces values with SHA256 hash.

    Uses native Polars hash function for better performance (no Python callbacks).
    Note: Uses xxhash3 (Polars native) instead of SHA256 for performance.
    The hash is converted to hex string and truncated to 16 characters.
    """
    c = pl.col(col)

    # Use Polars native hash function (xxhash3) - much faster than Python SHA256
    # Convert to hex string representation
    hashed = c.hash().cast(pl.String).str.slice(0, 16)

    return df.with_columns(
        pl.when(c.is_null())
        .then(pl.lit(None))
        .otherwise(hashed)
        .alias(col)
    )


def _apply_fake(df: pl.DataFrame, col: str) -> pl.DataFrame:
    """Apply fake data masking - replaces with realistic-looking fake data.

    Uses native Polars expressions for better performance (no Python callbacks).
    Generates deterministic fake data based on hash of original value.
    """
    col_lower = col.lower()
    c = pl.col(col)

    # Use hash to generate deterministic "random" values
    # This ensures same input always produces same fake output (reproducible)
    hash_val = c.hash()
    hash_str = hash_val.cast(pl.String)

    # Email pattern detection (check first - most specific with @)
    is_email_col = "email" in col_lower
    is_email_val = c.str.contains("@") & c.str.contains(r"\.")
    is_email = pl.lit(is_email_col) | is_email_val

    # Generate fake email: user{hash8}@masked.com
    fake_email = pl.concat_str([
        pl.lit("user"),
        hash_str.str.slice(0, 8),
        pl.lit("@masked.com")
    ])

    # SSN pattern detection (check before phone - more specific: exactly 11 chars with 2 dashes)
    is_ssn_col = "ssn" in col_lower
    is_ssn_val = (c.str.len_chars() == 11) & (c.str.count_matches("-") == 2)
    is_ssn = pl.lit(is_ssn_col) | is_ssn_val

    # Generate fake SSN: XXX-XX-XXXX using hash digits
    fake_ssn = pl.concat_str([
        hash_str.str.slice(0, 3),
        pl.lit("-"),
        hash_str.str.slice(3, 2),
        pl.lit("-"),
        hash_str.str.slice(5, 4)
    ])

    # Phone pattern detection (after SSN to avoid false positives)
    is_phone_col = "phone" in col_lower
    stripped_phone = c.str.replace_all(r"[-\s+]", "")
    is_phone_val = stripped_phone.str.contains(r"^\d+$") & (stripped_phone.str.len_chars() >= 7)
    is_phone = pl.lit(is_phone_col) | is_phone_val

    # Generate fake phone: +1-555-XXX-XXXX using hash digits
    fake_phone = pl.concat_str([
        pl.lit("+1-555-"),
        hash_str.str.slice(0, 3),
        pl.lit("-"),
        hash_str.str.slice(3, 4)
    ])

    # Default: generate alphanumeric string of same length using hash
    # Pad hash to ensure enough characters, then slice to original length
    padded_hash = pl.concat_str([hash_str, hash_str, hash_str])  # Ensure enough length
    fake_default = padded_hash.str.slice(0, c.str.len_chars())

    return df.with_columns(
        pl.when(c.is_null())
        .then(pl.lit(None))
        .when(is_email)
        .then(fake_email)
        .when(is_ssn)
        .then(fake_ssn)
        .when(is_phone)
        .then(fake_phone)
        .otherwise(fake_default)
        .alias(col)
    )
