"""PII scanners for detecting personally identifiable information."""

import re
from dataclasses import dataclass

import polars as pl

from truthound.types import PIIType


@dataclass
class PIIPattern:
    """Pattern definition for PII detection."""

    pii_type: PIIType
    pattern: re.Pattern
    confidence_base: int  # Base confidence score (0-100)


# PII detection patterns
PII_PATTERNS: list[PIIPattern] = [
    PIIPattern(
        pii_type=PIIType.EMAIL,
        pattern=re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"),
        confidence_base=95,
    ),
    PIIPattern(
        pii_type=PIIType.PHONE,
        pattern=re.compile(r"^(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}$"),
        confidence_base=85,
    ),
    PIIPattern(
        pii_type=PIIType.SSN,
        pattern=re.compile(r"^\d{3}-\d{2}-\d{4}$"),
        confidence_base=98,
    ),
    PIIPattern(
        pii_type=PIIType.CREDIT_CARD,
        pattern=re.compile(r"^\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}$"),
        confidence_base=90,
    ),
    PIIPattern(
        pii_type=PIIType.IP_ADDRESS,
        pattern=re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}$"),
        confidence_base=95,
    ),
    PIIPattern(
        pii_type=PIIType.DATE_OF_BIRTH,
        pattern=re.compile(r"^\d{4}-\d{2}-\d{2}$"),
        confidence_base=60,  # Lower confidence as dates aren't always DOB
    ),
    # Korean specific patterns
    PIIPattern(
        pii_type=PIIType.KOREAN_RRN,
        # 주민등록번호: 6자리-7자리 (YYMMDD-GNNNNNN)
        pattern=re.compile(r"^\d{6}-[1-4]\d{6}$"),
        confidence_base=98,
    ),
    PIIPattern(
        pii_type=PIIType.KOREAN_PHONE,
        # 한국 휴대폰: 010-XXXX-XXXX, 지역번호 02-XXX-XXXX 등
        pattern=re.compile(r"^0\d{1,2}-\d{3,4}-\d{4}$"),
        confidence_base=90,
    ),
    PIIPattern(
        pii_type=PIIType.BANK_ACCOUNT,
        # 계좌번호: 다양한 형식 (XXX-XXX-XXXXXX, XXX-XXXX-XXXX-XX 등)
        pattern=re.compile(r"^\d{3,4}-\d{2,4}-\d{4,6}(?:-\d{1,2})?$"),
        confidence_base=80,
    ),
    PIIPattern(
        pii_type=PIIType.PASSPORT,
        # 여권번호: 알파벳 1-2자 + 숫자 7-8자
        pattern=re.compile(r"^[A-Z]{1,2}\d{7,8}$"),
        confidence_base=85,
    ),
]

# Column name hints that increase confidence
COLUMN_HINTS: dict[PIIType, list[str]] = {
    PIIType.EMAIL: ["email", "e-mail", "mail", "contact"],
    PIIType.PHONE: ["phone", "tel", "mobile", "cell", "fax", "contact"],
    PIIType.SSN: ["ssn", "social", "security", "sin", "tax_id", "taxid"],
    PIIType.CREDIT_CARD: ["card", "credit", "cc", "payment"],
    PIIType.IP_ADDRESS: ["ip", "ip_address", "ipaddress", "client_ip"],
    PIIType.DATE_OF_BIRTH: ["dob", "birth", "birthday", "born"],
    # Korean hints
    PIIType.KOREAN_RRN: ["주민", "rrn", "resident", "주민번호", "주민등록"],
    PIIType.KOREAN_PHONE: ["phone", "전화", "휴대폰", "연락처", "tel", "mobile"],
    PIIType.BANK_ACCOUNT: ["account", "계좌", "bank", "은행", "acct"],
    PIIType.PASSPORT: ["passport", "여권"],
}


def scan_pii(lf: pl.LazyFrame) -> list[dict]:
    """Scan a LazyFrame for PII.

    Args:
        lf: Polars LazyFrame to scan.

    Returns:
        List of PII findings with column, type, count, and confidence.
    """
    findings: list[dict] = []
    schema = lf.collect_schema()
    string_columns = [
        col for col in schema.names() if schema[col] in (pl.String, pl.Utf8)
    ]
    if not string_columns:
        return findings

    # Preserve full non-null denominators while collecting only scalar counts.
    counts = lf.select(
        [pl.col(col).count() for col in string_columns]
    ).collect(streaming=True).row(0, named=True)

    for col in string_columns:
        non_null_count = counts[col]
        if non_null_count == 0:
            continue

        # The existing scanner uses the first 1000 non-null values, not rows.
        # Project and bound in the lazy plan before materializing any values.
        sample = lf.select(
            pl.col(col).drop_nulls().head(1000)
        ).collect(streaming=True).get_column(col)
        finding = _score_pii_sample(col, non_null_count, sample.to_list())
        if finding is not None:
            findings.append(finding)

    # Sort by confidence descending
    findings.sort(key=lambda x: x["confidence"], reverse=True)

    return findings


def _score_pii_sample(col: str, non_null_count: int, sample: list[str]) -> dict | None:
    """Shared scoring for lazy and batch readers; preserve existing thresholds."""
    if not sample:
        return None
    sample_size = len(sample)
    col_lower = col.lower()

    # Check each PII pattern
    for pii_pattern in PII_PATTERNS:
        match_count = 0
        for val in sample:
            if isinstance(val, str) and pii_pattern.pattern.match(val):
                match_count += 1

        if match_count == 0:
            continue

        # Calculate confidence
        match_ratio = match_count / sample_size
        confidence = pii_pattern.confidence_base

        # Boost confidence if column name hints at PII type
        hints = COLUMN_HINTS.get(pii_pattern.pii_type, [])
        if any(hint in col_lower for hint in hints):
            confidence = min(99, confidence + 10)

        # Adjust based on match ratio
        if match_ratio > 0.8:
            confidence = min(99, confidence + 5)
        elif match_ratio < 0.3:
            confidence = max(50, confidence - 20)

        # Only report if confidence is reasonable and match ratio is significant
        if confidence >= 50 and match_ratio >= 0.1:
            # Extrapolate count to full dataset
            estimated_count = int(non_null_count * match_ratio)

            return {
                    "column": col,
                    "pii_type": pii_pattern.pii_type.value,
                    "count": estimated_count,
                    "confidence": confidence,
            }

    return None


def _scan_parquet_pii(path, schema: pl.Schema) -> tuple[int, list[dict]] | None:
    """Read local Parquet in bounded Arrow batches when Arrow is installed.

    The optional fast path retains the same first-1000-non-null sample and
    full non-null denominator. Reader errors propagate, never become a pass.
    """
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        if exc.name == "pyarrow":
            return None
        raise

    columns = [col for col in schema.names() if schema[col] in (pl.String, pl.Utf8)]
    samples = {col: [] for col in columns}
    counts = {col: 0 for col in columns}
    with pq.ParquetFile(path) as parquet:
        row_count = parquet.metadata.num_rows
        if columns:
            for batch in parquet.iter_batches(batch_size=1024, columns=columns, use_threads=False):
                for index, col in enumerate(columns):
                    values = batch.column(index)
                    counts[col] += len(values) - values.null_count
                    remaining = 1000 - len(samples[col])
                    if remaining:
                        samples[col].extend(values.drop_null().slice(0, remaining).to_pylist())
    findings = []
    for col in columns:
        finding = _score_pii_sample(col, counts[col], samples[col])
        if finding is not None:
            findings.append(finding)
    findings.sort(key=lambda item: item["confidence"], reverse=True)
    return row_count, findings
