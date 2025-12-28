"""License management for commercial validators.

This module provides license tracking and enforcement:
- License types (open source, commercial, enterprise)
- License validation and expiry checking
- Usage tracking and reporting
- License key verification

Example:
    from truthound.validators.sdk.enterprise.licensing import (
        LicenseManager,
        LicenseInfo,
        LicenseType,
    )

    # Check validator license
    manager = LicenseManager()
    license_info = manager.get_license(validator_class)

    # Validate license
    if manager.validate_license(validator_class):
        result = validator.validate(data)
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


class LicenseType(Enum):
    """Types of licenses."""

    # Open source licenses
    MIT = auto()
    APACHE_2 = auto()
    BSD_3 = auto()
    GPL_3 = auto()
    LGPL_3 = auto()

    # Commercial licenses
    COMMERCIAL = auto()
    ENTERPRISE = auto()
    TRIAL = auto()

    # Special
    PROPRIETARY = auto()
    CUSTOM = auto()


class LicenseExpiredError(Exception):
    """Raised when license has expired."""

    def __init__(
        self,
        message: str,
        validator_name: str = "",
        expired_at: datetime | None = None,
    ):
        self.validator_name = validator_name
        self.expired_at = expired_at
        super().__init__(message)


class LicenseViolationError(Exception):
    """Raised when license terms are violated."""

    def __init__(
        self,
        message: str,
        validator_name: str = "",
        violation_type: str = "",
    ):
        self.validator_name = validator_name
        self.violation_type = violation_type
        super().__init__(message)


class LicenseNotFoundError(Exception):
    """Raised when license is not found."""
    pass


@dataclass
class LicenseInfo:
    """License information for a validator.

    Attributes:
        license_type: Type of license
        license_key: License key (for commercial)
        licensee: Name of the licensee
        issued_at: When license was issued
        expires_at: When license expires (None = never)
        max_users: Maximum concurrent users
        max_rows: Maximum rows that can be validated
        features: List of licensed features
        restrictions: List of restrictions
        validator_name: Name of the validator
        validator_version: Version of the validator
    """

    license_type: LicenseType
    license_key: str = ""
    licensee: str = ""
    issued_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime | None = None
    max_users: int = 0  # 0 = unlimited
    max_rows: int = 0   # 0 = unlimited
    features: tuple[str, ...] = field(default_factory=tuple)
    restrictions: tuple[str, ...] = field(default_factory=tuple)
    validator_name: str = ""
    validator_version: str = ""

    def is_expired(self) -> bool:
        """Check if license is expired."""
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at

    def is_open_source(self) -> bool:
        """Check if license is open source."""
        return self.license_type in {
            LicenseType.MIT,
            LicenseType.APACHE_2,
            LicenseType.BSD_3,
            LicenseType.GPL_3,
            LicenseType.LGPL_3,
        }

    def is_commercial(self) -> bool:
        """Check if license requires payment."""
        return self.license_type in {
            LicenseType.COMMERCIAL,
            LicenseType.ENTERPRISE,
            LicenseType.PROPRIETARY,
        }

    def days_until_expiry(self) -> int | None:
        """Get days until license expires."""
        if self.expires_at is None:
            return None
        delta = self.expires_at - datetime.now(timezone.utc)
        return max(0, delta.days)

    def has_feature(self, feature: str) -> bool:
        """Check if feature is licensed."""
        if not self.features:
            return True  # Empty means all features
        return feature in self.features

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "license_type": self.license_type.name,
            "license_key": self.license_key[:8] + "..." if self.license_key else "",
            "licensee": self.licensee,
            "issued_at": self.issued_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "max_users": self.max_users,
            "max_rows": self.max_rows,
            "features": list(self.features),
            "restrictions": list(self.restrictions),
            "validator_name": self.validator_name,
            "validator_version": self.validator_version,
            "is_expired": self.is_expired(),
            "days_until_expiry": self.days_until_expiry(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LicenseInfo":
        """Create from dictionary."""
        return cls(
            license_type=LicenseType[data["license_type"]],
            license_key=data.get("license_key", ""),
            licensee=data.get("licensee", ""),
            issued_at=datetime.fromisoformat(data["issued_at"])
                if data.get("issued_at") else datetime.now(timezone.utc),
            expires_at=datetime.fromisoformat(data["expires_at"])
                if data.get("expires_at") else None,
            max_users=data.get("max_users", 0),
            max_rows=data.get("max_rows", 0),
            features=tuple(data.get("features", [])),
            restrictions=tuple(data.get("restrictions", [])),
            validator_name=data.get("validator_name", ""),
            validator_version=data.get("validator_version", ""),
        )

    @classmethod
    def mit(cls, validator_name: str = "") -> "LicenseInfo":
        """Create MIT license."""
        return cls(
            license_type=LicenseType.MIT,
            validator_name=validator_name,
        )

    @classmethod
    def apache2(cls, validator_name: str = "") -> "LicenseInfo":
        """Create Apache 2.0 license."""
        return cls(
            license_type=LicenseType.APACHE_2,
            validator_name=validator_name,
        )

    @classmethod
    def trial(
        cls,
        validator_name: str = "",
        days: int = 30,
    ) -> "LicenseInfo":
        """Create trial license."""
        return cls(
            license_type=LicenseType.TRIAL,
            validator_name=validator_name,
            expires_at=datetime.now(timezone.utc) + timedelta(days=days),
            max_rows=100000,  # 100k row limit for trial
        )


class LicenseKeyGenerator:
    """Generates and validates license keys."""

    def __init__(self, secret_key: str):
        """Initialize generator.

        Args:
            secret_key: Secret key for signing
        """
        if not secret_key:
            raise ValueError("Secret key is required")
        self.secret_key = secret_key.encode()

    def generate(
        self,
        license_info: LicenseInfo,
    ) -> str:
        """Generate license key for license info.

        Args:
            license_info: License information

        Returns:
            License key string
        """
        # Create payload
        payload = {
            "type": license_info.license_type.name,
            "licensee": license_info.licensee,
            "issued": license_info.issued_at.isoformat(),
            "expires": license_info.expires_at.isoformat()
                if license_info.expires_at else None,
            "validator": license_info.validator_name,
            "version": license_info.validator_version,
            "features": list(license_info.features),
        }

        # Encode payload
        payload_json = json.dumps(payload, sort_keys=True)
        payload_b64 = base64.urlsafe_b64encode(payload_json.encode()).decode()

        # Create signature
        signature = hmac.new(
            self.secret_key,
            payload_json.encode(),
            hashlib.sha256,
        ).digest()
        signature_b64 = base64.urlsafe_b64encode(signature).decode()

        # Combine
        return f"{payload_b64}.{signature_b64}"

    def validate(self, license_key: str) -> LicenseInfo | None:
        """Validate license key and extract info.

        Args:
            license_key: License key to validate

        Returns:
            LicenseInfo if valid, None otherwise
        """
        try:
            # Split key
            parts = license_key.split(".")
            if len(parts) != 2:
                return None

            payload_b64, signature_b64 = parts

            # Decode payload
            payload_json = base64.urlsafe_b64decode(payload_b64).decode()
            payload = json.loads(payload_json)

            # Verify signature
            expected_signature = hmac.new(
                self.secret_key,
                payload_json.encode(),
                hashlib.sha256,
            ).digest()
            expected_b64 = base64.urlsafe_b64encode(expected_signature).decode()

            if not hmac.compare_digest(signature_b64, expected_b64):
                return None

            # Build license info
            return LicenseInfo(
                license_type=LicenseType[payload["type"]],
                license_key=license_key,
                licensee=payload.get("licensee", ""),
                issued_at=datetime.fromisoformat(payload["issued"]),
                expires_at=datetime.fromisoformat(payload["expires"])
                    if payload.get("expires") else None,
                validator_name=payload.get("validator", ""),
                validator_version=payload.get("version", ""),
                features=tuple(payload.get("features", [])),
            )
        except Exception as e:
            logger.debug(f"License key validation failed: {e}")
            return None


class LicenseValidator:
    """Validates licenses against policies."""

    def __init__(
        self,
        allow_expired: bool = False,
        allow_trial: bool = True,
        require_commercial: bool = False,
        required_features: list[str] | None = None,
    ):
        """Initialize validator.

        Args:
            allow_expired: Whether to allow expired licenses
            allow_trial: Whether to allow trial licenses
            require_commercial: Whether commercial license is required
            required_features: Features that must be licensed
        """
        self.allow_expired = allow_expired
        self.allow_trial = allow_trial
        self.require_commercial = require_commercial
        self.required_features = required_features or []

    def validate(
        self,
        license_info: LicenseInfo,
        raise_on_invalid: bool = True,
    ) -> bool:
        """Validate license against policy.

        Args:
            license_info: License to validate
            raise_on_invalid: Whether to raise exception

        Returns:
            True if valid

        Raises:
            LicenseExpiredError: If license is expired
            LicenseViolationError: If license violates policy
        """
        # Check expiry
        if not self.allow_expired and license_info.is_expired():
            if raise_on_invalid:
                raise LicenseExpiredError(
                    f"License for '{license_info.validator_name}' has expired",
                    validator_name=license_info.validator_name,
                    expired_at=license_info.expires_at,
                )
            return False

        # Check trial
        if not self.allow_trial and license_info.license_type == LicenseType.TRIAL:
            if raise_on_invalid:
                raise LicenseViolationError(
                    f"Trial license not allowed for '{license_info.validator_name}'",
                    validator_name=license_info.validator_name,
                    violation_type="trial_not_allowed",
                )
            return False

        # Check commercial requirement
        if self.require_commercial and not license_info.is_commercial():
            if raise_on_invalid:
                raise LicenseViolationError(
                    f"Commercial license required for '{license_info.validator_name}'",
                    validator_name=license_info.validator_name,
                    violation_type="commercial_required",
                )
            return False

        # Check features
        for feature in self.required_features:
            if not license_info.has_feature(feature):
                if raise_on_invalid:
                    raise LicenseViolationError(
                        f"Feature '{feature}' not licensed for "
                        f"'{license_info.validator_name}'",
                        validator_name=license_info.validator_name,
                        violation_type="feature_not_licensed",
                    )
                return False

        return True


@dataclass
class UsageRecord:
    """Record of license usage."""

    validator_name: str
    timestamp: datetime
    rows_processed: int
    user_id: str = ""
    session_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "validator_name": self.validator_name,
            "timestamp": self.timestamp.isoformat(),
            "rows_processed": self.rows_processed,
            "user_id": self.user_id,
            "session_id": self.session_id,
        }


class UsageTracker:
    """Tracks license usage for reporting."""

    def __init__(self, storage_path: Path | None = None):
        """Initialize tracker.

        Args:
            storage_path: Path to store usage data
        """
        self.storage_path = storage_path
        self._records: list[UsageRecord] = []
        self._usage_by_validator: dict[str, int] = {}

    def record_usage(
        self,
        validator_name: str,
        rows_processed: int,
        user_id: str = "",
        session_id: str = "",
    ) -> None:
        """Record usage event.

        Args:
            validator_name: Name of validator used
            rows_processed: Number of rows processed
            user_id: User identifier
            session_id: Session identifier
        """
        record = UsageRecord(
            validator_name=validator_name,
            timestamp=datetime.now(timezone.utc),
            rows_processed=rows_processed,
            user_id=user_id,
            session_id=session_id,
        )
        self._records.append(record)

        # Update totals
        if validator_name not in self._usage_by_validator:
            self._usage_by_validator[validator_name] = 0
        self._usage_by_validator[validator_name] += rows_processed

    def get_usage(self, validator_name: str) -> int:
        """Get total usage for a validator.

        Args:
            validator_name: Validator name

        Returns:
            Total rows processed
        """
        return self._usage_by_validator.get(validator_name, 0)

    def get_usage_report(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> dict[str, Any]:
        """Get usage report.

        Args:
            start_date: Start of reporting period
            end_date: End of reporting period

        Returns:
            Usage report dictionary
        """
        filtered_records = self._records

        if start_date:
            filtered_records = [
                r for r in filtered_records
                if r.timestamp >= start_date
            ]
        if end_date:
            filtered_records = [
                r for r in filtered_records
                if r.timestamp <= end_date
            ]

        # Aggregate
        by_validator: dict[str, dict[str, Any]] = {}
        for record in filtered_records:
            if record.validator_name not in by_validator:
                by_validator[record.validator_name] = {
                    "total_rows": 0,
                    "invocation_count": 0,
                    "unique_users": set(),
                }
            by_validator[record.validator_name]["total_rows"] += record.rows_processed
            by_validator[record.validator_name]["invocation_count"] += 1
            if record.user_id:
                by_validator[record.validator_name]["unique_users"].add(record.user_id)

        # Convert sets to counts
        for v in by_validator.values():
            v["unique_users"] = len(v["unique_users"])

        return {
            "period_start": start_date.isoformat() if start_date else None,
            "period_end": end_date.isoformat() if end_date else None,
            "total_records": len(filtered_records),
            "by_validator": by_validator,
        }

    def save(self) -> None:
        """Save usage data to storage."""
        if self.storage_path:
            data = {
                "records": [r.to_dict() for r in self._records],
                "totals": self._usage_by_validator,
            }
            with open(self.storage_path, "w") as f:
                json.dump(data, f, indent=2)


class LicenseManager:
    """Manages validator licenses.

    Provides high-level API for:
    - Checking validator licenses
    - Validating license compliance
    - Tracking usage
    """

    def __init__(
        self,
        secret_key: str = "",
        license_dir: Path | None = None,
        validator: LicenseValidator | None = None,
        usage_tracker: UsageTracker | None = None,
    ):
        """Initialize license manager.

        Args:
            secret_key: Secret key for license validation
            license_dir: Directory containing license files
            validator: Custom license validator
            usage_tracker: Usage tracker instance
        """
        self.secret_key = secret_key or os.environ.get("TRUTHOUND_LICENSE_KEY", "")
        self.license_dir = license_dir
        self._validator = validator or LicenseValidator()
        self._usage_tracker = usage_tracker or UsageTracker()
        self._key_generator = LicenseKeyGenerator(self.secret_key) if self.secret_key else None
        self._license_cache: dict[str, LicenseInfo] = {}

    def get_license(
        self,
        validator_class: type,
    ) -> LicenseInfo:
        """Get license information for a validator.

        Args:
            validator_class: Validator class

        Returns:
            LicenseInfo for the validator
        """
        validator_name = getattr(validator_class, "name", validator_class.__name__)

        # Check cache
        if validator_name in self._license_cache:
            return self._license_cache[validator_name]

        # Check class attributes
        license_type = getattr(validator_class, "license_type", None)
        license_key = getattr(validator_class, "license_key", "")

        if license_key and self._key_generator:
            # Validate embedded license key
            license_info = self._key_generator.validate(license_key)
            if license_info:
                self._license_cache[validator_name] = license_info
                return license_info

        if license_type:
            # Use declared license type
            if isinstance(license_type, str):
                license_type = LicenseType[license_type.upper()]
            license_info = LicenseInfo(
                license_type=license_type,
                validator_name=validator_name,
            )
            self._license_cache[validator_name] = license_info
            return license_info

        # Check license directory
        if self.license_dir:
            license_file = self.license_dir / f"{validator_name}.license"
            if license_file.exists():
                with open(license_file) as f:
                    data = json.load(f)
                license_info = LicenseInfo.from_dict(data)
                self._license_cache[validator_name] = license_info
                return license_info

        # Default to MIT
        return LicenseInfo.mit(validator_name)

    def validate_license(
        self,
        validator_class: type,
        raise_on_invalid: bool = True,
    ) -> bool:
        """Validate a validator's license.

        Args:
            validator_class: Validator class
            raise_on_invalid: Whether to raise on invalid license

        Returns:
            True if license is valid
        """
        license_info = self.get_license(validator_class)
        return self._validator.validate(license_info, raise_on_invalid)

    def track_usage(
        self,
        validator_class: type,
        rows_processed: int,
        user_id: str = "",
        session_id: str = "",
    ) -> None:
        """Track validator usage.

        Args:
            validator_class: Validator used
            rows_processed: Rows processed
            user_id: User identifier
            session_id: Session identifier
        """
        validator_name = getattr(validator_class, "name", validator_class.__name__)
        license_info = self.get_license(validator_class)

        # Check row limit
        if license_info.max_rows > 0:
            total_usage = self._usage_tracker.get_usage(validator_name)
            if total_usage + rows_processed > license_info.max_rows:
                raise LicenseViolationError(
                    f"Row limit exceeded for '{validator_name}': "
                    f"{total_usage + rows_processed} > {license_info.max_rows}",
                    validator_name=validator_name,
                    violation_type="row_limit_exceeded",
                )

        self._usage_tracker.record_usage(
            validator_name=validator_name,
            rows_processed=rows_processed,
            user_id=user_id,
            session_id=session_id,
        )

    def get_usage_report(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> dict[str, Any]:
        """Get usage report.

        Args:
            start_date: Start of period
            end_date: End of period

        Returns:
            Usage report
        """
        return self._usage_tracker.get_usage_report(start_date, end_date)

    def list_licenses(self) -> list[LicenseInfo]:
        """List all cached licenses."""
        return list(self._license_cache.values())
