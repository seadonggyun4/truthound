"""Tests for version constraints."""

import pytest

from truthound.plugins.versioning.constraints import (
    VersionConstraint,
    parse_constraint,
)


class TestVersionConstraint:
    """Tests for VersionConstraint dataclass."""

    def test_create_constraint_defaults(self):
        """Test creating constraint with defaults."""
        constraint = VersionConstraint()

        assert constraint.min_version is None
        assert constraint.max_version is None
        assert constraint.excluded_versions == ()
        assert constraint.prerelease_ok is False

    def test_create_constraint_custom(self):
        """Test creating constraint with custom values."""
        constraint = VersionConstraint(
            min_version="1.0.0",
            max_version="2.0.0",
            excluded_versions=("1.5.0",),
            prerelease_ok=True,
        )

        assert constraint.min_version == "1.0.0"
        assert constraint.max_version == "2.0.0"
        assert constraint.excluded_versions == ("1.5.0",)
        assert constraint.prerelease_ok is True

    def test_any_version(self):
        """Test any_version factory."""
        constraint = VersionConstraint.any_version()

        assert constraint.is_satisfied_by("1.0.0") is True
        assert constraint.is_satisfied_by("99.99.99") is True
        assert constraint.prerelease_ok is True

    def test_exact_version(self):
        """Test exact version factory."""
        constraint = VersionConstraint.exact("1.5.0")

        assert constraint.is_satisfied_by("1.5.0") is True
        assert constraint.is_satisfied_by("1.5.1") is False
        assert constraint.is_satisfied_by("1.4.9") is False

    def test_at_least(self):
        """Test at_least factory."""
        constraint = VersionConstraint.at_least("2.0.0")

        assert constraint.is_satisfied_by("1.9.9") is False
        assert constraint.is_satisfied_by("2.0.0") is True
        assert constraint.is_satisfied_by("3.0.0") is True

    def test_compatible_with(self):
        """Test compatible_with factory (caret semver)."""
        constraint = VersionConstraint.compatible_with("1.2.3")

        assert constraint.is_satisfied_by("1.2.3") is True
        assert constraint.is_satisfied_by("1.9.0") is True
        assert constraint.is_satisfied_by("1.2.0") is False  # Below min
        assert constraint.is_satisfied_by("2.0.0") is False  # Different major


class TestVersionConstraintSatisfaction:
    """Tests for version satisfaction checking."""

    def test_min_version_check(self):
        """Test minimum version checking."""
        constraint = VersionConstraint(min_version="1.5.0")

        assert constraint.is_satisfied_by("1.4.9") is False
        assert constraint.is_satisfied_by("1.5.0") is True
        assert constraint.is_satisfied_by("1.5.1") is True
        assert constraint.is_satisfied_by("2.0.0") is True

    def test_max_version_check(self):
        """Test maximum version checking."""
        constraint = VersionConstraint(max_version="2.0.0")

        assert constraint.is_satisfied_by("1.9.9") is True
        assert constraint.is_satisfied_by("2.0.0") is True  # Inclusive
        assert constraint.is_satisfied_by("2.0.1") is False

    def test_max_version_exclusive(self):
        """Test exclusive maximum version (with asterisk)."""
        constraint = VersionConstraint(max_version="2.0.0*")

        assert constraint.is_satisfied_by("1.9.9") is True
        assert constraint.is_satisfied_by("2.0.0") is False  # Exclusive

    def test_excluded_versions(self):
        """Test excluded versions."""
        constraint = VersionConstraint(
            min_version="1.0.0",
            max_version="2.0.0",
            excluded_versions=("1.5.0", "1.6.0"),
        )

        assert constraint.is_satisfied_by("1.4.0") is True
        assert constraint.is_satisfied_by("1.5.0") is False
        assert constraint.is_satisfied_by("1.6.0") is False
        assert constraint.is_satisfied_by("1.7.0") is True

    def test_range_constraint(self):
        """Test range constraint."""
        constraint = VersionConstraint(
            min_version="1.2.0",
            max_version="1.5.0*",
        )

        assert constraint.is_satisfied_by("1.1.9") is False
        assert constraint.is_satisfied_by("1.2.0") is True
        assert constraint.is_satisfied_by("1.4.9") is True
        assert constraint.is_satisfied_by("1.5.0") is False


class TestVersionConstraintSerialization:
    """Tests for constraint serialization."""

    def test_to_dict(self):
        """Test converting to dictionary."""
        constraint = VersionConstraint(
            min_version="1.0.0",
            max_version="2.0.0",
            excluded_versions=("1.5.0",),
            prerelease_ok=True,
        )

        data = constraint.to_dict()

        assert data["min_version"] == "1.0.0"
        assert data["max_version"] == "2.0.0"
        assert data["excluded_versions"] == ["1.5.0"]
        assert data["prerelease_ok"] is True

    def test_from_dict(self):
        """Test creating from dictionary."""
        data = {
            "min_version": "1.0.0",
            "max_version": "2.0.0",
            "excluded_versions": ["1.5.0"],
            "prerelease_ok": True,
        }

        constraint = VersionConstraint.from_dict(data)

        assert constraint.min_version == "1.0.0"
        assert constraint.max_version == "2.0.0"
        assert constraint.excluded_versions == ("1.5.0",)
        assert constraint.prerelease_ok is True

    def test_str_representation(self):
        """Test string representation."""
        constraint = VersionConstraint(
            min_version="1.0.0",
            max_version="2.0.0",
        )

        assert ">=1.0.0" in str(constraint)
        assert "<2.0.0" in str(constraint)


class TestParseConstraint:
    """Tests for parse_constraint function."""

    def test_parse_any(self):
        """Test parsing any version."""
        constraint = parse_constraint("*")

        assert constraint.min_version is None
        assert constraint.max_version is None

    def test_parse_empty_as_any(self):
        """Test parsing empty string as any."""
        constraint = parse_constraint("")

        assert constraint.is_satisfied_by("1.0.0") is True

    def test_parse_exact(self):
        """Test parsing exact version."""
        constraint = parse_constraint("1.2.3")

        assert constraint.is_satisfied_by("1.2.3") is True
        assert constraint.is_satisfied_by("1.2.4") is False

    def test_parse_gte(self):
        """Test parsing >= constraint."""
        constraint = parse_constraint(">=1.2.3")

        assert constraint.is_satisfied_by("1.2.2") is False
        assert constraint.is_satisfied_by("1.2.3") is True
        assert constraint.is_satisfied_by("2.0.0") is True

    def test_parse_gt(self):
        """Test parsing > constraint."""
        constraint = parse_constraint(">1.2.3")

        assert constraint.is_satisfied_by("1.2.3") is False
        assert constraint.is_satisfied_by("1.2.4") is True

    def test_parse_lte(self):
        """Test parsing <= constraint."""
        constraint = parse_constraint("<=1.2.3")

        assert constraint.is_satisfied_by("1.2.3") is True
        assert constraint.is_satisfied_by("1.2.4") is False

    def test_parse_lt(self):
        """Test parsing < constraint."""
        constraint = parse_constraint("<1.2.3")

        assert constraint.is_satisfied_by("1.2.2") is True
        assert constraint.is_satisfied_by("1.2.3") is False

    def test_parse_caret(self):
        """Test parsing caret constraint."""
        constraint = parse_constraint("^1.2.3")

        assert constraint.is_satisfied_by("1.2.3") is True
        assert constraint.is_satisfied_by("1.9.0") is True
        assert constraint.is_satisfied_by("2.0.0") is False

    def test_parse_tilde(self):
        """Test parsing tilde constraint."""
        constraint = parse_constraint("~1.2.3")

        assert constraint.is_satisfied_by("1.2.3") is True
        assert constraint.is_satisfied_by("1.2.9") is True
        assert constraint.is_satisfied_by("1.3.0") is False

    def test_parse_range(self):
        """Test parsing comma-separated range."""
        constraint = parse_constraint(">=1.0.0,<2.0.0")

        assert constraint.is_satisfied_by("0.9.9") is False
        assert constraint.is_satisfied_by("1.0.0") is True
        assert constraint.is_satisfied_by("1.9.9") is True
        assert constraint.is_satisfied_by("2.0.0") is False

    def test_parse_range_with_exclusion(self):
        """Test parsing range with exclusion."""
        constraint = parse_constraint(">=1.0.0,<2.0.0,!=1.5.0")

        assert constraint.is_satisfied_by("1.4.0") is True
        assert constraint.is_satisfied_by("1.5.0") is False
        assert constraint.is_satisfied_by("1.6.0") is True

    def test_parse_invalid_raises(self):
        """Test parsing invalid constraint raises error."""
        with pytest.raises(ValueError, match="Cannot parse"):
            parse_constraint("invalid version spec")
