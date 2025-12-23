"""Coordinate validators.

Validators for checking latitude, longitude, and coordinate pairs.
"""

from typing import Any

import polars as pl

from truthound.types import Severity
from truthound.validators.base import ValidationIssue
from truthound.validators.geospatial.base import GeoValidator
from truthound.validators.registry import register_validator


@register_validator
class LatitudeValidator(GeoValidator):
    """Validates that latitude values are within valid range (-90 to 90).

    Example:
        validator = LatitudeValidator(column="latitude")

        # With custom range (e.g., only Northern Hemisphere)
        validator = LatitudeValidator(
            column="latitude",
            min_lat=0,
            max_lat=90,
        )
    """

    name = "latitude"
    category = "geospatial"

    def __init__(
        self,
        column: str,
        min_lat: float = -90.0,
        max_lat: float = 90.0,
        **kwargs: Any,
    ):
        """Initialize latitude validator.

        Args:
            column: Column containing latitude values
            min_lat: Minimum valid latitude (default -90)
            max_lat: Maximum valid latitude (default 90)
            **kwargs: Additional config
        """
        super().__init__(**kwargs)
        self.column = column
        self.min_lat = max(min_lat, self.LAT_MIN)
        self.max_lat = min(max_lat, self.LAT_MAX)

        if self.min_lat > self.max_lat:
            raise ValueError("'min_lat' cannot be greater than 'max_lat'")

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        result = lf.select([
            pl.len().alias("_total"),
            (
                (pl.col(self.column) < self.min_lat) |
                (pl.col(self.column) > self.max_lat)
            ).sum().alias("_invalid"),
            pl.col(self.column).is_null().sum().alias("_null"),
        ]).collect()

        total = result["_total"][0]
        invalid = result["_invalid"][0]
        null_count = result["_null"][0]

        if invalid > 0:
            if self._passes_mostly(invalid, total - null_count):
                return issues

            issues.append(
                ValidationIssue(
                    column=self.column,
                    issue_type="latitude_out_of_range",
                    count=invalid,
                    severity=Severity.HIGH,
                    details=f"{invalid} values outside valid latitude range [{self.min_lat}, {self.max_lat}]",
                    expected=f"Latitude between {self.min_lat} and {self.max_lat}",
                )
            )

        return issues


@register_validator
class LongitudeValidator(GeoValidator):
    """Validates that longitude values are within valid range (-180 to 180).

    Example:
        validator = LongitudeValidator(column="longitude")

        # With custom range (e.g., only Western Hemisphere)
        validator = LongitudeValidator(
            column="longitude",
            min_lon=-180,
            max_lon=0,
        )
    """

    name = "longitude"
    category = "geospatial"

    def __init__(
        self,
        column: str,
        min_lon: float = -180.0,
        max_lon: float = 180.0,
        **kwargs: Any,
    ):
        """Initialize longitude validator.

        Args:
            column: Column containing longitude values
            min_lon: Minimum valid longitude (default -180)
            max_lon: Maximum valid longitude (default 180)
            **kwargs: Additional config
        """
        super().__init__(**kwargs)
        self.column = column
        self.min_lon = max(min_lon, self.LON_MIN)
        self.max_lon = min(max_lon, self.LON_MAX)

        if self.min_lon > self.max_lon:
            raise ValueError("'min_lon' cannot be greater than 'max_lon'")

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        result = lf.select([
            pl.len().alias("_total"),
            (
                (pl.col(self.column) < self.min_lon) |
                (pl.col(self.column) > self.max_lon)
            ).sum().alias("_invalid"),
            pl.col(self.column).is_null().sum().alias("_null"),
        ]).collect()

        total = result["_total"][0]
        invalid = result["_invalid"][0]
        null_count = result["_null"][0]

        if invalid > 0:
            if self._passes_mostly(invalid, total - null_count):
                return issues

            issues.append(
                ValidationIssue(
                    column=self.column,
                    issue_type="longitude_out_of_range",
                    count=invalid,
                    severity=Severity.HIGH,
                    details=f"{invalid} values outside valid longitude range [{self.min_lon}, {self.max_lon}]",
                    expected=f"Longitude between {self.min_lon} and {self.max_lon}",
                )
            )

        return issues


@register_validator
class CoordinateValidator(GeoValidator):
    """Validates that coordinate pairs (lat, lon) are valid.

    Example:
        validator = CoordinateValidator(
            lat_column="latitude",
            lon_column="longitude",
        )

        # Validate coordinates are in specific region
        validator = CoordinateValidator(
            lat_column="lat",
            lon_column="lon",
            min_lat=33.0,
            max_lat=43.0,
            min_lon=124.0,
            max_lon=132.0,  # Korea region
        )
    """

    name = "coordinate"
    category = "geospatial"

    def __init__(
        self,
        lat_column: str,
        lon_column: str,
        min_lat: float = -90.0,
        max_lat: float = 90.0,
        min_lon: float = -180.0,
        max_lon: float = 180.0,
        **kwargs: Any,
    ):
        """Initialize coordinate validator.

        Args:
            lat_column: Column containing latitude values
            lon_column: Column containing longitude values
            min_lat: Minimum valid latitude
            max_lat: Maximum valid latitude
            min_lon: Minimum valid longitude
            max_lon: Maximum valid longitude
            **kwargs: Additional config
        """
        super().__init__(**kwargs)
        self.lat_column = lat_column
        self.lon_column = lon_column
        self.min_lat = max(min_lat, self.LAT_MIN)
        self.max_lat = min(max_lat, self.LAT_MAX)
        self.min_lon = max(min_lon, self.LON_MIN)
        self.max_lon = min(max_lon, self.LON_MAX)

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        # Check for invalid latitudes
        lat_invalid = (
            (pl.col(self.lat_column) < self.min_lat) |
            (pl.col(self.lat_column) > self.max_lat)
        )

        # Check for invalid longitudes
        lon_invalid = (
            (pl.col(self.lon_column) < self.min_lon) |
            (pl.col(self.lon_column) > self.max_lon)
        )

        result = lf.select([
            pl.len().alias("_total"),
            lat_invalid.sum().alias("_lat_invalid"),
            lon_invalid.sum().alias("_lon_invalid"),
            (lat_invalid | lon_invalid).sum().alias("_any_invalid"),
        ]).collect()

        total = result["_total"][0]
        lat_invalid_count = result["_lat_invalid"][0]
        lon_invalid_count = result["_lon_invalid"][0]
        any_invalid = result["_any_invalid"][0]

        if lat_invalid_count > 0:
            if not self._passes_mostly(lat_invalid_count, total):
                issues.append(
                    ValidationIssue(
                        column=self.lat_column,
                        issue_type="latitude_out_of_range",
                        count=lat_invalid_count,
                        severity=Severity.HIGH,
                        details=f"{lat_invalid_count} latitude values outside range [{self.min_lat}, {self.max_lat}]",
                        expected=f"Latitude in [{self.min_lat}, {self.max_lat}]",
                    )
                )

        if lon_invalid_count > 0:
            if not self._passes_mostly(lon_invalid_count, total):
                issues.append(
                    ValidationIssue(
                        column=self.lon_column,
                        issue_type="longitude_out_of_range",
                        count=lon_invalid_count,
                        severity=Severity.HIGH,
                        details=f"{lon_invalid_count} longitude values outside range [{self.min_lon}, {self.max_lon}]",
                        expected=f"Longitude in [{self.min_lon}, {self.max_lon}]",
                    )
                )

        return issues


@register_validator
class CoordinateNotNullIslandValidator(GeoValidator):
    """Validates that coordinates are not at (0, 0) - "Null Island".

    Null Island is a common data quality issue where missing coordinates
    default to (0, 0).

    Example:
        validator = CoordinateNotNullIslandValidator(
            lat_column="latitude",
            lon_column="longitude",
        )

        # With tolerance for near-zero coordinates
        validator = CoordinateNotNullIslandValidator(
            lat_column="lat",
            lon_column="lon",
            tolerance=0.001,  # ~100m tolerance
        )
    """

    name = "coordinate_not_null_island"
    category = "geospatial"

    def __init__(
        self,
        lat_column: str,
        lon_column: str,
        tolerance: float = 0.0,
        **kwargs: Any,
    ):
        """Initialize Null Island validator.

        Args:
            lat_column: Column containing latitude values
            lon_column: Column containing longitude values
            tolerance: Tolerance around (0, 0) to consider as Null Island
            **kwargs: Additional config
        """
        super().__init__(**kwargs)
        self.lat_column = lat_column
        self.lon_column = lon_column
        self.tolerance = tolerance

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        # Check for coordinates at or near (0, 0)
        is_null_island = (
            (pl.col(self.lat_column).abs() <= self.tolerance) &
            (pl.col(self.lon_column).abs() <= self.tolerance)
        )

        result = lf.select([
            pl.len().alias("_total"),
            is_null_island.sum().alias("_null_island"),
        ]).collect()

        total = result["_total"][0]
        null_island_count = result["_null_island"][0]

        if null_island_count > 0:
            if self._passes_mostly(null_island_count, total):
                return issues

            issues.append(
                ValidationIssue(
                    column=f"{self.lat_column}, {self.lon_column}",
                    issue_type="null_island_coordinates",
                    count=null_island_count,
                    severity=Severity.MEDIUM,
                    details=f"{null_island_count} coordinates at or near (0, 0) - possible missing data",
                    expected="Coordinates not at Null Island (0, 0)",
                )
            )

        return issues
