"""Distance validators.

Validators for checking distances between geographic points.
"""

from math import radians, sin, cos, sqrt, atan2
from typing import Any

import numpy as np
import polars as pl

from truthound.types import Severity
from truthound.validators.base import ValidationIssue
from truthound.validators.geospatial.base import GeoValidator, EARTH_RADIUS_KM, EARTH_RADIUS_MILES
from truthound.validators.registry import register_validator
from truthound.validators.optimization import VectorizedGeoMixin, DistanceUnit


@register_validator
class GeoDistanceValidator(GeoValidator):
    """Validates that distance between two coordinate pairs is within bounds.

    Example:
        # Distance between pickup and dropoff should be <= 100km
        validator = GeoDistanceValidator(
            lat1_column="pickup_lat",
            lon1_column="pickup_lon",
            lat2_column="dropoff_lat",
            lon2_column="dropoff_lon",
            max_distance=100,
            unit="km",
        )

        # Distance should be at least 1km (not same location)
        validator = GeoDistanceValidator(
            lat1_column="start_lat",
            lon1_column="start_lon",
            lat2_column="end_lat",
            lon2_column="end_lon",
            min_distance=1,
            unit="km",
        )
    """

    name = "geo_distance"
    category = "geospatial"

    def __init__(
        self,
        lat1_column: str,
        lon1_column: str,
        lat2_column: str,
        lon2_column: str,
        min_distance: float | None = None,
        max_distance: float | None = None,
        unit: str = "km",
        **kwargs: Any,
    ):
        """Initialize distance validator.

        Args:
            lat1_column: Column for first point's latitude
            lon1_column: Column for first point's longitude
            lat2_column: Column for second point's latitude
            lon2_column: Column for second point's longitude
            min_distance: Minimum allowed distance
            max_distance: Maximum allowed distance
            unit: Distance unit ('km' or 'miles')
            **kwargs: Additional config
        """
        super().__init__(**kwargs)
        self.lat1_column = lat1_column
        self.lon1_column = lon1_column
        self.lat2_column = lat2_column
        self.lon2_column = lon2_column
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.unit = unit

        if unit not in ("km", "miles"):
            raise ValueError("'unit' must be 'km' or 'miles'")

        if min_distance is None and max_distance is None:
            raise ValueError("At least one of 'min_distance' or 'max_distance' required")

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        # Earth radius based on unit
        radius = EARTH_RADIUS_MILES if self.unit == "miles" else EARTH_RADIUS_KM

        # Haversine formula in Polars expressions
        # Using arcsin version: c = 2 * arcsin(sqrt(a))
        lat1_rad = pl.col(self.lat1_column).radians()
        lat2_rad = pl.col(self.lat2_column).radians()
        delta_lat = (pl.col(self.lat2_column) - pl.col(self.lat1_column)).radians()
        delta_lon = (pl.col(self.lon2_column) - pl.col(self.lon1_column)).radians()

        a = (
            (delta_lat / 2).sin().pow(2) +
            lat1_rad.cos() * lat2_rad.cos() * (delta_lon / 2).sin().pow(2)
        )
        # Use arcsin formula: c = 2 * arcsin(sqrt(a))
        c = 2 * a.sqrt().arcsin()
        distance = radius * c

        # Build validation expressions
        exprs = [
            pl.len().alias("_total"),
            distance.alias("_distance"),
        ]

        if self.min_distance is not None:
            exprs.append((distance < self.min_distance).sum().alias("_below_min"))

        if self.max_distance is not None:
            exprs.append((distance > self.max_distance).sum().alias("_above_max"))

        result = lf.select(exprs).collect()
        total = result["_total"][0]

        if self.min_distance is not None:
            below_min = result["_below_min"][0]
            if below_min > 0:
                if not self._passes_mostly(below_min, total):
                    issues.append(
                        ValidationIssue(
                            column=f"{self.lat1_column},{self.lon1_column} -> {self.lat2_column},{self.lon2_column}",
                            issue_type="distance_below_minimum",
                            count=below_min,
                            severity=Severity.MEDIUM,
                            details=f"{below_min} pairs with distance < {self.min_distance} {self.unit}",
                            expected=f"Distance >= {self.min_distance} {self.unit}",
                        )
                    )

        if self.max_distance is not None:
            above_max = result["_above_max"][0]
            if above_max > 0:
                if not self._passes_mostly(above_max, total):
                    issues.append(
                        ValidationIssue(
                            column=f"{self.lat1_column},{self.lon1_column} -> {self.lat2_column},{self.lon2_column}",
                            issue_type="distance_above_maximum",
                            count=above_max,
                            severity=Severity.MEDIUM,
                            details=f"{above_max} pairs with distance > {self.max_distance} {self.unit}",
                            expected=f"Distance <= {self.max_distance} {self.unit}",
                        )
                    )

        return issues


@register_validator
class GeoDistanceFromPointValidator(GeoValidator):
    """Validates that coordinates are within distance from a reference point.

    Example:
        # All locations should be within 50km of headquarters
        validator = GeoDistanceFromPointValidator(
            lat_column="store_lat",
            lon_column="store_lon",
            ref_lat=37.5665,
            ref_lon=126.9780,  # Seoul
            max_distance=50,
            unit="km",
        )
    """

    name = "geo_distance_from_point"
    category = "geospatial"

    def __init__(
        self,
        lat_column: str,
        lon_column: str,
        ref_lat: float,
        ref_lon: float,
        min_distance: float | None = None,
        max_distance: float | None = None,
        unit: str = "km",
        **kwargs: Any,
    ):
        """Initialize distance from point validator.

        Args:
            lat_column: Column containing latitude values
            lon_column: Column containing longitude values
            ref_lat: Reference point latitude
            ref_lon: Reference point longitude
            min_distance: Minimum distance from reference point
            max_distance: Maximum distance from reference point
            unit: Distance unit ('km' or 'miles')
            **kwargs: Additional config
        """
        super().__init__(**kwargs)
        self.lat_column = lat_column
        self.lon_column = lon_column
        self.ref_lat = ref_lat
        self.ref_lon = ref_lon
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.unit = unit

        if not self.is_valid_coordinate(ref_lat, ref_lon):
            raise ValueError(f"Invalid reference coordinates: ({ref_lat}, {ref_lon})")

        if unit not in ("km", "miles"):
            raise ValueError("'unit' must be 'km' or 'miles'")

        if min_distance is None and max_distance is None:
            raise ValueError("At least one of 'min_distance' or 'max_distance' required")

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        radius = EARTH_RADIUS_MILES if self.unit == "miles" else EARTH_RADIUS_KM

        # Haversine formula with fixed reference point
        # Using arcsin version: c = 2 * arcsin(sqrt(a))
        lat1_rad = radians(self.ref_lat)
        lat2_rad = pl.col(self.lat_column).radians()
        delta_lat = (pl.col(self.lat_column) - self.ref_lat).radians()
        delta_lon = (pl.col(self.lon_column) - self.ref_lon).radians()

        a = (
            (delta_lat / 2).sin().pow(2) +
            pl.lit(cos(lat1_rad)) * lat2_rad.cos() * (delta_lon / 2).sin().pow(2)
        )
        # Use arcsin formula: c = 2 * arcsin(sqrt(a))
        c = 2 * a.sqrt().arcsin()
        distance = radius * c

        exprs = [pl.len().alias("_total")]

        if self.min_distance is not None:
            exprs.append((distance < self.min_distance).sum().alias("_below_min"))

        if self.max_distance is not None:
            exprs.append((distance > self.max_distance).sum().alias("_above_max"))

        result = lf.select(exprs).collect()
        total = result["_total"][0]

        if self.min_distance is not None:
            below_min = result["_below_min"][0]
            if below_min > 0 and not self._passes_mostly(below_min, total):
                issues.append(
                    ValidationIssue(
                        column=f"{self.lat_column}, {self.lon_column}",
                        issue_type="distance_from_point_below_minimum",
                        count=below_min,
                        severity=Severity.MEDIUM,
                        details=f"{below_min} points closer than {self.min_distance} {self.unit} from ({self.ref_lat}, {self.ref_lon})",
                        expected=f"Distance >= {self.min_distance} {self.unit}",
                    )
                )

        if self.max_distance is not None:
            above_max = result["_above_max"][0]
            if above_max > 0 and not self._passes_mostly(above_max, total):
                issues.append(
                    ValidationIssue(
                        column=f"{self.lat_column}, {self.lon_column}",
                        issue_type="distance_from_point_above_maximum",
                        count=above_max,
                        severity=Severity.MEDIUM,
                        details=f"{above_max} points farther than {self.max_distance} {self.unit} from ({self.ref_lat}, {self.ref_lon})",
                        expected=f"Distance <= {self.max_distance} {self.unit}",
                    )
                )

        return issues


@register_validator
class OptimizedGeoDistanceValidator(GeoValidator, VectorizedGeoMixin):
    """Optimized geo distance validator using vectorized NumPy operations.

    Uses VectorizedGeoMixin for efficient batch distance calculations:
    - Vectorized Haversine using NumPy broadcasting
    - Batch distance matrix computation
    - Memory-efficient chunked processing for large datasets

    Example:
        validator = OptimizedGeoDistanceValidator(
            lat1_column="pickup_lat",
            lon1_column="pickup_lon",
            lat2_column="dropoff_lat",
            lon2_column="dropoff_lon",
            max_distance=100,
            unit="km",
        )
    """

    name = "optimized_geo_distance"
    category = "geospatial"

    def __init__(
        self,
        lat1_column: str,
        lon1_column: str,
        lat2_column: str,
        lon2_column: str,
        min_distance: float | None = None,
        max_distance: float | None = None,
        unit: str = "km",
        chunk_size: int = 100000,
        **kwargs: Any,
    ):
        """Initialize optimized distance validator.

        Args:
            lat1_column: Column for first point's latitude
            lon1_column: Column for first point's longitude
            lat2_column: Column for second point's latitude
            lon2_column: Column for second point's longitude
            min_distance: Minimum allowed distance
            max_distance: Maximum allowed distance
            unit: Distance unit ('km' or 'miles')
            chunk_size: Chunk size for batch processing
            **kwargs: Additional config
        """
        super().__init__(**kwargs)
        self.lat1_column = lat1_column
        self.lon1_column = lon1_column
        self.lat2_column = lat2_column
        self.lon2_column = lon2_column
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.unit = unit
        self._geo_chunk_size = chunk_size

        if unit not in ("km", "miles"):
            raise ValueError("'unit' must be 'km' or 'miles'")

        if min_distance is None and max_distance is None:
            raise ValueError("At least one of 'min_distance' or 'max_distance' required")

        # Map to DistanceUnit enum
        self._distance_unit = DistanceUnit.KILOMETERS if unit == "km" else DistanceUnit.MILES

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        # Collect data for vectorized processing
        df = lf.select([
            pl.col(self.lat1_column),
            pl.col(self.lon1_column),
            pl.col(self.lat2_column),
            pl.col(self.lon2_column),
        ]).drop_nulls().collect()

        if len(df) == 0:
            return issues

        total = len(df)

        # Extract as numpy arrays for vectorized processing
        lat1 = df[self.lat1_column].to_numpy()
        lon1 = df[self.lon1_column].to_numpy()
        lat2 = df[self.lat2_column].to_numpy()
        lon2 = df[self.lon2_column].to_numpy()

        # Compute distances using vectorized mixin
        distances = self.haversine_vectorized(lat1, lon1, lat2, lon2, self._distance_unit)

        if self.min_distance is not None:
            below_min = int(np.sum(distances < self.min_distance))
            if below_min > 0:
                if not self._passes_mostly(below_min, total):
                    issues.append(
                        ValidationIssue(
                            column=f"{self.lat1_column},{self.lon1_column} -> {self.lat2_column},{self.lon2_column}",
                            issue_type="optimized_distance_below_minimum",
                            count=below_min,
                            severity=Severity.MEDIUM,
                            details=f"{below_min} pairs with distance < {self.min_distance} {self.unit} (vectorized)",
                            expected=f"Distance >= {self.min_distance} {self.unit}",
                        )
                    )

        if self.max_distance is not None:
            above_max = int(np.sum(distances > self.max_distance))
            if above_max > 0:
                if not self._passes_mostly(above_max, total):
                    issues.append(
                        ValidationIssue(
                            column=f"{self.lat1_column},{self.lon1_column} -> {self.lat2_column},{self.lon2_column}",
                            issue_type="optimized_distance_above_maximum",
                            count=above_max,
                            severity=Severity.MEDIUM,
                            details=f"{above_max} pairs with distance > {self.max_distance} {self.unit} (vectorized)",
                            expected=f"Distance <= {self.max_distance} {self.unit}",
                        )
                    )

        return issues
