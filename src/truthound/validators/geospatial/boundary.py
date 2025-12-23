"""Boundary validators.

Validators for checking if coordinates fall within geographic boundaries.
"""

from typing import Any

import polars as pl

from truthound.types import Severity
from truthound.validators.base import ValidationIssue
from truthound.validators.geospatial.base import GeoValidator
from truthound.validators.registry import register_validator


@register_validator
class GeoBoundingBoxValidator(GeoValidator):
    """Validates that coordinates fall within a bounding box.

    Example:
        # Coordinates should be within South Korea
        validator = GeoBoundingBoxValidator(
            lat_column="latitude",
            lon_column="longitude",
            min_lat=33.0,
            max_lat=43.0,
            min_lon=124.0,
            max_lon=132.0,
        )

        # Coordinates should be within continental US
        validator = GeoBoundingBoxValidator(
            lat_column="lat",
            lon_column="lon",
            min_lat=24.396308,
            max_lat=49.384358,
            min_lon=-125.0,
            max_lon=-66.93457,
        )
    """

    name = "geo_bounding_box"
    category = "geospatial"

    # Common region bounding boxes
    REGIONS = {
        "korea": {"min_lat": 33.0, "max_lat": 43.0, "min_lon": 124.0, "max_lon": 132.0},
        "japan": {"min_lat": 24.0, "max_lat": 46.0, "min_lon": 122.0, "max_lon": 154.0},
        "us_continental": {"min_lat": 24.4, "max_lat": 49.4, "min_lon": -125.0, "max_lon": -66.9},
        "europe": {"min_lat": 35.0, "max_lat": 72.0, "min_lon": -25.0, "max_lon": 65.0},
        "china": {"min_lat": 18.0, "max_lat": 54.0, "min_lon": 73.0, "max_lon": 135.0},
    }

    def __init__(
        self,
        lat_column: str,
        lon_column: str,
        min_lat: float | None = None,
        max_lat: float | None = None,
        min_lon: float | None = None,
        max_lon: float | None = None,
        region: str | None = None,
        **kwargs: Any,
    ):
        """Initialize bounding box validator.

        Args:
            lat_column: Column containing latitude values
            lon_column: Column containing longitude values
            min_lat: Minimum latitude of bounding box
            max_lat: Maximum latitude of bounding box
            min_lon: Minimum longitude of bounding box
            max_lon: Maximum longitude of bounding box
            region: Predefined region name (overrides explicit bounds)
            **kwargs: Additional config
        """
        super().__init__(**kwargs)
        self.lat_column = lat_column
        self.lon_column = lon_column

        # Use predefined region if specified
        if region:
            if region not in self.REGIONS:
                raise ValueError(f"Unknown region: {region}. Available: {list(self.REGIONS.keys())}")
            bounds = self.REGIONS[region]
            self.min_lat = bounds["min_lat"]
            self.max_lat = bounds["max_lat"]
            self.min_lon = bounds["min_lon"]
            self.max_lon = bounds["max_lon"]
            self.region = region
        else:
            if min_lat is None or max_lat is None or min_lon is None or max_lon is None:
                raise ValueError("All bounds (min_lat, max_lat, min_lon, max_lon) required unless using 'region'")
            self.min_lat = min_lat
            self.max_lat = max_lat
            self.min_lon = min_lon
            self.max_lon = max_lon
            self.region = None

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        outside_box = (
            (pl.col(self.lat_column) < self.min_lat) |
            (pl.col(self.lat_column) > self.max_lat) |
            (pl.col(self.lon_column) < self.min_lon) |
            (pl.col(self.lon_column) > self.max_lon)
        )

        result = lf.select([
            pl.len().alias("_total"),
            outside_box.sum().alias("_outside"),
        ]).collect()

        total = result["_total"][0]
        outside = result["_outside"][0]

        if outside > 0:
            if self._passes_mostly(outside, total):
                return issues

            region_name = self.region or f"[{self.min_lat},{self.max_lat}] x [{self.min_lon},{self.max_lon}]"
            issues.append(
                ValidationIssue(
                    column=f"{self.lat_column}, {self.lon_column}",
                    issue_type="outside_bounding_box",
                    count=outside,
                    severity=Severity.MEDIUM,
                    details=f"{outside} coordinates outside bounding box {region_name}",
                    expected=f"Coordinates within {region_name}",
                )
            )

        return issues


@register_validator
class GeoCountryValidator(GeoValidator):
    """Validates that coordinates fall within a country's approximate bounds.

    Uses simplified rectangular bounds for quick validation.
    For precise validation, use a GIS library.

    Example:
        validator = GeoCountryValidator(
            lat_column="lat",
            lon_column="lon",
            country="korea",
        )
    """

    name = "geo_country"
    category = "geospatial"

    # Approximate country bounds (simplified rectangles)
    COUNTRY_BOUNDS = {
        "korea": {"min_lat": 33.0, "max_lat": 43.0, "min_lon": 124.0, "max_lon": 132.0},
        "korea_south": {"min_lat": 33.0, "max_lat": 38.6, "min_lon": 124.0, "max_lon": 132.0},
        "korea_north": {"min_lat": 37.5, "max_lat": 43.0, "min_lon": 124.0, "max_lon": 131.0},
        "japan": {"min_lat": 24.0, "max_lat": 46.0, "min_lon": 122.0, "max_lon": 154.0},
        "china": {"min_lat": 18.0, "max_lat": 54.0, "min_lon": 73.0, "max_lon": 135.0},
        "usa": {"min_lat": 18.9, "max_lat": 71.4, "min_lon": -179.1, "max_lon": -66.9},
        "uk": {"min_lat": 49.9, "max_lat": 60.9, "min_lon": -8.6, "max_lon": 1.8},
        "germany": {"min_lat": 47.3, "max_lat": 55.1, "min_lon": 5.9, "max_lon": 15.0},
        "france": {"min_lat": 41.3, "max_lat": 51.1, "min_lon": -5.1, "max_lon": 9.6},
        "australia": {"min_lat": -43.6, "max_lat": -10.7, "min_lon": 113.2, "max_lon": 153.6},
        "brazil": {"min_lat": -33.8, "max_lat": 5.3, "min_lon": -73.9, "max_lon": -34.8},
        "india": {"min_lat": 6.7, "max_lat": 35.5, "min_lon": 68.2, "max_lon": 97.4},
        "russia": {"min_lat": 41.2, "max_lat": 81.9, "min_lon": 19.6, "max_lon": 180.0},
        "canada": {"min_lat": 41.7, "max_lat": 83.1, "min_lon": -141.0, "max_lon": -52.6},
    }

    def __init__(
        self,
        lat_column: str,
        lon_column: str,
        country: str,
        **kwargs: Any,
    ):
        """Initialize country validator.

        Args:
            lat_column: Column containing latitude values
            lon_column: Column containing longitude values
            country: Country code/name
            **kwargs: Additional config
        """
        super().__init__(**kwargs)
        self.lat_column = lat_column
        self.lon_column = lon_column

        country_lower = country.lower()
        if country_lower not in self.COUNTRY_BOUNDS:
            raise ValueError(
                f"Unknown country: {country}. Available: {list(self.COUNTRY_BOUNDS.keys())}"
            )

        self.country = country_lower
        bounds = self.COUNTRY_BOUNDS[country_lower]
        self.min_lat = bounds["min_lat"]
        self.max_lat = bounds["max_lat"]
        self.min_lon = bounds["min_lon"]
        self.max_lon = bounds["max_lon"]

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        outside_country = (
            (pl.col(self.lat_column) < self.min_lat) |
            (pl.col(self.lat_column) > self.max_lat) |
            (pl.col(self.lon_column) < self.min_lon) |
            (pl.col(self.lon_column) > self.max_lon)
        )

        result = lf.select([
            pl.len().alias("_total"),
            outside_country.sum().alias("_outside"),
        ]).collect()

        total = result["_total"][0]
        outside = result["_outside"][0]

        if outside > 0:
            if self._passes_mostly(outside, total):
                return issues

            issues.append(
                ValidationIssue(
                    column=f"{self.lat_column}, {self.lon_column}",
                    issue_type="outside_country_bounds",
                    count=outside,
                    severity=Severity.MEDIUM,
                    details=f"{outside} coordinates outside {self.country} bounds",
                    expected=f"Coordinates within {self.country}",
                )
            )

        return issues
