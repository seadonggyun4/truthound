"""Base classes for geospatial validators.

This module provides base classes and utilities for geospatial validation.
"""

from abc import abstractmethod
from math import radians, sin, cos, sqrt, atan2
from typing import Any

import polars as pl

from truthound.types import Severity
from truthound.validators.base import ValidationIssue, Validator

# Earth's radius in kilometers
EARTH_RADIUS_KM = 6371.0
EARTH_RADIUS_MILES = 3958.8


class GeoValidator(Validator):
    """Base class for geospatial validators.

    Provides common utilities for coordinate validation and distance calculations.
    """

    name = "geo_base"
    category = "geospatial"

    # Valid latitude range: -90 to 90
    LAT_MIN = -90.0
    LAT_MAX = 90.0

    # Valid longitude range: -180 to 180
    LON_MIN = -180.0
    LON_MAX = 180.0

    def __init__(self, **kwargs: Any):
        """Initialize geospatial validator.

        Args:
            **kwargs: Additional config passed to base Validator
        """
        super().__init__(**kwargs)

    @staticmethod
    def haversine_distance(
        lat1: float, lon1: float, lat2: float, lon2: float, unit: str = "km"
    ) -> float:
        """Calculate the great-circle distance between two points using Haversine formula.

        Args:
            lat1: Latitude of first point
            lon1: Longitude of first point
            lat2: Latitude of second point
            lon2: Longitude of second point
            unit: Distance unit ('km' or 'miles')

        Returns:
            Distance between the two points
        """
        lat1_rad = radians(lat1)
        lat2_rad = radians(lat2)
        delta_lat = radians(lat2 - lat1)
        delta_lon = radians(lon2 - lon1)

        a = sin(delta_lat / 2) ** 2 + cos(lat1_rad) * cos(lat2_rad) * sin(delta_lon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))

        radius = EARTH_RADIUS_MILES if unit == "miles" else EARTH_RADIUS_KM
        return radius * c

    @staticmethod
    def is_valid_latitude(lat: float) -> bool:
        """Check if latitude is valid."""
        return GeoValidator.LAT_MIN <= lat <= GeoValidator.LAT_MAX

    @staticmethod
    def is_valid_longitude(lon: float) -> bool:
        """Check if longitude is valid."""
        return GeoValidator.LON_MIN <= lon <= GeoValidator.LON_MAX

    @staticmethod
    def is_valid_coordinate(lat: float, lon: float) -> bool:
        """Check if coordinate pair is valid."""
        return GeoValidator.is_valid_latitude(lat) and GeoValidator.is_valid_longitude(lon)

    @abstractmethod
    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Validate the LazyFrame.

        Args:
            lf: LazyFrame to validate

        Returns:
            List of validation issues
        """
        pass
