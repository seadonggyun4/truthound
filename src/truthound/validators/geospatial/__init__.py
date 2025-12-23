"""Geospatial validators.

This module provides 8 validators for geographic coordinate validation:

Coordinate Validators:
- LatitudeValidator: Validate latitude values within range
- LongitudeValidator: Validate longitude values within range
- CoordinateValidator: Validate coordinate pairs (lat, lon)
- CoordinateNotNullIslandValidator: Detect (0, 0) "Null Island" coordinates

Distance Validators:
- GeoDistanceValidator: Validate distance between two coordinate pairs
- GeoDistanceFromPointValidator: Validate distance from a reference point

Boundary Validators:
- GeoBoundingBoxValidator: Validate coordinates within bounding box
- GeoCountryValidator: Validate coordinates within country bounds
"""

from truthound.validators.geospatial.base import (
    GeoValidator,
    EARTH_RADIUS_KM,
    EARTH_RADIUS_MILES,
)
from truthound.validators.geospatial.coordinate import (
    LatitudeValidator,
    LongitudeValidator,
    CoordinateValidator,
    CoordinateNotNullIslandValidator,
)
from truthound.validators.geospatial.distance import (
    GeoDistanceValidator,
    GeoDistanceFromPointValidator,
)
from truthound.validators.geospatial.boundary import (
    GeoBoundingBoxValidator,
    GeoCountryValidator,
)

__all__ = [
    # Base
    "GeoValidator",
    "EARTH_RADIUS_KM",
    "EARTH_RADIUS_MILES",
    # Coordinate
    "LatitudeValidator",
    "LongitudeValidator",
    "CoordinateValidator",
    "CoordinateNotNullIslandValidator",
    # Distance
    "GeoDistanceValidator",
    "GeoDistanceFromPointValidator",
    # Boundary
    "GeoBoundingBoxValidator",
    "GeoCountryValidator",
]
