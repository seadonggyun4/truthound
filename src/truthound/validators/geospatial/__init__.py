"""Geospatial validators.

This module provides 11 validators for geographic coordinate validation:

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

Polygon Validators (requires Shapely):
- PolygonValidator: Validate coordinates within polygon boundary
- MultiPolygonValidator: Validate coordinates within multiple polygons
- PolygonDistanceValidator: Validate distance to polygon boundary
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

# Polygon validators (optional, requires Shapely)
try:
    from truthound.validators.geospatial.polygon import (
        PolygonValidator,
        MultiPolygonValidator,
        PolygonDistanceValidator,
        SHAPELY_AVAILABLE,
    )
except ImportError:
    PolygonValidator = None
    MultiPolygonValidator = None
    PolygonDistanceValidator = None
    SHAPELY_AVAILABLE = False

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
    # Polygon (optional)
    "PolygonValidator",
    "MultiPolygonValidator",
    "PolygonDistanceValidator",
    "SHAPELY_AVAILABLE",
]
