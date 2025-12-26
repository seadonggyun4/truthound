"""Polygon validators with Shapely integration.

This module provides validators for checking if coordinates fall within
complex polygon boundaries using the Shapely library for precise geospatial
operations.

Optional Dependencies:
    - shapely: Required for polygon operations
    - geojson: Optional for GeoJSON file support

Example:
    from truthound.validators.geospatial import PolygonValidator

    # Using WKT string
    validator = PolygonValidator(
        lat_column="latitude",
        lon_column="longitude",
        polygon_wkt="POLYGON((0 0, 0 10, 10 10, 10 0, 0 0))",
    )

    # Using coordinate list
    validator = PolygonValidator(
        lat_column="lat",
        lon_column="lon",
        polygon_coords=[(0, 0), (0, 10), (10, 10), (10, 0), (0, 0)],
    )

    # Using GeoJSON file
    validator = PolygonValidator(
        lat_column="lat",
        lon_column="lon",
        geojson_file="boundary.geojson",
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import polars as pl

from truthound.types import Severity
from truthound.validators.base import ValidationIssue
from truthound.validators.geospatial.base import GeoValidator
from truthound.validators.registry import register_validator

# Type stubs for optional Shapely
try:
    from shapely import wkt
    from shapely.geometry import Point, Polygon, MultiPolygon, shape
    from shapely.prepared import prep
    from shapely.ops import unary_union

    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False
    wkt = None
    Point = None
    Polygon = None
    MultiPolygon = None
    shape = None
    prep = None
    unary_union = None

# Type stubs for optional GeoJSON
try:
    import json

    GEOJSON_AVAILABLE = True
except ImportError:
    GEOJSON_AVAILABLE = False


def _require_shapely() -> None:
    """Raise ImportError if Shapely is not available."""
    if not SHAPELY_AVAILABLE:
        raise ImportError(
            "Shapely is required for polygon validators. "
            "Install with: pip install shapely"
        )


@register_validator
class PolygonValidator(GeoValidator):
    """Validates that coordinates fall within a polygon boundary.

    Uses Shapely for precise point-in-polygon tests. Supports:
    - WKT (Well-Known Text) polygon definitions
    - Coordinate list polygons
    - GeoJSON file input
    - Multiple polygons (MultiPolygon)

    Performance:
        Uses prepared geometry for faster point-in-polygon tests.
        For very large datasets, consider using sample_size parameter.

    Example:
        # Simple polygon from coordinates
        validator = PolygonValidator(
            lat_column="lat",
            lon_column="lon",
            polygon_coords=[
                (126.9, 37.5),  # Seoul area
                (127.1, 37.5),
                (127.1, 37.6),
                (126.9, 37.6),
                (126.9, 37.5),
            ],
        )

        # From WKT string
        validator = PolygonValidator(
            lat_column="lat",
            lon_column="lon",
            polygon_wkt="POLYGON((126.9 37.5, 127.1 37.5, 127.1 37.6, 126.9 37.6, 126.9 37.5))",
        )

        # From GeoJSON file
        validator = PolygonValidator(
            lat_column="lat",
            lon_column="lon",
            geojson_file="seoul_district.geojson",
        )
    """

    name = "polygon"
    category = "geospatial"

    def __init__(
        self,
        lat_column: str,
        lon_column: str,
        polygon_wkt: str | None = None,
        polygon_coords: Sequence[tuple[float, float]] | None = None,
        geojson_file: str | Path | None = None,
        geojson_dict: dict | None = None,
        boundary_name: str | None = None,
        sample_size: int | None = None,
        **kwargs: Any,
    ):
        """Initialize polygon validator.

        Args:
            lat_column: Column containing latitude values
            lon_column: Column containing longitude values
            polygon_wkt: Polygon definition in WKT format
            polygon_coords: Polygon as list of (lon, lat) tuples
            geojson_file: Path to GeoJSON file containing polygon
            geojson_dict: GeoJSON dictionary containing polygon
            boundary_name: Human-readable name for the boundary
            sample_size: If set, validate on a sample of rows
            **kwargs: Additional config

        Note:
            Coordinates in polygon_coords should be (longitude, latitude) order,
            following GeoJSON convention.
        """
        _require_shapely()
        super().__init__(**kwargs)

        self.lat_column = lat_column
        self.lon_column = lon_column
        self.boundary_name = boundary_name
        self.sample_size = sample_size

        # Build the geometry
        self._geometry = self._build_geometry(
            polygon_wkt, polygon_coords, geojson_file, geojson_dict
        )

        # Prepare geometry for faster operations
        self._prepared = prep(self._geometry)

    def _build_geometry(
        self,
        polygon_wkt: str | None,
        polygon_coords: Sequence[tuple[float, float]] | None,
        geojson_file: str | Path | None,
        geojson_dict: dict | None,
    ):
        """Build Shapely geometry from input.

        Args:
            polygon_wkt: WKT string
            polygon_coords: Coordinate list
            geojson_file: Path to GeoJSON file
            geojson_dict: GeoJSON dictionary

        Returns:
            Shapely geometry (Polygon or MultiPolygon)

        Raises:
            ValueError: If no valid input provided or geometry is invalid
        """
        inputs_count = sum(
            x is not None for x in [polygon_wkt, polygon_coords, geojson_file, geojson_dict]
        )

        if inputs_count == 0:
            raise ValueError(
                "One of polygon_wkt, polygon_coords, geojson_file, or geojson_dict must be provided"
            )
        if inputs_count > 1:
            raise ValueError(
                "Only one of polygon_wkt, polygon_coords, geojson_file, or geojson_dict should be provided"
            )

        geometry = None

        if polygon_wkt:
            geometry = wkt.loads(polygon_wkt)

        elif polygon_coords:
            # Coords are (lon, lat) tuples
            geometry = Polygon(polygon_coords)

        elif geojson_file:
            geojson_path = Path(geojson_file)
            if not geojson_path.exists():
                raise ValueError(f"GeoJSON file not found: {geojson_file}")

            with open(geojson_path) as f:
                geojson_data = json.load(f)
            geometry = self._geometry_from_geojson(geojson_data)

        elif geojson_dict:
            geometry = self._geometry_from_geojson(geojson_dict)

        if geometry is None or geometry.is_empty:
            raise ValueError("Could not create valid geometry from input")

        if not geometry.is_valid:
            # Attempt to fix invalid geometry
            geometry = geometry.buffer(0)
            if not geometry.is_valid:
                raise ValueError("Invalid geometry that could not be fixed")

        return geometry

    def _geometry_from_geojson(self, geojson_data: dict):
        """Extract geometry from GeoJSON data.

        Handles:
        - Feature with geometry
        - FeatureCollection (unions all geometries)
        - Direct geometry object

        Args:
            geojson_data: GeoJSON dictionary

        Returns:
            Shapely geometry
        """
        geojson_type = geojson_data.get("type")

        if geojson_type == "Feature":
            return shape(geojson_data["geometry"])

        elif geojson_type == "FeatureCollection":
            features = geojson_data.get("features", [])
            if not features:
                raise ValueError("FeatureCollection contains no features")

            geometries = [shape(f["geometry"]) for f in features]
            return unary_union(geometries)

        elif geojson_type in ("Polygon", "MultiPolygon"):
            return shape(geojson_data)

        else:
            raise ValueError(f"Unsupported GeoJSON type: {geojson_type}")

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Validate that coordinates fall within the polygon.

        Args:
            lf: LazyFrame with coordinate columns

        Returns:
            List of validation issues
        """
        issues: list[ValidationIssue] = []

        # Get coordinates
        df = lf.select([
            pl.col(self.lat_column),
            pl.col(self.lon_column),
        ])

        # Apply sampling if configured
        if self.sample_size:
            total_rows = df.select(pl.len()).collect().item()
            if total_rows > self.sample_size:
                df = df.collect().sample(n=self.sample_size, seed=42)
                was_sampled = True
                sample_count = self.sample_size
            else:
                df = df.collect()
                was_sampled = False
                sample_count = total_rows
        else:
            df = df.collect()
            was_sampled = False
            total_rows = len(df)
            sample_count = total_rows

        # Check each point
        outside_count = 0
        sample_outside: list[dict] = []

        lat_series = df[self.lat_column]
        lon_series = df[self.lon_column]

        for i in range(len(df)):
            lat = lat_series[i]
            lon = lon_series[i]

            # Skip nulls
            if lat is None or lon is None:
                continue

            # Create point (lon, lat order for Shapely)
            point = Point(lon, lat)

            if not self._prepared.contains(point):
                outside_count += 1
                if len(sample_outside) < 5:
                    sample_outside.append({"lat": lat, "lon": lon})

        if outside_count > 0:
            # Estimate total if sampled
            if was_sampled:
                estimated_outside = int((outside_count / sample_count) * total_rows)
                ratio = outside_count / sample_count
            else:
                estimated_outside = outside_count
                ratio = outside_count / total_rows if total_rows > 0 else 0

            if self._passes_mostly(estimated_outside, total_rows):
                return issues

            boundary_desc = self.boundary_name or "polygon boundary"

            if was_sampled:
                details = (
                    f"Found {outside_count} coordinates outside {boundary_desc} in sample of "
                    f"{sample_count:,} rows ({ratio:.2%}). Estimated {estimated_outside:,} total "
                    f"violations. Samples: {sample_outside}"
                )
            else:
                details = (
                    f"{outside_count} coordinates ({ratio:.2%}) outside {boundary_desc}. "
                    f"Samples: {sample_outside}"
                )

            issues.append(
                ValidationIssue(
                    column=f"{self.lat_column}, {self.lon_column}",
                    issue_type="outside_polygon",
                    count=estimated_outside,
                    severity=Severity.MEDIUM,
                    details=details,
                    expected=f"Coordinates within {boundary_desc}",
                )
            )

        return issues

    @property
    def geometry(self):
        """Get the underlying Shapely geometry."""
        return self._geometry

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Get the bounding box of the polygon.

        Returns:
            Tuple of (min_lon, min_lat, max_lon, max_lat)
        """
        return self._geometry.bounds


@register_validator
class MultiPolygonValidator(GeoValidator):
    """Validates that coordinates fall within any of multiple polygons.

    Useful for validating against multiple regions or complex boundaries
    that cannot be represented by a single polygon.

    Example:
        # Multiple city boundaries
        validator = MultiPolygonValidator(
            lat_column="lat",
            lon_column="lon",
            polygons=[
                {"name": "Seoul", "wkt": "POLYGON(...)"},
                {"name": "Busan", "wkt": "POLYGON(...)"},
                {"name": "Incheon", "coords": [...]},
            ],
        )
    """

    name = "multi_polygon"
    category = "geospatial"

    def __init__(
        self,
        lat_column: str,
        lon_column: str,
        polygons: list[dict[str, Any]],
        require_all: bool = False,
        sample_size: int | None = None,
        **kwargs: Any,
    ):
        """Initialize multi-polygon validator.

        Args:
            lat_column: Column containing latitude values
            lon_column: Column containing longitude values
            polygons: List of polygon definitions. Each dict can contain:
                - name: Optional name for the polygon
                - wkt: WKT string
                - coords: List of (lon, lat) tuples
                - geojson: GeoJSON dictionary
            require_all: If True, points must be in ALL polygons
            sample_size: If set, validate on a sample of rows
            **kwargs: Additional config
        """
        _require_shapely()
        super().__init__(**kwargs)

        self.lat_column = lat_column
        self.lon_column = lon_column
        self.require_all = require_all
        self.sample_size = sample_size

        # Build all polygons
        self._polygons: list[tuple[str, Any]] = []
        for i, poly_def in enumerate(polygons):
            name = poly_def.get("name", f"polygon_{i}")
            geom = self._build_single_polygon(poly_def)
            self._polygons.append((name, prep(geom)))

        if not self._polygons:
            raise ValueError("At least one polygon must be provided")

    def _build_single_polygon(self, poly_def: dict[str, Any]):
        """Build a single polygon from definition."""
        if "wkt" in poly_def:
            return wkt.loads(poly_def["wkt"])
        elif "coords" in poly_def:
            return Polygon(poly_def["coords"])
        elif "geojson" in poly_def:
            return shape(poly_def["geojson"])
        else:
            raise ValueError(f"Polygon definition must contain 'wkt', 'coords', or 'geojson': {poly_def}")

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Validate coordinates against multiple polygons."""
        issues: list[ValidationIssue] = []

        df = lf.select([
            pl.col(self.lat_column),
            pl.col(self.lon_column),
        ]).collect()

        if self.sample_size and len(df) > self.sample_size:
            df = df.sample(n=self.sample_size, seed=42)

        total_rows = len(df)
        outside_count = 0

        lat_series = df[self.lat_column]
        lon_series = df[self.lon_column]

        for i in range(len(df)):
            lat = lat_series[i]
            lon = lon_series[i]

            if lat is None or lon is None:
                continue

            point = Point(lon, lat)

            if self.require_all:
                # Must be in ALL polygons
                in_all = all(prepared.contains(point) for _, prepared in self._polygons)
                if not in_all:
                    outside_count += 1
            else:
                # Must be in ANY polygon
                in_any = any(prepared.contains(point) for _, prepared in self._polygons)
                if not in_any:
                    outside_count += 1

        if outside_count > 0:
            ratio = outside_count / total_rows if total_rows > 0 else 0

            if self._passes_mostly(outside_count, total_rows):
                return issues

            polygon_names = [name for name, _ in self._polygons]
            mode = "all of" if self.require_all else "any of"

            issues.append(
                ValidationIssue(
                    column=f"{self.lat_column}, {self.lon_column}",
                    issue_type="outside_multi_polygon",
                    count=outside_count,
                    severity=Severity.MEDIUM,
                    details=f"{outside_count} coordinates ({ratio:.2%}) outside {mode}: {polygon_names}",
                    expected=f"Coordinates within {mode} polygons",
                )
            )

        return issues


@register_validator
class PolygonDistanceValidator(GeoValidator):
    """Validates distance from coordinates to a polygon boundary.

    Useful for checking proximity to geographic features like:
    - Distance from coastline
    - Distance from city center
    - Buffer zone validation

    Example:
        # Must be within 10km of city center
        validator = PolygonDistanceValidator(
            lat_column="lat",
            lon_column="lon",
            polygon_wkt="POINT(127.0 37.5)",  # Seoul center
            max_distance_km=10.0,
        )
    """

    name = "polygon_distance"
    category = "geospatial"

    def __init__(
        self,
        lat_column: str,
        lon_column: str,
        polygon_wkt: str | None = None,
        polygon_coords: Sequence[tuple[float, float]] | None = None,
        max_distance_km: float | None = None,
        min_distance_km: float | None = None,
        boundary_name: str | None = None,
        sample_size: int | None = None,
        **kwargs: Any,
    ):
        """Initialize polygon distance validator.

        Args:
            lat_column: Column containing latitude values
            lon_column: Column containing longitude values
            polygon_wkt: Polygon/Point definition in WKT format
            polygon_coords: Polygon as list of (lon, lat) tuples
            max_distance_km: Maximum allowed distance in kilometers
            min_distance_km: Minimum required distance in kilometers
            boundary_name: Human-readable name for the boundary
            sample_size: If set, validate on a sample of rows
            **kwargs: Additional config
        """
        _require_shapely()
        super().__init__(**kwargs)

        self.lat_column = lat_column
        self.lon_column = lon_column
        self.max_distance_km = max_distance_km
        self.min_distance_km = min_distance_km
        self.boundary_name = boundary_name
        self.sample_size = sample_size

        if max_distance_km is None and min_distance_km is None:
            raise ValueError("At least one of max_distance_km or min_distance_km must be provided")

        # Build geometry
        if polygon_wkt:
            self._geometry = wkt.loads(polygon_wkt)
        elif polygon_coords:
            self._geometry = Polygon(polygon_coords)
        else:
            raise ValueError("Either polygon_wkt or polygon_coords must be provided")

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Validate distance to polygon."""
        issues: list[ValidationIssue] = []

        df = lf.select([
            pl.col(self.lat_column),
            pl.col(self.lon_column),
        ]).collect()

        if self.sample_size and len(df) > self.sample_size:
            df = df.sample(n=self.sample_size, seed=42)

        total_rows = len(df)
        too_far_count = 0
        too_close_count = 0

        lat_series = df[self.lat_column]
        lon_series = df[self.lon_column]

        for i in range(len(df)):
            lat = lat_series[i]
            lon = lon_series[i]

            if lat is None or lon is None:
                continue

            point = Point(lon, lat)

            # Approximate distance in km (using degrees, rough approximation)
            # For precise distance, would need proper projection
            distance_deg = point.distance(self._geometry)
            # Rough conversion: 1 degree ~ 111 km at equator
            distance_km = distance_deg * 111.0

            if self.max_distance_km is not None and distance_km > self.max_distance_km:
                too_far_count += 1

            if self.min_distance_km is not None and distance_km < self.min_distance_km:
                too_close_count += 1

        boundary_desc = self.boundary_name or "boundary"

        if too_far_count > 0:
            ratio = too_far_count / total_rows if total_rows > 0 else 0
            if not self._passes_mostly(too_far_count, total_rows):
                issues.append(
                    ValidationIssue(
                        column=f"{self.lat_column}, {self.lon_column}",
                        issue_type="too_far_from_boundary",
                        count=too_far_count,
                        severity=Severity.MEDIUM,
                        details=(
                            f"{too_far_count} coordinates ({ratio:.2%}) more than "
                            f"{self.max_distance_km}km from {boundary_desc}"
                        ),
                        expected=f"Within {self.max_distance_km}km of {boundary_desc}",
                    )
                )

        if too_close_count > 0:
            ratio = too_close_count / total_rows if total_rows > 0 else 0
            if not self._passes_mostly(too_close_count, total_rows):
                issues.append(
                    ValidationIssue(
                        column=f"{self.lat_column}, {self.lon_column}",
                        issue_type="too_close_to_boundary",
                        count=too_close_count,
                        severity=Severity.MEDIUM,
                        details=(
                            f"{too_close_count} coordinates ({ratio:.2%}) less than "
                            f"{self.min_distance_km}km from {boundary_desc}"
                        ),
                        expected=f"At least {self.min_distance_km}km from {boundary_desc}",
                    )
                )

        return issues
