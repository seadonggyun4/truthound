"""Vectorized geospatial computation algorithms.

This module provides memory-efficient and vectorized implementations
of common geospatial operations for distance and polygon validators.

Key Optimizations:
    - Vectorized Haversine using NumPy broadcasting
    - Batch distance matrix computation
    - Spatial indexing for nearest neighbor queries
    - Chunked processing for large datasets

Usage:
    class OptimizedGeoValidator(GeoDistanceValidator, VectorizedGeoMixin):
        def _compute_distances(self, coords1, coords2):
            return self.haversine_vectorized(coords1, coords2)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator
from enum import Enum, auto

import numpy as np


class DistanceUnit(Enum):
    """Distance units for geospatial calculations."""

    METERS = auto()
    KILOMETERS = auto()
    MILES = auto()
    NAUTICAL_MILES = auto()


# Earth radius in different units
EARTH_RADIUS = {
    DistanceUnit.METERS: 6_371_000.0,
    DistanceUnit.KILOMETERS: 6_371.0,
    DistanceUnit.MILES: 3_958.8,
    DistanceUnit.NAUTICAL_MILES: 3_440.1,
}


@dataclass
class BoundingBox:
    """Geographic bounding box.

    Attributes:
        min_lat: Minimum latitude
        max_lat: Maximum latitude
        min_lon: Minimum longitude
        max_lon: Maximum longitude
    """

    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float

    def contains(self, lat: float, lon: float) -> bool:
        """Check if point is inside bounding box."""
        return (
            self.min_lat <= lat <= self.max_lat
            and self.min_lon <= lon <= self.max_lon
        )

    def contains_vectorized(
        self, lats: np.ndarray, lons: np.ndarray
    ) -> np.ndarray:
        """Check if points are inside bounding box (vectorized)."""
        return (
            (lats >= self.min_lat)
            & (lats <= self.max_lat)
            & (lons >= self.min_lon)
            & (lons <= self.max_lon)
        )

    def expand(self, margin_degrees: float) -> "BoundingBox":
        """Expand bounding box by margin."""
        return BoundingBox(
            min_lat=max(-90.0, self.min_lat - margin_degrees),
            max_lat=min(90.0, self.max_lat + margin_degrees),
            min_lon=self.min_lon - margin_degrees,
            max_lon=self.max_lon + margin_degrees,
        )

    @classmethod
    def from_points(
        cls, lats: np.ndarray, lons: np.ndarray
    ) -> "BoundingBox":
        """Create bounding box from points."""
        return cls(
            min_lat=float(np.min(lats)),
            max_lat=float(np.max(lats)),
            min_lon=float(np.min(lons)),
            max_lon=float(np.max(lons)),
        )


class VectorizedGeoMixin:
    """Mixin providing vectorized geospatial computations.

    Use in validators that need efficient distance calculations
    or spatial filtering.

    Features:
        - Vectorized Haversine distance (no loops)
        - Batch pairwise distance matrices
        - Spatial indexing integration
        - Memory-efficient chunked processing

    Example:
        class OptimizedGeoValidator(GeoDistanceValidator, VectorizedGeoMixin):
            def validate_distances(self, coords):
                distances = self.haversine_vectorized(
                    coords[:, 0], coords[:, 1],
                    target_lat, target_lon
                )
                return distances < self.max_distance
    """

    # Configuration
    _geo_chunk_size: int = 100000
    _default_unit: DistanceUnit = DistanceUnit.KILOMETERS

    def haversine_vectorized(
        self,
        lat1: np.ndarray | float,
        lon1: np.ndarray | float,
        lat2: np.ndarray | float,
        lon2: np.ndarray | float,
        unit: DistanceUnit | None = None,
    ) -> np.ndarray:
        """Compute Haversine distance (vectorized).

        Computes great-circle distance between points on a sphere.
        Fully vectorized using NumPy broadcasting.

        Args:
            lat1: Latitude(s) of first point(s) in degrees
            lon1: Longitude(s) of first point(s) in degrees
            lat2: Latitude(s) of second point(s) in degrees
            lon2: Longitude(s) of second point(s) in degrees
            unit: Distance unit (default: kilometers)

        Returns:
            Distance(s) in specified unit
        """
        unit = unit or self._default_unit
        radius = EARTH_RADIUS[unit]

        # Convert to radians
        lat1_rad = np.radians(lat1)
        lon1_rad = np.radians(lon1)
        lat2_rad = np.radians(lat2)
        lon2_rad = np.radians(lon2)

        # Haversine formula
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad

        a = (
            np.sin(dlat / 2) ** 2
            + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
        )
        c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

        return radius * c

    def pairwise_distances(
        self,
        lats1: np.ndarray,
        lons1: np.ndarray,
        lats2: np.ndarray,
        lons2: np.ndarray,
        unit: DistanceUnit | None = None,
    ) -> np.ndarray:
        """Compute pairwise distance matrix.

        Computes distance from each point in set 1 to each point in set 2.
        Uses broadcasting for memory efficiency.

        Args:
            lats1: Latitudes of first set (n,)
            lons1: Longitudes of first set (n,)
            lats2: Latitudes of second set (m,)
            lons2: Longitudes of second set (m,)
            unit: Distance unit

        Returns:
            Distance matrix of shape (n, m)
        """
        # Reshape for broadcasting: (n, 1) vs (1, m)
        lats1 = np.asarray(lats1).reshape(-1, 1)
        lons1 = np.asarray(lons1).reshape(-1, 1)
        lats2 = np.asarray(lats2).reshape(1, -1)
        lons2 = np.asarray(lons2).reshape(1, -1)

        return self.haversine_vectorized(lats1, lons1, lats2, lons2, unit)

    def pairwise_distances_chunked(
        self,
        lats1: np.ndarray,
        lons1: np.ndarray,
        lats2: np.ndarray,
        lons2: np.ndarray,
        chunk_size: int | None = None,
        unit: DistanceUnit | None = None,
    ) -> Iterator[tuple[int, int, np.ndarray]]:
        """Compute pairwise distances in chunks.

        Memory-efficient version that yields chunks of the distance matrix
        instead of building the full matrix.

        Args:
            lats1: Latitudes of first set (n,)
            lons1: Longitudes of first set (n,)
            lats2: Latitudes of second set (m,)
            lons2: Longitudes of second set (m,)
            chunk_size: Rows per chunk
            unit: Distance unit

        Yields:
            Tuples of (start_idx, end_idx, distance_chunk)
        """
        chunk_size = chunk_size or self._geo_chunk_size
        n = len(lats1)

        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)

            chunk_distances = self.pairwise_distances(
                lats1[start:end],
                lons1[start:end],
                lats2,
                lons2,
                unit,
            )

            yield start, end, chunk_distances

    def nearest_point(
        self,
        query_lat: float,
        query_lon: float,
        lats: np.ndarray,
        lons: np.ndarray,
        unit: DistanceUnit | None = None,
    ) -> tuple[int, float]:
        """Find nearest point to query.

        Args:
            query_lat: Query latitude
            query_lon: Query longitude
            lats: Candidate latitudes
            lons: Candidate longitudes
            unit: Distance unit

        Returns:
            Tuple of (nearest_index, distance)
        """
        distances = self.haversine_vectorized(
            query_lat, query_lon, lats, lons, unit
        )
        idx = int(np.argmin(distances))
        return idx, float(distances[idx])

    def k_nearest_points(
        self,
        query_lat: float,
        query_lon: float,
        lats: np.ndarray,
        lons: np.ndarray,
        k: int,
        unit: DistanceUnit | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Find k nearest points to query.

        Args:
            query_lat: Query latitude
            query_lon: Query longitude
            lats: Candidate latitudes
            lons: Candidate longitudes
            k: Number of neighbors
            unit: Distance unit

        Returns:
            Tuple of (indices, distances) for k nearest
        """
        distances = self.haversine_vectorized(
            query_lat, query_lon, lats, lons, unit
        )

        k = min(k, len(distances))
        indices = np.argpartition(distances, k - 1)[:k]
        sorted_order = np.argsort(distances[indices])
        indices = indices[sorted_order]

        return indices, distances[indices]

    def points_within_radius(
        self,
        center_lat: float,
        center_lon: float,
        lats: np.ndarray,
        lons: np.ndarray,
        radius: float,
        unit: DistanceUnit | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Find all points within radius of center.

        Args:
            center_lat: Center latitude
            center_lon: Center longitude
            lats: Candidate latitudes
            lons: Candidate longitudes
            radius: Maximum distance
            unit: Distance unit

        Returns:
            Tuple of (indices, distances) for points within radius
        """
        distances = self.haversine_vectorized(
            center_lat, center_lon, lats, lons, unit
        )

        mask = distances <= radius
        indices = np.where(mask)[0]

        return indices, distances[indices]

    def bearing_vectorized(
        self,
        lat1: np.ndarray | float,
        lon1: np.ndarray | float,
        lat2: np.ndarray | float,
        lon2: np.ndarray | float,
    ) -> np.ndarray:
        """Compute initial bearing from point 1 to point 2.

        Args:
            lat1: Latitude(s) of starting point(s)
            lon1: Longitude(s) of starting point(s)
            lat2: Latitude(s) of ending point(s)
            lon2: Longitude(s) of ending point(s)

        Returns:
            Bearing(s) in degrees (0-360)
        """
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        dlon = np.radians(lon2 - lon1)

        x = np.sin(dlon) * np.cos(lat2_rad)
        y = (
            np.cos(lat1_rad) * np.sin(lat2_rad)
            - np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(dlon)
        )

        bearing = np.degrees(np.arctan2(x, y))
        return (bearing + 360) % 360

    def destination_point(
        self,
        lat: float,
        lon: float,
        bearing: float,
        distance: float,
        unit: DistanceUnit | None = None,
    ) -> tuple[float, float]:
        """Compute destination point given start, bearing, and distance.

        Args:
            lat: Starting latitude
            lon: Starting longitude
            bearing: Bearing in degrees
            distance: Distance to travel
            unit: Distance unit

        Returns:
            Tuple of (destination_lat, destination_lon)
        """
        unit = unit or self._default_unit
        radius = EARTH_RADIUS[unit]

        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        bearing_rad = np.radians(bearing)

        angular_distance = distance / radius

        dest_lat = np.arcsin(
            np.sin(lat_rad) * np.cos(angular_distance)
            + np.cos(lat_rad) * np.sin(angular_distance) * np.cos(bearing_rad)
        )

        dest_lon = lon_rad + np.arctan2(
            np.sin(bearing_rad) * np.sin(angular_distance) * np.cos(lat_rad),
            np.cos(angular_distance) - np.sin(lat_rad) * np.sin(dest_lat),
        )

        return float(np.degrees(dest_lat)), float(np.degrees(dest_lon))

    def create_bounding_box(
        self,
        center_lat: float,
        center_lon: float,
        radius: float,
        unit: DistanceUnit | None = None,
    ) -> BoundingBox:
        """Create bounding box around center point.

        Args:
            center_lat: Center latitude
            center_lon: Center longitude
            radius: Radius in specified unit
            unit: Distance unit

        Returns:
            BoundingBox that contains the circle
        """
        # Compute corner points
        north_lat, _ = self.destination_point(center_lat, center_lon, 0, radius, unit)
        south_lat, _ = self.destination_point(center_lat, center_lon, 180, radius, unit)
        _, east_lon = self.destination_point(center_lat, center_lon, 90, radius, unit)
        _, west_lon = self.destination_point(center_lat, center_lon, 270, radius, unit)

        return BoundingBox(
            min_lat=south_lat,
            max_lat=north_lat,
            min_lon=west_lon,
            max_lon=east_lon,
        )

    def filter_by_bounding_box(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
        bbox: BoundingBox,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Filter points by bounding box.

        Fast pre-filter before expensive distance calculations.

        Args:
            lats: Latitudes
            lons: Longitudes
            bbox: Bounding box

        Returns:
            Tuple of (mask, filtered_lats, filtered_lons)
        """
        mask = bbox.contains_vectorized(lats, lons)
        return mask, lats[mask], lons[mask]

    def validate_coordinates(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
    ) -> np.ndarray:
        """Validate coordinate ranges.

        Args:
            lats: Latitudes to validate
            lons: Longitudes to validate

        Returns:
            Boolean mask of valid coordinates
        """
        valid_lat = (lats >= -90) & (lats <= 90)
        valid_lon = (lons >= -180) & (lons <= 180)
        return valid_lat & valid_lon


class SpatialIndexMixin:
    """Mixin for spatial indexing support.

    Provides integration with spatial index libraries (BallTree, R-tree)
    for efficient nearest neighbor queries on geographic data.

    Example:
        class IndexedGeoValidator(GeoDistanceValidator, SpatialIndexMixin):
            def setup(self, reference_coords):
                self.build_spatial_index(reference_coords)

            def find_nearest(self, query_coords):
                return self.query_nearest(query_coords, k=5)
    """

    _spatial_index: Any = None
    _index_coords: np.ndarray | None = None

    def build_spatial_index(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
        leaf_size: int = 40,
    ) -> None:
        """Build spatial index from coordinates.

        Uses BallTree with Haversine metric for efficient queries.

        Args:
            lats: Latitudes in degrees
            lons: Longitudes in degrees
            leaf_size: BallTree leaf size
        """
        try:
            from sklearn.neighbors import BallTree

            # Convert to radians for BallTree with haversine metric
            coords_rad = np.column_stack([
                np.radians(lats),
                np.radians(lons),
            ])

            self._spatial_index = BallTree(
                coords_rad,
                metric="haversine",
                leaf_size=leaf_size,
            )
            self._index_coords = np.column_stack([lats, lons])

        except ImportError:
            # Fall back to storing coordinates for brute force
            self._spatial_index = None
            self._index_coords = np.column_stack([lats, lons])

    def query_radius(
        self,
        query_lats: np.ndarray,
        query_lons: np.ndarray,
        radius_km: float,
    ) -> list[np.ndarray]:
        """Query all points within radius.

        Args:
            query_lats: Query latitudes
            query_lons: Query longitudes
            radius_km: Radius in kilometers

        Returns:
            List of index arrays for each query point
        """
        if self._index_coords is None:
            raise ValueError("Spatial index not built. Call build_spatial_index first.")

        query_rad = np.column_stack([
            np.radians(query_lats),
            np.radians(query_lons),
        ])

        # Convert radius to radians (Earth radius = 6371 km)
        radius_rad = radius_km / 6371.0

        if self._spatial_index is not None:
            indices = self._spatial_index.query_radius(query_rad, radius_rad)
            return [np.array(idx) for idx in indices]
        else:
            # Brute force fallback
            results = []
            geo = VectorizedGeoMixin()
            for qlat, qlon in zip(query_lats, query_lons):
                idx, _ = geo.points_within_radius(
                    qlat, qlon,
                    self._index_coords[:, 0],
                    self._index_coords[:, 1],
                    radius_km,
                    DistanceUnit.KILOMETERS,
                )
                results.append(idx)
            return results

    def query_nearest(
        self,
        query_lats: np.ndarray,
        query_lons: np.ndarray,
        k: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Query k nearest neighbors.

        Args:
            query_lats: Query latitudes
            query_lons: Query longitudes
            k: Number of neighbors

        Returns:
            Tuple of (distances_km, indices)
        """
        if self._index_coords is None:
            raise ValueError("Spatial index not built. Call build_spatial_index first.")

        query_rad = np.column_stack([
            np.radians(query_lats),
            np.radians(query_lons),
        ])

        if self._spatial_index is not None:
            distances_rad, indices = self._spatial_index.query(query_rad, k=k)
            # Convert from radians to km
            distances_km = distances_rad * 6371.0
            return distances_km, indices
        else:
            # Brute force fallback
            geo = VectorizedGeoMixin()
            all_indices = []
            all_distances = []

            for qlat, qlon in zip(query_lats, query_lons):
                idx, dist = geo.k_nearest_points(
                    qlat, qlon,
                    self._index_coords[:, 0],
                    self._index_coords[:, 1],
                    k,
                    DistanceUnit.KILOMETERS,
                )
                all_indices.append(idx)
                all_distances.append(dist)

            return np.array(all_distances), np.array(all_indices)

    def clear_spatial_index(self) -> None:
        """Clear spatial index to free memory."""
        self._spatial_index = None
        self._index_coords = None
