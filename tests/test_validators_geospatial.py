"""Tests for geospatial validators."""

import polars as pl
import pytest

from truthound.validators.geospatial import (
    GeoValidator,
    LatitudeValidator,
    LongitudeValidator,
    CoordinateValidator,
    CoordinateNotNullIslandValidator,
    GeoDistanceValidator,
    GeoDistanceFromPointValidator,
    GeoBoundingBoxValidator,
    GeoCountryValidator,
    EARTH_RADIUS_KM,
)


# =============================================================================
# Base GeoValidator Tests
# =============================================================================


class TestGeoValidator:
    """Tests for GeoValidator base class."""

    def test_haversine_distance_known_values(self):
        """Test Haversine distance calculation with known values."""
        # Seoul to Busan: approximately 325 km
        seoul_lat, seoul_lon = 37.5665, 126.9780
        busan_lat, busan_lon = 35.1796, 129.0756

        distance = GeoValidator.haversine_distance(
            seoul_lat, seoul_lon, busan_lat, busan_lon, unit="km"
        )
        assert 320 < distance < 330  # Approximately 325 km

    def test_haversine_distance_miles(self):
        """Test Haversine distance in miles."""
        # Same points, different unit
        distance_km = GeoValidator.haversine_distance(0, 0, 0, 1, unit="km")
        distance_miles = GeoValidator.haversine_distance(0, 0, 0, 1, unit="miles")
        assert distance_km > distance_miles  # km > miles

    def test_is_valid_latitude(self):
        """Test latitude validation."""
        assert GeoValidator.is_valid_latitude(0)
        assert GeoValidator.is_valid_latitude(45.5)
        assert GeoValidator.is_valid_latitude(-89.9)
        assert GeoValidator.is_valid_latitude(90)
        assert GeoValidator.is_valid_latitude(-90)
        assert not GeoValidator.is_valid_latitude(91)
        assert not GeoValidator.is_valid_latitude(-91)

    def test_is_valid_longitude(self):
        """Test longitude validation."""
        assert GeoValidator.is_valid_longitude(0)
        assert GeoValidator.is_valid_longitude(180)
        assert GeoValidator.is_valid_longitude(-180)
        assert GeoValidator.is_valid_longitude(126.9)
        assert not GeoValidator.is_valid_longitude(181)
        assert not GeoValidator.is_valid_longitude(-181)


# =============================================================================
# Coordinate Validators Tests
# =============================================================================


class TestLatitudeValidator:
    """Tests for LatitudeValidator."""

    def test_valid_latitudes(self):
        """Test valid latitudes pass."""
        df = pl.DataFrame({"lat": [37.5, -45.2, 0, 89.9, -89.9]})
        validator = LatitudeValidator(column="lat")
        issues = validator.validate(df.lazy())
        assert len(issues) == 0

    def test_invalid_latitudes(self):
        """Test invalid latitudes fail."""
        df = pl.DataFrame({"lat": [37.5, 95.0, -100.0, 45.0]})  # 95 and -100 are invalid
        validator = LatitudeValidator(column="lat")
        issues = validator.validate(df.lazy())
        assert len(issues) == 1
        assert issues[0].issue_type == "latitude_out_of_range"
        assert issues[0].count == 2

    def test_custom_latitude_range(self):
        """Test custom latitude range."""
        df = pl.DataFrame({"lat": [35.0, 40.0, 45.0, -5.0]})  # -5 is outside 0-90
        validator = LatitudeValidator(column="lat", min_lat=0, max_lat=90)
        issues = validator.validate(df.lazy())
        assert len(issues) == 1
        assert issues[0].count == 1

    def test_invalid_range_raises(self):
        """Test that min > max raises error."""
        with pytest.raises(ValueError, match="cannot be greater"):
            LatitudeValidator(column="lat", min_lat=50, max_lat=30)


class TestLongitudeValidator:
    """Tests for LongitudeValidator."""

    def test_valid_longitudes(self):
        """Test valid longitudes pass."""
        df = pl.DataFrame({"lon": [126.9, -74.0, 0, 179.9, -179.9]})
        validator = LongitudeValidator(column="lon")
        issues = validator.validate(df.lazy())
        assert len(issues) == 0

    def test_invalid_longitudes(self):
        """Test invalid longitudes fail."""
        df = pl.DataFrame({"lon": [126.9, 185.0, -200.0]})
        validator = LongitudeValidator(column="lon")
        issues = validator.validate(df.lazy())
        assert len(issues) == 1
        assert issues[0].issue_type == "longitude_out_of_range"
        assert issues[0].count == 2


class TestCoordinateValidator:
    """Tests for CoordinateValidator."""

    def test_valid_coordinates(self):
        """Test valid coordinate pairs pass."""
        df = pl.DataFrame({
            "lat": [37.5, 35.2, 40.7],
            "lon": [126.9, 129.0, -74.0],
        })
        validator = CoordinateValidator(lat_column="lat", lon_column="lon")
        issues = validator.validate(df.lazy())
        assert len(issues) == 0

    def test_invalid_coordinates(self):
        """Test invalid coordinates fail."""
        df = pl.DataFrame({
            "lat": [37.5, 95.0, 40.7],  # 95 is invalid
            "lon": [126.9, 129.0, 200.0],  # 200 is invalid
        })
        validator = CoordinateValidator(lat_column="lat", lon_column="lon")
        issues = validator.validate(df.lazy())
        assert len(issues) == 2  # One for lat, one for lon

    def test_korea_region(self):
        """Test coordinates in Korea region."""
        df = pl.DataFrame({
            "lat": [37.5, 35.2, 33.5, 10.0],  # 10.0 is outside Korea
            "lon": [126.9, 129.0, 126.5, 126.9],
        })
        validator = CoordinateValidator(
            lat_column="lat",
            lon_column="lon",
            min_lat=33.0,
            max_lat=43.0,
            min_lon=124.0,
            max_lon=132.0,
        )
        issues = validator.validate(df.lazy())
        assert len(issues) == 1
        assert issues[0].count == 1  # Only lat 10.0 is outside


class TestCoordinateNotNullIslandValidator:
    """Tests for CoordinateNotNullIslandValidator."""

    def test_no_null_island(self):
        """Test non-zero coordinates pass."""
        df = pl.DataFrame({
            "lat": [37.5, 35.2, 40.7],
            "lon": [126.9, 129.0, -74.0],
        })
        validator = CoordinateNotNullIslandValidator(lat_column="lat", lon_column="lon")
        issues = validator.validate(df.lazy())
        assert len(issues) == 0

    def test_null_island_detected(self):
        """Test (0, 0) coordinates are detected."""
        df = pl.DataFrame({
            "lat": [37.5, 0.0, 40.7],
            "lon": [126.9, 0.0, -74.0],
        })
        validator = CoordinateNotNullIslandValidator(lat_column="lat", lon_column="lon")
        issues = validator.validate(df.lazy())
        assert len(issues) == 1
        assert issues[0].issue_type == "null_island_coordinates"
        assert issues[0].count == 1

    def test_null_island_with_tolerance(self):
        """Test near-zero coordinates with tolerance."""
        df = pl.DataFrame({
            "lat": [37.5, 0.0001, 40.7],  # 0.0001 is near zero
            "lon": [126.9, 0.0001, -74.0],
        })
        validator = CoordinateNotNullIslandValidator(
            lat_column="lat", lon_column="lon", tolerance=0.001
        )
        issues = validator.validate(df.lazy())
        assert len(issues) == 1
        assert issues[0].count == 1


# =============================================================================
# Distance Validators Tests
# =============================================================================


class TestGeoDistanceValidator:
    """Tests for GeoDistanceValidator."""

    def test_distance_within_max(self):
        """Test distance within maximum passes."""
        df = pl.DataFrame({
            "start_lat": [37.5665],  # Seoul
            "start_lon": [126.9780],
            "end_lat": [35.1796],  # Busan
            "end_lon": [129.0756],
        })
        # Seoul to Busan is ~325km
        validator = GeoDistanceValidator(
            lat1_column="start_lat",
            lon1_column="start_lon",
            lat2_column="end_lat",
            lon2_column="end_lon",
            max_distance=400,  # 400km max
            unit="km",
        )
        issues = validator.validate(df.lazy())
        assert len(issues) == 0

    def test_distance_exceeds_max(self):
        """Test distance exceeding maximum fails."""
        df = pl.DataFrame({
            "start_lat": [37.5665],  # Seoul
            "start_lon": [126.9780],
            "end_lat": [35.1796],  # Busan
            "end_lon": [129.0756],
        })
        # Seoul to Busan is ~325km, set max to 200km
        validator = GeoDistanceValidator(
            lat1_column="start_lat",
            lon1_column="start_lon",
            lat2_column="end_lat",
            lon2_column="end_lon",
            max_distance=200,
            unit="km",
        )
        issues = validator.validate(df.lazy())
        assert len(issues) == 1
        assert issues[0].issue_type == "distance_above_maximum"

    def test_distance_below_min(self):
        """Test distance below minimum fails."""
        df = pl.DataFrame({
            "lat1": [37.5665, 37.5666],  # Nearly same location
            "lon1": [126.9780, 126.9781],
            "lat2": [37.5666, 37.5667],
            "lon2": [126.9781, 126.9782],
        })
        validator = GeoDistanceValidator(
            lat1_column="lat1",
            lon1_column="lon1",
            lat2_column="lat2",
            lon2_column="lon2",
            min_distance=1,  # At least 1km apart
            unit="km",
        )
        issues = validator.validate(df.lazy())
        assert len(issues) == 1
        assert issues[0].issue_type == "distance_below_minimum"


class TestGeoDistanceFromPointValidator:
    """Tests for GeoDistanceFromPointValidator."""

    def test_within_distance_from_point(self):
        """Test points within distance from reference pass."""
        df = pl.DataFrame({
            "lat": [37.5, 37.6, 37.4],  # Points near Seoul
            "lon": [126.9, 127.0, 126.8],
        })
        validator = GeoDistanceFromPointValidator(
            lat_column="lat",
            lon_column="lon",
            ref_lat=37.5665,  # Seoul
            ref_lon=126.9780,
            max_distance=50,  # 50km radius
            unit="km",
        )
        issues = validator.validate(df.lazy())
        assert len(issues) == 0

    def test_outside_distance_from_point(self):
        """Test points outside distance from reference fail."""
        df = pl.DataFrame({
            "lat": [37.5, 35.2],  # Seoul and Busan
            "lon": [126.9, 129.0],
        })
        validator = GeoDistanceFromPointValidator(
            lat_column="lat",
            lon_column="lon",
            ref_lat=37.5665,  # Seoul
            ref_lon=126.9780,
            max_distance=100,  # 100km radius (Busan is ~325km away)
            unit="km",
        )
        issues = validator.validate(df.lazy())
        assert len(issues) == 1
        assert issues[0].issue_type == "distance_from_point_above_maximum"


# =============================================================================
# Boundary Validators Tests
# =============================================================================


class TestGeoBoundingBoxValidator:
    """Tests for GeoBoundingBoxValidator."""

    def test_within_bounding_box(self):
        """Test points within bounding box pass."""
        df = pl.DataFrame({
            "lat": [37.5, 35.2, 38.0],
            "lon": [126.9, 129.0, 127.5],
        })
        validator = GeoBoundingBoxValidator(
            lat_column="lat",
            lon_column="lon",
            min_lat=33.0,
            max_lat=43.0,
            min_lon=124.0,
            max_lon=132.0,
        )
        issues = validator.validate(df.lazy())
        assert len(issues) == 0

    def test_outside_bounding_box(self):
        """Test points outside bounding box fail."""
        df = pl.DataFrame({
            "lat": [37.5, 10.0, 38.0],  # 10.0 is outside
            "lon": [126.9, 129.0, 127.5],
        })
        validator = GeoBoundingBoxValidator(
            lat_column="lat",
            lon_column="lon",
            min_lat=33.0,
            max_lat=43.0,
            min_lon=124.0,
            max_lon=132.0,
        )
        issues = validator.validate(df.lazy())
        assert len(issues) == 1
        assert issues[0].issue_type == "outside_bounding_box"

    def test_predefined_region_korea(self):
        """Test predefined Korea region."""
        df = pl.DataFrame({
            "lat": [37.5, 35.2],
            "lon": [126.9, 129.0],
        })
        validator = GeoBoundingBoxValidator(
            lat_column="lat",
            lon_column="lon",
            region="korea",
        )
        issues = validator.validate(df.lazy())
        assert len(issues) == 0

    def test_unknown_region_raises(self):
        """Test unknown region raises error."""
        with pytest.raises(ValueError, match="Unknown region"):
            GeoBoundingBoxValidator(
                lat_column="lat",
                lon_column="lon",
                region="unknown_country",
            )


class TestGeoCountryValidator:
    """Tests for GeoCountryValidator."""

    def test_within_country_korea(self):
        """Test points within Korea pass."""
        df = pl.DataFrame({
            "lat": [37.5665, 35.1796, 33.4996],  # Seoul, Busan, Jeju
            "lon": [126.9780, 129.0756, 126.5312],
        })
        validator = GeoCountryValidator(
            lat_column="lat",
            lon_column="lon",
            country="korea_south",
        )
        issues = validator.validate(df.lazy())
        assert len(issues) == 0

    def test_outside_country(self):
        """Test points outside country fail."""
        df = pl.DataFrame({
            "lat": [37.5665, 35.6762],  # Seoul, Tokyo
            "lon": [126.9780, 139.6503],
        })
        validator = GeoCountryValidator(
            lat_column="lat",
            lon_column="lon",
            country="korea_south",
        )
        issues = validator.validate(df.lazy())
        assert len(issues) == 1
        assert issues[0].issue_type == "outside_country_bounds"

    def test_multiple_countries(self):
        """Test validation with Japan."""
        df = pl.DataFrame({
            "lat": [35.6762, 34.6937, 43.0618],  # Tokyo, Osaka, Sapporo
            "lon": [139.6503, 135.5023, 141.3545],
        })
        validator = GeoCountryValidator(
            lat_column="lat",
            lon_column="lon",
            country="japan",
        )
        issues = validator.validate(df.lazy())
        assert len(issues) == 0

    def test_unknown_country_raises(self):
        """Test unknown country raises error."""
        with pytest.raises(ValueError, match="Unknown country"):
            GeoCountryValidator(
                lat_column="lat",
                lon_column="lon",
                country="atlantis",
            )
