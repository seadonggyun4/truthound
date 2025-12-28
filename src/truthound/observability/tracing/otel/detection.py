"""OpenTelemetry availability detection.

This module provides utilities to detect whether the official OpenTelemetry
packages are installed and which features are available.

The detection is lazy and cached for performance.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import logging
import sys
import threading
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

logger = logging.getLogger(__name__)


class OTELPackage(str, Enum):
    """Known OpenTelemetry packages."""

    API = "opentelemetry-api"
    SDK = "opentelemetry-sdk"
    EXPORTER_OTLP = "opentelemetry-exporter-otlp"
    EXPORTER_OTLP_PROTO_GRPC = "opentelemetry-exporter-otlp-proto-grpc"
    EXPORTER_OTLP_PROTO_HTTP = "opentelemetry-exporter-otlp-proto-http"
    EXPORTER_JAEGER = "opentelemetry-exporter-jaeger"
    EXPORTER_ZIPKIN = "opentelemetry-exporter-zipkin"
    PROPAGATOR_B3 = "opentelemetry-propagator-b3"
    PROPAGATOR_JAEGER = "opentelemetry-propagator-jaeger"
    INSTRUMENTATION = "opentelemetry-instrumentation"


@dataclass
class PackageInfo:
    """Information about an installed package."""

    name: str
    version: str
    location: str = ""
    available: bool = True


@dataclass
class OTELAvailability:
    """OpenTelemetry availability status.

    Attributes:
        api_available: Whether opentelemetry-api is installed.
        sdk_available: Whether opentelemetry-sdk is installed.
        api_version: Version of opentelemetry-api if installed.
        sdk_version: Version of opentelemetry-sdk if installed.
        packages: Detailed info about installed OTEL packages.
        features: Available features based on installed packages.
    """

    api_available: bool = False
    sdk_available: bool = False
    api_version: str = ""
    sdk_version: str = ""
    packages: dict[str, PackageInfo] = field(default_factory=dict)
    features: set[str] = field(default_factory=set)

    @property
    def fully_available(self) -> bool:
        """Check if both API and SDK are available."""
        return self.api_available and self.sdk_available

    @property
    def can_export_otlp(self) -> bool:
        """Check if OTLP export is available."""
        return "otlp_export" in self.features

    @property
    def can_export_jaeger(self) -> bool:
        """Check if Jaeger export is available."""
        return "jaeger_export" in self.features

    @property
    def can_export_zipkin(self) -> bool:
        """Check if Zipkin export is available."""
        return "zipkin_export" in self.features

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "api_available": self.api_available,
            "sdk_available": self.sdk_available,
            "api_version": self.api_version,
            "sdk_version": self.sdk_version,
            "fully_available": self.fully_available,
            "packages": {
                k: {"name": v.name, "version": v.version, "available": v.available}
                for k, v in self.packages.items()
            },
            "features": list(self.features),
        }


# Cache for detection results
_detection_cache: OTELAvailability | None = None
_detection_lock = threading.Lock()


def _check_package(name: str) -> PackageInfo | None:
    """Check if a package is installed and get its info.

    Args:
        name: Package name (with dashes, e.g., 'opentelemetry-api')

    Returns:
        PackageInfo if installed, None otherwise.
    """
    # Convert package name to module name
    module_name = name.replace("-", ".")

    # Special mapping for known packages
    module_mapping = {
        "opentelemetry-api": "opentelemetry.trace",
        "opentelemetry-sdk": "opentelemetry.sdk.trace",
        "opentelemetry-exporter-otlp": "opentelemetry.exporter.otlp.proto.grpc.trace_exporter",
        "opentelemetry-exporter-otlp-proto-grpc": "opentelemetry.exporter.otlp.proto.grpc.trace_exporter",
        "opentelemetry-exporter-otlp-proto-http": "opentelemetry.exporter.otlp.proto.http.trace_exporter",
        "opentelemetry-exporter-jaeger": "opentelemetry.exporter.jaeger.thrift",
        "opentelemetry-exporter-zipkin": "opentelemetry.exporter.zipkin.json",
        "opentelemetry-propagator-b3": "opentelemetry.propagators.b3",
        "opentelemetry-propagator-jaeger": "opentelemetry.propagators.jaeger",
        "opentelemetry-instrumentation": "opentelemetry.instrumentation",
    }

    check_module = module_mapping.get(name, module_name)

    # Check if module exists
    spec = importlib.util.find_spec(check_module.split(".")[0])
    if spec is None:
        return None

    # Try to get version
    version = ""
    try:
        # Try standard package name first
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            # Try with underscores
            try:
                version = importlib.metadata.version(name.replace("-", "_"))
            except importlib.metadata.PackageNotFoundError:
                pass
    except Exception:
        pass

    # Get location if available
    location = ""
    if spec and spec.origin:
        location = spec.origin

    return PackageInfo(name=name, version=version, location=location)


def _determine_features(packages: dict[str, PackageInfo]) -> set[str]:
    """Determine available features from installed packages."""
    features = set()

    if OTELPackage.API.value in packages:
        features.add("trace_api")
        features.add("context_api")
        features.add("baggage")

    if OTELPackage.SDK.value in packages:
        features.add("span_processor")
        features.add("sampler")
        features.add("resource")
        features.add("span_limits")

    if OTELPackage.EXPORTER_OTLP.value in packages:
        features.add("otlp_export")
        features.add("otlp_grpc")

    if OTELPackage.EXPORTER_OTLP_PROTO_GRPC.value in packages:
        features.add("otlp_export")
        features.add("otlp_grpc")

    if OTELPackage.EXPORTER_OTLP_PROTO_HTTP.value in packages:
        features.add("otlp_export")
        features.add("otlp_http")

    if OTELPackage.EXPORTER_JAEGER.value in packages:
        features.add("jaeger_export")

    if OTELPackage.EXPORTER_ZIPKIN.value in packages:
        features.add("zipkin_export")

    if OTELPackage.PROPAGATOR_B3.value in packages:
        features.add("b3_propagation")

    if OTELPackage.PROPAGATOR_JAEGER.value in packages:
        features.add("jaeger_propagation")

    if OTELPackage.INSTRUMENTATION.value in packages:
        features.add("auto_instrumentation")

    return features


def detect_otel_availability(force_refresh: bool = False) -> OTELAvailability:
    """Detect OpenTelemetry availability.

    This function checks for installed OpenTelemetry packages and determines
    which features are available. Results are cached for performance.

    Args:
        force_refresh: Force re-detection even if cached.

    Returns:
        OTELAvailability with detected status.

    Example:
        >>> availability = detect_otel_availability()
        >>> if availability.fully_available:
        ...     print("Full OTEL support available")
        >>> elif availability.api_available:
        ...     print("OTEL API available, SDK missing")
        >>> else:
        ...     print("Using Truthound native tracing")
    """
    global _detection_cache

    with _detection_lock:
        if _detection_cache is not None and not force_refresh:
            return _detection_cache

        result = OTELAvailability()
        packages_to_check = [
            OTELPackage.API.value,
            OTELPackage.SDK.value,
            OTELPackage.EXPORTER_OTLP.value,
            OTELPackage.EXPORTER_OTLP_PROTO_GRPC.value,
            OTELPackage.EXPORTER_OTLP_PROTO_HTTP.value,
            OTELPackage.EXPORTER_JAEGER.value,
            OTELPackage.EXPORTER_ZIPKIN.value,
            OTELPackage.PROPAGATOR_B3.value,
            OTELPackage.PROPAGATOR_JAEGER.value,
            OTELPackage.INSTRUMENTATION.value,
        ]

        for pkg_name in packages_to_check:
            info = _check_package(pkg_name)
            if info:
                result.packages[pkg_name] = info

                if pkg_name == OTELPackage.API.value:
                    result.api_available = True
                    result.api_version = info.version

                if pkg_name == OTELPackage.SDK.value:
                    result.sdk_available = True
                    result.sdk_version = info.version

        result.features = _determine_features(result.packages)
        _detection_cache = result

        logger.debug(
            "OpenTelemetry detection complete: api=%s sdk=%s features=%s",
            result.api_available,
            result.sdk_available,
            result.features,
        )

        return result


def is_otel_sdk_available() -> bool:
    """Check if OpenTelemetry SDK is available.

    Convenience function for quick availability checks.

    Returns:
        True if opentelemetry-sdk is installed.
    """
    return detect_otel_availability().sdk_available


def is_otel_api_available() -> bool:
    """Check if OpenTelemetry API is available.

    Convenience function for quick availability checks.

    Returns:
        True if opentelemetry-api is installed.
    """
    return detect_otel_availability().api_available


def get_otel_version() -> str | None:
    """Get the installed OpenTelemetry API version.

    Returns:
        Version string or None if not installed.
    """
    availability = detect_otel_availability()
    return availability.api_version if availability.api_available else None


def get_installed_otel_packages() -> list[PackageInfo]:
    """Get list of installed OpenTelemetry packages.

    Returns:
        List of PackageInfo for installed OTEL packages.
    """
    availability = detect_otel_availability()
    return list(availability.packages.values())


def clear_detection_cache() -> None:
    """Clear the detection cache.

    Useful for testing or when packages are installed at runtime.
    """
    global _detection_cache

    with _detection_lock:
        _detection_cache = None
