"""Resource detection for distributed tracing.

Resources describe the entity producing telemetry, such as a service
running on a container. This module provides automatic detection of
common resource attributes.

Resource Semantic Conventions:
    - service.name: Name of the service
    - service.version: Version of the service
    - host.name: Hostname
    - process.pid: Process ID
    - telemetry.sdk.name: SDK name
    - telemetry.sdk.version: SDK version
"""

from __future__ import annotations

import os
import platform
import socket
import sys
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping


# =============================================================================
# Resource
# =============================================================================


@dataclass(frozen=True)
class Resource:
    """Immutable representation of the entity producing telemetry.

    Resources are used to describe the source of spans, metrics, and logs.
    They contain key-value attributes describing the entity.

    Example:
        >>> resource = Resource({
        ...     "service.name": "truthound",
        ...     "service.version": "1.0.0",
        ...     "deployment.environment": "production",
        ... })
    """

    attributes: Mapping[str, Any] = field(default_factory=dict)
    schema_url: str = ""

    def merge(self, other: "Resource") -> "Resource":
        """Merge with another resource.

        The other resource's attributes take precedence for conflicts.

        Args:
            other: Resource to merge with.

        Returns:
            New merged Resource.
        """
        merged_attrs = {**self.attributes, **other.attributes}
        return Resource(attributes=merged_attrs, schema_url=other.schema_url or self.schema_url)

    def get(self, key: str, default: Any = None) -> Any:
        """Get an attribute value.

        Args:
            key: Attribute key.
            default: Default value if not found.

        Returns:
            Attribute value or default.
        """
        return self.attributes.get(key, default)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "attributes": dict(self.attributes),
            "schema_url": self.schema_url,
        }

    @classmethod
    def empty(cls) -> "Resource":
        """Create an empty resource."""
        return cls(attributes={})

    @classmethod
    def create(cls, attributes: Mapping[str, Any] | None = None) -> "Resource":
        """Create a resource with given attributes.

        Args:
            attributes: Resource attributes.

        Returns:
            New Resource.
        """
        return cls(attributes=dict(attributes or {}))


# =============================================================================
# Resource Detector Interface
# =============================================================================


class ResourceDetector(ABC):
    """Abstract base class for resource detectors.

    Resource detectors automatically discover resource attributes
    from the environment.
    """

    @abstractmethod
    def detect(self) -> Resource:
        """Detect resource attributes.

        Returns:
            Detected Resource.
        """
        pass


# =============================================================================
# Process Resource Detector
# =============================================================================


class ProcessResourceDetector(ResourceDetector):
    """Detector for process-related resource attributes.

    Detects:
        - process.pid
        - process.executable.name
        - process.executable.path
        - process.command_line
        - process.runtime.name
        - process.runtime.version
        - process.runtime.description
    """

    def detect(self) -> Resource:
        """Detect process resource attributes."""
        attributes = {
            "process.pid": os.getpid(),
            "process.runtime.name": platform.python_implementation(),
            "process.runtime.version": platform.python_version(),
            "process.runtime.description": sys.version,
        }

        # Executable info
        if sys.executable:
            attributes["process.executable.path"] = sys.executable
            attributes["process.executable.name"] = os.path.basename(sys.executable)

        # Command line
        if sys.argv:
            attributes["process.command_line"] = " ".join(sys.argv)

        return Resource(attributes=attributes)


# =============================================================================
# Host Resource Detector
# =============================================================================


class HostResourceDetector(ResourceDetector):
    """Detector for host-related resource attributes.

    Detects:
        - host.name
        - host.arch
        - os.type
        - os.description
        - os.name
        - os.version
    """

    def detect(self) -> Resource:
        """Detect host resource attributes."""
        attributes = {}

        # Host info
        try:
            attributes["host.name"] = socket.gethostname()
        except Exception:
            pass

        try:
            attributes["host.arch"] = platform.machine()
        except Exception:
            pass

        # OS info
        try:
            attributes["os.type"] = platform.system().lower()
            attributes["os.description"] = platform.platform()
            attributes["os.name"] = platform.system()
            attributes["os.version"] = platform.release()
        except Exception:
            pass

        return Resource(attributes=attributes)


# =============================================================================
# Service Resource Detector
# =============================================================================


class ServiceResourceDetector(ResourceDetector):
    """Detector for service-related resource attributes.

    Detects from environment variables:
        - OTEL_SERVICE_NAME -> service.name
        - OTEL_SERVICE_VERSION -> service.version
        - OTEL_RESOURCE_ATTRIBUTES -> additional attributes

    Example:
        >>> os.environ["OTEL_SERVICE_NAME"] = "my-service"
        >>> detector = ServiceResourceDetector()
        >>> resource = detector.detect()
        >>> resource.get("service.name")
        'my-service'
    """

    def __init__(self, service_name: str | None = None) -> None:
        """Initialize detector.

        Args:
            service_name: Default service name if not in environment.
        """
        self._default_service_name = service_name or "unknown_service"

    def detect(self) -> Resource:
        """Detect service resource attributes."""
        attributes = {}

        # Service name
        service_name = os.environ.get(
            "OTEL_SERVICE_NAME",
            self._default_service_name,
        )
        attributes["service.name"] = service_name

        # Service version
        service_version = os.environ.get("OTEL_SERVICE_VERSION")
        if service_version:
            attributes["service.version"] = service_version

        # Parse OTEL_RESOURCE_ATTRIBUTES
        resource_attrs = os.environ.get("OTEL_RESOURCE_ATTRIBUTES", "")
        if resource_attrs:
            for pair in resource_attrs.split(","):
                if "=" in pair:
                    key, value = pair.split("=", 1)
                    attributes[key.strip()] = value.strip()

        return Resource(attributes=attributes)


# =============================================================================
# SDK Resource Detector
# =============================================================================


class SDKResourceDetector(ResourceDetector):
    """Detector for SDK/telemetry resource attributes.

    Detects:
        - telemetry.sdk.name
        - telemetry.sdk.version
        - telemetry.sdk.language
    """

    SDK_NAME = "truthound"
    SDK_VERSION = "1.0.0"
    SDK_LANGUAGE = "python"

    def detect(self) -> Resource:
        """Detect SDK resource attributes."""
        return Resource(attributes={
            "telemetry.sdk.name": self.SDK_NAME,
            "telemetry.sdk.version": self.SDK_VERSION,
            "telemetry.sdk.language": self.SDK_LANGUAGE,
        })


# =============================================================================
# Container Resource Detector
# =============================================================================


class ContainerResourceDetector(ResourceDetector):
    """Detector for container-related resource attributes.

    Detects from cgroup files and environment:
        - container.id
        - container.name
        - container.runtime

    Also detects Kubernetes attributes if available:
        - k8s.pod.name
        - k8s.namespace.name
        - k8s.deployment.name
    """

    def detect(self) -> Resource:
        """Detect container resource attributes."""
        attributes = {}

        # Try to detect container ID from cgroup
        container_id = self._detect_container_id()
        if container_id:
            attributes["container.id"] = container_id

        # Check for container runtime
        if os.path.exists("/.dockerenv"):
            attributes["container.runtime"] = "docker"
        elif os.environ.get("KUBERNETES_SERVICE_HOST"):
            attributes["container.runtime"] = "containerd"  # or cri-o

        # Kubernetes-specific
        pod_name = os.environ.get("HOSTNAME") or os.environ.get("K8S_POD_NAME")
        if pod_name and os.environ.get("KUBERNETES_SERVICE_HOST"):
            attributes["k8s.pod.name"] = pod_name

        namespace = os.environ.get("K8S_NAMESPACE")
        if namespace:
            attributes["k8s.namespace.name"] = namespace

        deployment = os.environ.get("K8S_DEPLOYMENT_NAME")
        if deployment:
            attributes["k8s.deployment.name"] = deployment

        return Resource(attributes=attributes)

    def _detect_container_id(self) -> str | None:
        """Detect container ID from cgroup file."""
        cgroup_paths = [
            "/proc/self/cgroup",
            "/proc/1/cgroup",
        ]

        for path in cgroup_paths:
            try:
                with open(path, "r") as f:
                    for line in f:
                        # Parse cgroup line: hierarchy-ID:controller-list:cgroup-path
                        parts = line.strip().split(":")
                        if len(parts) >= 3:
                            cgroup_path = parts[2]
                            # Extract container ID from path
                            # e.g., /docker/abc123... or /kubepods/burstable/pod.../container...
                            if "/docker/" in cgroup_path:
                                container_id = cgroup_path.split("/docker/")[-1][:64]
                                if len(container_id) >= 12:
                                    return container_id
                            elif "/cri-containerd-" in cgroup_path:
                                container_id = cgroup_path.split("/cri-containerd-")[-1].split(".")[0]
                                if container_id:
                                    return container_id
            except (IOError, OSError):
                continue

        return None


# =============================================================================
# Cloud Resource Detectors
# =============================================================================


class AWSResourceDetector(ResourceDetector):
    """Detector for AWS resource attributes.

    Detects from environment and metadata service:
        - cloud.provider
        - cloud.region
        - cloud.availability_zone
        - cloud.account.id
    """

    def detect(self) -> Resource:
        """Detect AWS resource attributes."""
        attributes = {}

        # Check if running on AWS
        region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
        if region:
            attributes["cloud.provider"] = "aws"
            attributes["cloud.region"] = region

        # Check for EC2 metadata (would need async/external call for actual metadata)
        # This is a simplified version that only checks environment

        return Resource(attributes=attributes)


class GCPResourceDetector(ResourceDetector):
    """Detector for GCP resource attributes.

    Detects from environment:
        - cloud.provider
        - cloud.region
        - cloud.project.id
    """

    def detect(self) -> Resource:
        """Detect GCP resource attributes."""
        attributes = {}

        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCP_PROJECT")
        if project_id:
            attributes["cloud.provider"] = "gcp"
            attributes["cloud.account.id"] = project_id

        region = os.environ.get("GOOGLE_CLOUD_REGION")
        if region:
            attributes["cloud.region"] = region

        return Resource(attributes=attributes)


class AzureResourceDetector(ResourceDetector):
    """Detector for Azure resource attributes.

    Detects from environment:
        - cloud.provider
        - cloud.region
    """

    def detect(self) -> Resource:
        """Detect Azure resource attributes."""
        attributes = {}

        # Check Azure-specific environment variables
        if os.environ.get("WEBSITE_SITE_NAME") or os.environ.get("FUNCTIONS_WORKER_RUNTIME"):
            attributes["cloud.provider"] = "azure"

        region = os.environ.get("REGION_NAME")
        if region:
            attributes["cloud.region"] = region

        return Resource(attributes=attributes)


# =============================================================================
# Resource Aggregation
# =============================================================================


_default_detectors: list[ResourceDetector] = [
    SDKResourceDetector(),
    ServiceResourceDetector(),
    ProcessResourceDetector(),
    HostResourceDetector(),
]

_resource_lock = threading.Lock()
_cached_resource: Resource | None = None


def get_aggregated_resources(
    detectors: list[ResourceDetector] | None = None,
    initial_resource: Resource | None = None,
) -> Resource:
    """Get aggregated resources from multiple detectors.

    Runs all detectors and merges their results. Later detectors
    override earlier ones for conflicting attributes.

    Args:
        detectors: List of detectors (default: standard detectors).
        initial_resource: Initial resource to merge with.

    Returns:
        Aggregated Resource.
    """
    detectors = detectors or _default_detectors
    resource = initial_resource or Resource.empty()

    for detector in detectors:
        try:
            detected = detector.detect()
            resource = resource.merge(detected)
        except Exception:
            pass  # Skip failed detectors

    return resource


def get_default_resource() -> Resource:
    """Get the default aggregated resource.

    This is cached for performance.

    Returns:
        Default Resource.
    """
    global _cached_resource

    with _resource_lock:
        if _cached_resource is None:
            _cached_resource = get_aggregated_resources()
        return _cached_resource


def clear_resource_cache() -> None:
    """Clear the cached resource."""
    global _cached_resource

    with _resource_lock:
        _cached_resource = None
