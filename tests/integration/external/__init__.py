"""External Service Integration Tests for Truthound.

This package provides a comprehensive integration testing framework for
external services: Redis, Docker, Cloud KMS, TMS APIs, and more.

Architecture:
    ExternalServiceBackend (Abstract Base)
           |
           +---> ServiceProvider (Protocol)
           |         |
           |         +---> DockerProvider
           |         +---> LocalProvider
           |         +---> CloudProvider
           |         +---> MockProvider
           |
           +---> ServiceRegistry (Singleton)
           |
           +---> HealthChecker (Composite)

Design Principles:
    1. Provider-agnostic: Same tests run against Docker, local, cloud, or mocks
    2. Self-contained: Tests manage their own service lifecycle
    3. Parallel-safe: Multiple test suites can run concurrently
    4. Cost-aware: Cloud services have cost tracking and limits
    5. Observable: Comprehensive metrics and logging

Usage:
    # Run all external integration tests
    pytest tests/integration/external/ -v

    # Run with Docker services
    pytest tests/integration/external/ -v --provider=docker

    # Run with mock services (no external dependencies)
    pytest tests/integration/external/ -v --provider=mock

    # Run specific service tests
    pytest tests/integration/external/ -v -m redis
    pytest tests/integration/external/ -v -m kms
"""

__all__ = [
    # Base infrastructure
    "ExternalServiceBackend",
    "ServiceProvider",
    "ServiceRegistry",
    "ServiceConfig",
    "ServiceStatus",
    "ServiceCategory",
    # Providers
    "DockerServiceProvider",
    "LocalServiceProvider",
    "CloudServiceProvider",
    "MockServiceProvider",
    # Health checking
    "HealthChecker",
    "HealthCheckResult",
    # Test utilities
    "ExternalServiceTestCase",
    "require_service",
    "skip_without_service",
]
