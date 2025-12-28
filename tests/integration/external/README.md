# External Service Integration Tests

Enterprise-grade integration testing framework for Truthound's external service dependencies.

## Architecture

```
tests/integration/external/
├── base.py                 # Core abstractions (Backend, Provider, Registry)
├── conftest.py             # pytest fixtures and configuration
├── docker-compose.yml      # Docker service definitions
├── fluentd.conf            # Fluentd configuration
│
├── providers/              # Service provider implementations
│   ├── docker_provider.py  # Docker container management
│   ├── mock_provider.py    # In-memory mock services
│   └── registry.py         # Provider discovery and selection
│
├── backends/               # Service-specific backends
│   ├── redis_backend.py    # Redis cache/coordination
│   ├── elasticsearch_backend.py  # Search and logging
│   ├── vault_backend.py    # Secrets management
│   ├── kms_backend.py      # Cloud KMS (AWS/GCP/Azure/LocalStack)
│   └── tms_backend.py      # Translation Management Systems
│
└── tests/                  # Test suites
    ├── test_redis.py       # Redis integration tests
    ├── test_kms.py         # Cloud KMS tests
    ├── test_tms.py         # TMS API tests
    └── test_logging_sinks.py  # Elasticsearch/Loki/Fluentd tests
```

## Quick Start

### Run Mock Tests (No Dependencies)

```bash
# Run all mock tests (fast, no external services needed)
pytest tests/integration/external/tests/ -v -k "Mock or mock"

# Run specific mock tests
pytest tests/integration/external/tests/test_redis.py::TestMockRedis -v
pytest tests/integration/external/tests/test_kms.py::TestMockKMS -v
pytest tests/integration/external/tests/test_tms.py::TestMockTMS -v
```

### Run Docker Tests

```bash
# Start external services
docker-compose -f tests/integration/external/docker-compose.yml up -d

# Wait for services to be ready
sleep 30

# Run Docker-based tests
pytest tests/integration/external/tests/ -v --provider=docker

# Stop services
docker-compose -f tests/integration/external/docker-compose.yml down
```

### Run Specific Service Tests

```bash
# Redis tests only
pytest tests/integration/external/tests/ -v -m redis

# Elasticsearch tests only
pytest tests/integration/external/tests/ -v -m elasticsearch

# KMS tests only
pytest tests/integration/external/tests/ -v -m kms

# TMS tests only
pytest tests/integration/external/tests/ -v -m tms
```

## Provider Types

| Provider | Description | Use Case |
|----------|-------------|----------|
| **mock** | In-memory simulation | Fast unit/CI tests |
| **docker** | Containerized services | Local development |
| **local** | Locally installed services | Development environment |
| **cloud** | Cloud-managed services | Production validation |

## Supported Services

### Redis
- Basic key-value operations
- TTL and expiration
- Hash, List, Set operations
- Pub/Sub
- Distributed locking
- Cluster and Sentinel modes

### Elasticsearch
- Index management
- Document CRUD
- Search queries
- Bulk operations
- Log aggregation testing

### HashiCorp Vault
- KV secrets engine (v1 & v2)
- Transit secrets engine
- Token authentication
- AppRole authentication

### Cloud KMS
- AWS KMS (via LocalStack)
- GCP Cloud KMS
- Azure Key Vault
- HashiCorp Vault Transit

### Translation Management Systems
- Crowdin
- Lokalise
- Phrase
- Transifex
- POEditor

## Configuration

### Environment Variables

```bash
# Provider selection
TRUTHOUND_TEST_PROVIDER=docker|mock|local|cloud

# Redis
TRUTHOUND_TEST_REDIS_HOST=localhost
TRUTHOUND_TEST_REDIS_PORT=6379
TRUTHOUND_TEST_REDIS_PASSWORD=
TRUTHOUND_TEST_REDIS_SSL=false

# Elasticsearch
TRUTHOUND_TEST_ELASTICSEARCH_HOST=localhost
TRUTHOUND_TEST_ELASTICSEARCH_PORT=9200
TRUTHOUND_TEST_ELASTICSEARCH_USERNAME=
TRUTHOUND_TEST_ELASTICSEARCH_PASSWORD=

# Vault
TRUTHOUND_TEST_VAULT_HOST=localhost
TRUTHOUND_TEST_VAULT_PORT=8200
TRUTHOUND_TEST_VAULT_TOKEN=root-token

# KMS (LocalStack)
TRUTHOUND_TEST_KMS_HOST=localhost
TRUTHOUND_TEST_KMS_PORT=4566
TRUTHOUND_TEST_KMS_PROVIDER=localstack|aws|gcp|azure|mock

# TMS
TRUTHOUND_TEST_TMS_PROVIDER=mock|crowdin|lokalise
TRUTHOUND_TEST_TMS_API_KEY=your-api-key
TRUTHOUND_TEST_TMS_PROJECT_ID=your-project
```

## Design Principles

1. **Provider Agnostic**: Same tests run against Docker, local, cloud, or mocks
2. **Self-Contained**: Tests manage their own service lifecycle
3. **Parallel-Safe**: Multiple test suites can run concurrently
4. **Cost-Aware**: Cloud services have cost tracking and limits
5. **Observable**: Comprehensive metrics and logging

## Extending the Framework

### Adding a New Service Backend

```python
# backends/my_service_backend.py
from tests.integration.external.base import (
    ExternalServiceBackend,
    HealthCheckResult,
    ServiceCategory,
)

class MyServiceConfig(DockerContainerConfig):
    """Configuration for MyService."""

    def __post_init__(self):
        self.name = "myservice"
        self.category = ServiceCategory.CACHE
        self.image = "myservice/image"
        self.ports = {"1234/tcp": None}

class MyServiceBackend(ExternalServiceBackend[MyServiceConfig, MyClient]):
    """MyService test backend."""

    service_name = "myservice"
    service_category = ServiceCategory.CACHE
    default_port = 1234

    def _create_client(self) -> MyClient:
        return MyClient(host=self.host, port=self.port)

    def _close_client(self) -> None:
        self._client.close()

    def _perform_health_check(self) -> HealthCheckResult:
        try:
            self._client.ping()
            return HealthCheckResult.success("MyService healthy")
        except Exception as e:
            return HealthCheckResult.failure(str(e))
```

### Adding a Mock Service

```python
# providers/mock_provider.py
class MockMyService(MockService):
    """In-memory MyService mock."""

    def __init__(self):
        super().__init__(name="myservice", category=ServiceCategory.CACHE)
        self._data = {}

    def start(self) -> dict[str, Any]:
        result = super().start()
        self._data.clear()
        return {**result, "port": 1234}

    def get(self, key: str) -> Any:
        return self._data.get(key)

    def set(self, key: str, value: Any) -> bool:
        self._data[key] = value
        return True
```

## CI/CD Integration

```yaml
# GitHub Actions example
jobs:
  integration-tests:
    runs-on: ubuntu-latest
    services:
      redis:
        image: redis:7-alpine
        ports:
          - 6379:6379
      elasticsearch:
        image: elasticsearch:8.11.0
        env:
          discovery.type: single-node
          xpack.security.enabled: 'false'
        ports:
          - 9200:9200
    steps:
      - uses: actions/checkout@v4
      - name: Run integration tests
        run: |
          pytest tests/integration/external/ -v --provider=docker
        env:
          TRUTHOUND_TEST_REDIS_HOST: localhost
          TRUTHOUND_TEST_ELASTICSEARCH_HOST: localhost
```

## Metrics and Observability

Each backend collects operation metrics:

```python
backend.metrics.operation_count      # Total operations
backend.metrics.total_duration_seconds  # Total time
backend.metrics.error_count          # Failed operations
backend.metrics.bytes_transferred    # Data volume
backend.metrics.estimated_cost_usd   # Cloud costs (if applicable)
```

## Troubleshooting

### Docker Tests Failing

```bash
# Check Docker is running
docker info

# Check container status
docker-compose -f tests/integration/external/docker-compose.yml ps

# View container logs
docker-compose -f tests/integration/external/docker-compose.yml logs redis
```

### Health Check Timeouts

Increase timeout in config:

```python
config.timeout_seconds = 120
config.health_check_interval = 10.0
```

### Port Conflicts

Use dynamic port allocation:

```python
config.ports = {"6379/tcp": None}  # None = auto-allocate
```
