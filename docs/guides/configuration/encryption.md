# Encryption Configuration

Truthound provides at-rest encryption with support for multiple key management providers.

## Quick Start

```python
from truthound.infrastructure.encryption import (
    configure_encryption,
    get_encryptor,
)

# Configure with AWS KMS
configure_encryption(
    provider="aws_kms",
    key_id="alias/truthound-data-key",
    aws_region="us-east-1",
)

# Encrypt/decrypt data
encryptor = get_encryptor()
encrypted = encryptor.encrypt(b"sensitive data")
decrypted = encryptor.decrypt(encrypted)
```

## EnterpriseEncryptionConfig

```python
from truthound.infrastructure.encryption import EnterpriseEncryptionConfig

config = EnterpriseEncryptionConfig(
    enabled=True,
    provider="local",
    # Options: local, vault, aws_kms, gcp_kms, azure_keyvault

    key_id="",                       # Key identifier

    # HashiCorp Vault
    vault_url="",
    vault_token="",
    vault_mount_point="transit",

    # AWS KMS
    aws_region="",

    # GCP KMS
    gcp_project_id="",
    gcp_location="global",
    gcp_key_ring="truthound",

    # Azure Key Vault
    azure_vault_url="",

    # Local (development only)
    local_key_file=".truthound_keys",
    local_master_password="",        # or TRUTHOUND_MASTER_KEY env var

    # Field-level encryption policies
    field_policies={},
)
```

## Key Providers

### LocalKeyProvider

For development and testing only.

```python
from truthound.infrastructure.encryption import LocalKeyProvider

provider = LocalKeyProvider(
    key_file=".truthound_keys",
    master_password="${TRUTHOUND_MASTER_KEY}",
)

configure_encryption(provider=provider)
```

### VaultKeyProvider

HashiCorp Vault Transit engine.

```python
from truthound.infrastructure.encryption import VaultKeyProvider

provider = VaultKeyProvider(
    url="https://vault.example.com",
    token="${VAULT_TOKEN}",
    mount_point="transit",
    cache_ttl=300.0,  # Key cache TTL in seconds
)

configure_encryption(provider=provider)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `url` | `str` | Required | Vault server URL |
| `token` | `str \| None` | `None` | Vault token (or `VAULT_TOKEN` env var) |
| `mount_point` | `str` | `"transit"` | Transit engine mount point |
| `cache_ttl` | `float` | `300.0` | Key cache TTL in seconds |

### AwsKmsProvider

AWS Key Management Service.

```python
from truthound.infrastructure.encryption import AwsKmsProvider

provider = AwsKmsProvider(
    key_id="alias/truthound-data-key",
    region="us-east-1",
)

configure_encryption(provider=provider)
```

Uses default AWS credential chain:
1. Environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`)
2. Shared credential file (`~/.aws/credentials`)
3. IAM role (EC2/ECS/Lambda)

### GcpKmsProvider

Google Cloud KMS.

```python
from truthound.infrastructure.encryption import GcpKmsProvider

provider = GcpKmsProvider(
    key_name="data-key",
    project_id="my-project",
    location="us-central1",
    key_ring="truthound-keyring",
)

configure_encryption(provider=provider)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `key_name` | `str` | Required | KMS key name |
| `project_id` | `str \| None` | `None` | GCP project ID (or `GCP_PROJECT_ID` env var) |
| `location` | `str` | `"global"` | KMS location |
| `key_ring` | `str` | `"truthound"` | Key ring name |

Uses Application Default Credentials.

### AzureKeyVaultProvider

Azure Key Vault.

```python
from truthound.infrastructure.encryption import AzureKeyVaultProvider

provider = AzureKeyVaultProvider(
    vault_url="https://my-vault.vault.azure.net",
    key_name="truthound-key",
)

configure_encryption(provider=provider)
```

Uses DefaultAzureCredential for authentication.

## At-Rest Encryption

```python
from truthound.infrastructure.encryption import (
    AtRestEncryption,
    configure_encryption,
    get_encryptor,
)

# Local encryption (development)
encryptor = AtRestEncryption(
    algorithm="AES-256-GCM",
    key_file="/secure/encryption.key",
)

configure_encryption(encryptor)

# Encrypt sensitive data
encrypted = get_encryptor().encrypt(b"sensitive data")
decrypted = get_encryptor().decrypt(encrypted)
```

### With Cloud KMS

```python
from truthound.infrastructure.encryption import AtRestEncryption, AwsKmsProvider

provider = AwsKmsProvider(key_id="alias/my-key")
encryptor = AtRestEncryption(key_provider=provider)

configure_encryption(encryptor)
```

## Field-Level Encryption

Encrypt specific fields in data records.

```python
from truthound.infrastructure.encryption import (
    FieldLevelEncryption,
    FieldEncryptionPolicy,
    AwsKmsProvider,
)

provider = AwsKmsProvider("alias/my-key")

fle = FieldLevelEncryption(
    provider=provider,
    policies={
        "ssn": FieldEncryptionPolicy(
            format_preserving=True,  # Preserve format (XXX-XX-XXXX)
        ),
        "email": FieldEncryptionPolicy(
            algorithm="aes_gcm",
        ),
        "credit_card": FieldEncryptionPolicy(
            format_preserving=True,
            mask_pattern="****-****-****-{last4}",
        ),
    },
)

# Encrypt specific fields
encrypted_record = fle.encrypt_record({
    "name": "John Doe",
    "ssn": "123-45-6789",
    "email": "john@example.com",
})

# Decrypt fields
decrypted_record = fle.decrypt_record(encrypted_record)
```

### Field Encryption Policies

```python
from truthound.infrastructure.encryption import FieldEncryptionPolicy

policy = FieldEncryptionPolicy(
    algorithm="aes_gcm",           # Encryption algorithm
    format_preserving=False,       # Preserve original format
    deterministic=False,           # Same input = same output
    mask_pattern=None,             # Masking pattern for display
)
```

| Option | Description |
|--------|-------------|
| `algorithm` | `aes_gcm` (default), `aes_cbc` |
| `format_preserving` | Preserve field format (e.g., SSN format) |
| `deterministic` | Enable searching on encrypted fields |
| `mask_pattern` | Pattern for masked display |

## Environment Variables

| Variable | Description |
|----------|-------------|
| `TRUTHOUND_MASTER_KEY` | Local key provider password |
| `AWS_REGION` | AWS region for KMS |
| `GCP_PROJECT_ID` | GCP project ID |
| `VAULT_TOKEN` | HashiCorp Vault token |
| `AZURE_TENANT_ID` | Azure tenant ID |
| `AZURE_CLIENT_ID` | Azure client ID |
| `AZURE_CLIENT_SECRET` | Azure client secret |

## Best Practices

### 1. Use Cloud KMS in Production

```python
# Development
configure_encryption(
    provider="local",
    local_master_password="${TRUTHOUND_MASTER_KEY}",
)

# Production
configure_encryption(
    provider="aws_kms",
    key_id="alias/truthound-prod-key",
)
```

### 2. Rotate Keys Regularly

Cloud KMS providers support automatic key rotation:

- AWS KMS: Enable automatic rotation (yearly)
- GCP KMS: Configure rotation schedule
- Azure Key Vault: Set rotation policy

### 3. Use Separate Keys per Environment

```python
# Development
key_id = "alias/truthound-dev-key"

# Staging
key_id = "alias/truthound-staging-key"

# Production
key_id = "alias/truthound-prod-key"
```

### 4. Enable Field-Level Encryption for PII

```python
fle = FieldLevelEncryption(
    provider=provider,
    policies={
        "ssn": FieldEncryptionPolicy(format_preserving=True),
        "email": FieldEncryptionPolicy(),
        "phone": FieldEncryptionPolicy(),
        "credit_card": FieldEncryptionPolicy(format_preserving=True),
        "address": FieldEncryptionPolicy(),
    },
)
```

### 5. Use Deterministic Encryption for Searchable Fields

```python
# Enable searching on encrypted email
policies = {
    "email": FieldEncryptionPolicy(
        deterministic=True,  # Same email always encrypts to same value
    ),
}

# Now you can search:
# SELECT * FROM users WHERE encrypted_email = encrypt('user@example.com')
```

## Security Considerations

1. **Never store keys in code** - Use environment variables or secret managers
2. **Use separate keys per environment** - Dev, staging, production
3. **Enable audit logging** - Track all encryption/decryption operations
4. **Rotate keys regularly** - Use cloud KMS automatic rotation
5. **Use format-preserving encryption carefully** - Less secure than standard encryption
6. **Avoid deterministic encryption unless necessary** - Enables frequency analysis attacks
