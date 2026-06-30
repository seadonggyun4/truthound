# Encryption 설정

실무 운영 가이드에서 Truthound을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 빠른 시작

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

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
from truthound.infrastructure.encryption import LocalKeyProvider

provider = LocalKeyProvider(
    key_file=".truthound_keys",
    master_password="${TRUTHOUND_MASTER_KEY}",
)

configure_encryption(provider=provider)
```

### VaultKeyProvider

HashiCorp Vault Transit 엔진.

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

| 실무 운영 가이드에서 Parameter을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Type을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Default을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|-----------|------|---------|-------------|
| 실무 운영 가이드에서 `url`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `str`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Required을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Vault, URL을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `token`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `None`, None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `VAULT_TOKEN`, Vault, VAULT_TOKEN을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `mount_point`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `str`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `"transit"`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Transit 엔진 mount point |
| 실무 운영 가이드에서 `cache_ttl`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `float`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `300.0`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Key 캐시 TTL in seconds |

### AwsKmsProvider

실무 운영 가이드에서 AWS, Key, Management, Service을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
from truthound.infrastructure.encryption import AwsKmsProvider

provider = AwsKmsProvider(
    key_id="alias/truthound-data-key",
    region="us-east-1",
)

configure_encryption(provider=provider)
```

실무 운영 가이드에서 AWS을(를) 다루는 항목입니다:
1. 실무 운영 가이드에서 `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, Environment, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
2. Shared credential 파일 (`~/.aws/credentials`)
3. 실무 운영 가이드에서 IAM, EC2/ECS/Lambda을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### GcpKmsProvider

실무 운영 가이드에서 Google, Cloud, KMS을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

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

| 실무 운영 가이드에서 Parameter을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Type을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Default을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|-----------|------|---------|-------------|
| 실무 운영 가이드에서 `key_name`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `str`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Required을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 KMS을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `project_id`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `None`, None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `GCP_PROJECT_ID`, GCP, GCP_PROJECT_ID을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `location`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `str`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `"global"`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 KMS을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `key_ring`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `str`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `"truthound"`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Key을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

실무 운영 가이드에서 Application, Default, Credentials을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### AzureKeyVaultProvider

실무 운영 가이드에서 Azure, Key, Vault을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
from truthound.infrastructure.encryption import AzureKeyVaultProvider

provider = AzureKeyVaultProvider(
    vault_url="https://my-vault.vault.azure.net",
    key_name="truthound-key",
)

configure_encryption(provider=provider)
```

실무 운영 가이드에서 DefaultAzureCredential을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

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

실무 운영 가이드에서 Encrypt을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

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

| 실무 운영 가이드에서 Option을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|--------|-------------|
| 실무 운영 가이드에서 `algorithm`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `aes_gcm`, `aes_cbc`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `format_preserving`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Preserve, SSN을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `deterministic`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Enable을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `mask_pattern`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Pattern을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

## 환경 변수

| 실무 운영 가이드에서 Variable을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|----------|-------------|
| 실무 운영 가이드에서 `TRUTHOUND_MASTER_KEY`, TRUTHOUND_MASTER_KEY을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Local을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `AWS_REGION`, AWS_REGION을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 AWS, KMS을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `GCP_PROJECT_ID`, GCP_PROJECT_ID을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 GCP을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `VAULT_TOKEN`, VAULT_TOKEN을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 HashiCorp, Vault을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `AZURE_TENANT_ID`, AZURE_TENANT_ID을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Azure을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 CLI, `AZURE_CLIENT_ID`, AZURE_CLIENT_ID을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Azure을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 CLI, `AZURE_CLIENT_SECRET`, AZURE_CLIENT_SECRET을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Azure client 시크릿 |

## 권장 방식

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

실무 운영 가이드에서 Cloud, KMS을(를) 다루는 항목입니다:

- 실무 운영 가이드에서 AWS, KMS, Enable을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 GCP, KMS, Configure을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Azure, Key, Vault, Set을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### 실무 운영 가이드 개요

```python
# Development
key_id = "alias/truthound-dev-key"

# Staging
key_id = "alias/truthound-staging-key"

# Production
key_id = "alias/truthound-prod-key"
```

### 실무 운영 가이드 개요

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

### 실무 운영 가이드 개요

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

## 보안 Considerations

1. 실무 운영 가이드에서 Never을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
2. 실무 운영 가이드에서 Dev을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
3. 실무 운영 가이드에서 Enable, Track을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
4. 실무 운영 가이드에서 Rotate, KMS을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
5. 실무 운영 가이드에서 Less을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
6. 실무 운영 가이드에서 Avoid, Enables을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
