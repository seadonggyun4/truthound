---
title: Mage Variables, Secrets, and Profiles
---

# Mage Variables, Secrets, and Profiles

Truthound's Mage integration expects project configuration to flow through the same places Mage already uses: `io_config.yaml`, environment-backed values, and profile selection. The adapter should not become a second secret-management system.

## Who This Is For

- Mage operators managing environment-specific profiles
- teams moving from local project config to production secrets
- engineers deciding where Truthound runtime values should live

## When To Use It

Use this page when:

- `io_config.yaml` needs environment-specific behavior
- credentials should move out of local config and into a secret source
- the same pipeline runs in dev, staging, and production

## Prerequisites

- a Mage project using `io_config.yaml`
- access to the environment or secret source used by the project
- understanding of which sources require credentials

## Minimal Quickstart

Load Mage config explicitly:

```python
from truthound_mage import load_io_config

config = load_io_config("io_config.yaml", profile="production")
```

Pair that config with a quality block:

```python
from truthound_mage import CheckTransformer, CheckBlockConfig

transformer = CheckTransformer(
    config=CheckBlockConfig(
        rules=[{"column": "email", "check": "email_format"}]
    )
)
```

## Production Pattern

Use this split of responsibility:

| Concern | Recommended Mage Boundary |
|--------|----------------------------|
| source and sink connection config | `io_config.yaml` plus profile selection |
| secret values | environment-backed or external secret management feeding Mage |
| validation rules | block config in source control |
| runtime observability | shared runtime plus Mage block metadata |

Recommended checklist:

- keep secrets out of committed config files
- name profiles consistently across environments
- document which profiles are allowed to perform production validation

## Failure Modes and Troubleshooting

| Symptom | Likely Cause | What To Do |
|--------|--------------|------------|
| production pipeline uses local credentials | profile selection is missing or incorrect | load the correct profile explicitly |
| config works locally but not in deployed Mage | environment-backed secrets are unavailable | align runtime env and secret provisioning |
| validation rules drift by environment | rules are embedded in environment config | keep rule intent in versioned block config |

## Related Pages

- [Mage Overview](index.md)
- [`io_config.yaml` Discovery](io-config.md)
- [Enterprise Secrets](../enterprise/secrets.md)
