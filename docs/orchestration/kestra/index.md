---
title: Kestra
---

# Kestra

Kestra integration is YAML-first. Python helpers exist to keep Truthound execution logic out of workflow files, but the intended user experience still feels native to Kestra.

## Install

```bash
pip install truthound-orchestration[kestra] "truthound>=3.0,<4.0"
```

## Zero-Config Path

```python
from truthound_kestra.scripts import check_quality_script

result = check_quality_script(data_uri="data/users.parquet")
```

## Flow Templates

Generated flows default to:

- Truthound as the engine
- shared preflight and runtime checks
- safe ephemeral execution unless the user explicitly opts into persistence
