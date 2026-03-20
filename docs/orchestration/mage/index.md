---
title: Mage
---

# Mage

Mage integration leans on project-root conventions rather than a heavy orchestration object model.

## Install

```bash
pip install truthound-orchestration[mage] "truthound>=3.0,<4.0"
```

## Zero-Config Path

```python
from truthound_mage import CheckTransformer, CheckBlockConfig

transformer = CheckTransformer(config=CheckBlockConfig(auto_schema=True))
result = transformer.execute(dataframe)
```

## `io_config.yaml` Discovery

- the loader searches the current directory and parent directories for `io_config.yaml` or `io_config.yml`
- the search is read-only
- if no project config is present, Mage still falls back to Truthound safe auto behavior
