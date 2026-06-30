# truthound plugins unload

CLI 명령 실행에서 Unload을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Synopsis

```bash
truthound plugins unload <NAME>
```

## Arguments

| CLI 명령 실행에서 Argument을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Required을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|----------|----------|-------------|
| CLI 명령 실행에서 `NAME`, NAME을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Yes을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Plugin을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

## Description

CLI 명령 실행에서 `plugins unload`을(를) 다루는 항목입니다:

1. CLI 명령 실행에서 Deactivates을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
2. CLI 명령 실행에서 Unloads을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
3. CLI 명령 실행에서 Returns을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## State Transitions

```
active → inactive → unloading → discovered
loaded → unloading → discovered
```

## 예시

### Unload Plugin

```bash
truthound plugins unload my-validator
```

CLI 명령 실행에서 Output을(를) 다루는 항목입니다:
```
Unloaded plugin: my-validator
```

### Unload Active Plugin

```bash
# Plugin is currently active
truthound plugins unload my-validator
```

CLI 명령 실행에서 Output을(를) 다루는 항목입니다:
```
Unloaded plugin: my-validator
```

### Verify After Unload

```bash
# Unload the plugin
truthound plugins unload my-validator

# Verify state changed to discovered
truthound plugins list
```

CLI 명령 실행에서 Output을(를) 다루는 항목입니다:
```
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Name            ┃ Version ┃ Type      ┃ State      ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ my-validator    │ 0.1.0   │ validator │ discovered │
└─────────────────┴─────────┴───────────┴────────────┘
```

## Use Cases

### 1. Cleanup After Testing

```bash
# Done testing plugin
truthound plugins unload my-validator
```

### 2. Reload Plugin (Update)

```bash
# Unload current version
truthound plugins unload my-validator

# Install update
pip install --upgrade truthound-plugin-my-validator

# Reload
truthound plugins load my-validator
```

### 3. 문제 해결

```bash
# Unload problematic plugin
truthound plugins unload buggy-plugin

# Check validation works without it
truthound check data.csv
```

## Error Handling

### Plugin Not Found or Error

```bash
truthound plugins unload unknown-plugin
```

CLI 명령 실행에서 Output을(를) 다루는 항목입니다:
```
Error unloading plugin: Plugin 'unknown-plugin' not found.
```

## Exit Codes

| CLI 명령 실행에서 Code을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Condition을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|------|-----------|
| CLI 명령 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Success을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| CLI 명령 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Error을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

## Related Commands

- CLI 명령 실행에서 `plugins load`, Load을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- CLI 명령 실행에서 `plugins disable`, Disable을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- CLI 명령 실행에서 `plugins list`, List을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 함께 보기

- [플러그인 명령 개요](index.md)
- CLI 명령 실행에서 Plugin, System을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
