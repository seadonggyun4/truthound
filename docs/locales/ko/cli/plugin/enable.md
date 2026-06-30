# truthound plugins enable

CLI 명령 실행에서 Enable을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Synopsis

```bash
truthound plugins enable <NAME>
```

## Arguments

| CLI 명령 실행에서 Argument을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Required을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|----------|----------|-------------|
| CLI 명령 실행에서 `NAME`, NAME을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Yes을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Plugin을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

## Description

CLI 명령 실행에서 `plugins enable`을(를) 다루는 항목입니다:

1. CLI 명령 실행에서 Loads을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
2. CLI 명령 실행에서 Activates을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
3. **Makes** it available for 검증
4. CLI 명령 실행에서 Changes을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## State Transitions

```
loaded → active
inactive → active
```

## 예시

### Enable Plugin

```bash
truthound plugins enable my-validator
```

CLI 명령 실행에서 Output을(를) 다루는 항목입니다:
```
Enabled plugin: my-validator
```

### CLI 명령 실행 개요

```bash
# Load without activating
truthound plugins load my-validator --no-activate

# Enable when ready
truthound plugins enable my-validator
```

### Re-enable Disabled Plugin

```bash
# Plugin was disabled
truthound plugins enable my-validator
```

CLI 명령 실행에서 Output을(를) 다루는 항목입니다:
```
Enabled plugin: my-validator
```

### Verify After Enable

```bash
# Enable the plugin
truthound plugins enable my-validator

# Verify it's active
truthound plugins list --state active
```

## Use Cases

### 1. Activate After Inspection

```bash
# Load without activating
truthound plugins load my-validator --no-activate

# Inspect plugin
truthound plugins info my-validator

# Enable if satisfied
truthound plugins enable my-validator
```

### 2. Re-enable Temporarily Disabled Plugin

```bash
# Previously disabled for testing
truthound plugins enable my-validator

# Use in validation
truthound check data.csv --validators my-validator
```

### 3. Selective Plugin Activation

```bash
# Load multiple plugins
truthound plugins load validator-a --no-activate
truthound plugins load validator-b --no-activate
truthound plugins load validator-c --no-activate

# Enable only what's needed
truthound plugins enable validator-a
truthound plugins enable validator-c
```

## Error Handling

### Plugin Not Found

```bash
truthound plugins enable unknown-plugin
```

CLI 명령 실행에서 Output을(를) 다루는 항목입니다:
```
Error enabling plugin: Plugin 'unknown-plugin' not found.
```

> CLI 명령 실행에서 `enable`, Note을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Exit Codes

| CLI 명령 실행에서 Code을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Condition을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|------|-----------|
| CLI 명령 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Success을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| CLI 명령 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Error을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

## Related Commands

- CLI 명령 실행에서 `plugins disable`, Disable을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- CLI 명령 실행에서 `plugins load`, Load을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- CLI 명령 실행에서 `plugins list`, List을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 함께 보기

- [플러그인 명령 개요](index.md)
- CLI 명령 실행에서 Plugin, System을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
