# Realtime 체크포인트 Commands

Manage streaming 검증 체크포인트.

## 개요

CLI 명령 실행에서 Checkpoints을(를) 다루는 항목입니다:

- Resuming 검증 after a 실패
- 감사ing 검증 history
- Debugging streaming 파이프라인

## Commands

| CLI 명령 실행에서 Command을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|---------|-------------|
| CLI 명령 실행에서 `list`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | List available streaming 검증 체크포인트 |
| CLI 명령 실행에서 `show`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Show details of a specific 체크포인트 |
| CLI 명령 실행에서 `delete`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Delete a 체크포인트 |

## Quick 레퍼런스

```bash
# List all checkpoints
truthound realtime checkpoint list

# List checkpoints in a specific directory
truthound realtime checkpoint list --dir ./my_checkpoints

# Show checkpoint details
truthound realtime checkpoint show abc12345

# Delete a checkpoint
truthound realtime checkpoint delete abc12345

# Delete without confirmation
truthound realtime checkpoint delete abc12345 --force
```

## 체크포인트 Storage

CLI 명령 실행에서 JSON, `./checkpoints/`, `checkpoint_{id}.json`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Related Commands

- CLI 명령 실행에서 `realtime validate`, Validate을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- [`realtime monitor`](../monitor.md) - Monitor streaming 파이프라인
