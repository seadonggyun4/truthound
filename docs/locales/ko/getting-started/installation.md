# 설치

초기 설정과 첫 실행에서 Truthound, CLI, Install, Add을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Prerequisites

- 초기 설정과 첫 실행에서 `3.11`, Python을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 초기 설정과 첫 실행에서 `pip`, `uv`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 초기 설정과 첫 실행에서 Polars, Polars-based을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Recommended Install Matrix

```bash
pip install truthound
```

초기 설정과 첫 실행에서 관련 설정과 실행 흐름을(를) 다루는 항목입니다:

- 초기 설정과 첫 실행에서 CLI을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- `th.check()`, `th.profile()`, `th.mask()`, `th.scan()`, and core 결과 types
- zero-설정 local 검증 with `.truthound/`

초기 설정과 첫 실행에서 Common을(를) 다루는 항목입니다:

| 초기 설정과 첫 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 초기 설정과 첫 실행에서 Command을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|----------|---------|
| HTML 리포트 generation | 초기 설정과 첫 실행에서 `pip install truthound[reports]`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 드리프트 detection | 초기 설정과 첫 실행에서 `pip install truthound[drift]`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| ML 이상치 detection | 초기 설정과 첫 실행에서 `pip install truthound[anomaly]`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 초기 설정과 첫 실행에서 Storage을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 초기 설정과 첫 실행에서 `pip install truthound[stores]`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Streaming 검증 | 초기 설정과 첫 실행에서 `pip install truthound[streaming]`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 초기 설정과 첫 실행에서 Contributor을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 초기 설정과 첫 실행에서 `pip install truthound[dev,docs]`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 초기 설정과 첫 실행에서 Broad을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 초기 설정과 첫 실행에서 `pip install truthound[all]`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

## Optional Extras

```bash
pip install truthound[reports]
pip install truthound[drift]
pip install truthound[anomaly]
pip install truthound[stores]
pip install truthound[streaming]
pip install truthound[docs]
pip install truthound[dev]
```

## Verify The 설치

```bash
truthound --version
python -c "import truthound as th; print(th.__version__)"
```

초기 설정과 첫 실행에서 Quick을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Development Setup

```bash
uv sync --extra dev --extra docs
uv run python -m pytest -q
uv run python docs/scripts/check_links.py --mkdocs mkdocs.yml README.md CLAUDE.md
uv run mkdocs build --strict
```

## 문제 해결

### Import or optional dependency errors

초기 설정과 첫 실행에서 `jinja2`, `scipy`, `pyarrow`, `all`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### 초기 설정과 첫 실행 개요

초기 설정과 첫 실행에서 CLI을(를) 다루는 항목입니다:

- 리포트 generation needs `truthound[reports]`
- 초기 설정과 첫 실행에서 `truthound[streaming]`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 초기 설정과 첫 실행에서 `truthound[drift]`, `truthound[anomaly]`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### 초기 설정과 첫 실행 개요

초기 설정과 첫 실행에서 `truthound`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Next Step

초기 설정과 첫 실행에서 Continue, Quick, First, Validation을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
