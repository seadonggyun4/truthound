<div align="center">
  <img width="500" alt="Truthound Banner" src="docs/assets/truthound_banner.png?v=a4bd297" />
</div>

<h1 align="center">Truthound — Data Quality Workflow</h1>

<p align="center">
  <strong>Polars 기반 제로 설정 데이터 품질 프레임워크</strong> <br/>
  <strong>Zero-Configuration Data Quality Framework Powered by Polars</strong>
</p>

<p align="center">
  <em>Sniffs out bad data.</em>
</p>

<p align="center">
  <a href="https://truthound.netlify.app/"><img src="https://img.shields.io/badge/docs-truthound.netlify.app-blue" alt="Documentation"></a>
  <a href="https://pypi.org/project/truthound/"><img src="https://img.shields.io/pypi/v/truthound.svg" alt="PyPI"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11+-blue.svg" alt="Python"></a>
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-orange.svg" alt="License"></a>
  <a href="https://pola.rs/"><img src="https://img.shields.io/badge/Powered%20by-Polars-2563EB?logo=polars&logoColor=white" alt="Powered by Polars"></a>
  <a href="https://github.com/ddotta/awesome-polars"><img src="https://awesome.re/badge.svg" alt="Awesome Polars"></a>
  <a href="https://pepy.tech/project/truthound">
    <img src="https://img.shields.io/pepy/dt/truthound?color=brightgreen" alt="Downloads">
  </a>
</p>

<!--
README HEADER FORMAT LOCK:
The exact header format above MUST be preserved.
Do not rewrite it as Markdown-only syntax, do not remove the centered HTML,
do not change the banner image block, title, bilingual subtitle, slogan, or
badge block, and do not reorder those elements.
This required format starts at:
  <div align="center">
and ends at the closing </p> of the badge block above.
-->


---

## 개요 (Overview)

Truthound는 데이터 품질 검증(Data Validation)과 데이터 워크플로우(Data Workflow)를 위한 오픈소스 프레임워크입니다. Polars 기반의 검증 커널을 중심으로, 스키마 검증·사용자 정의 규칙·품질 검사·이상 데이터 탐지를 코드로 선언하고, 그 결과를 재현 가능하게 관리합니다.

Truthound is an open-source framework for data validation and data quality workflows, built on a Polars-first validation kernel.


**English Readme**: [English README](README.en.md) <br/>
**문서 (Documentation)**: [truthound.netlify.app](https://truthound.netlify.app/)

---

## 개발 목적 (Motivation)

데이터 품질 검증은 대부분 개별 프로젝트마다 서로 다른 방식으로 구현되어 재사용과 표준화가 어렵습니다. Truthound는 데이터 검증과 품질 관리 과정을 코드 기반 프레임워크와 워크플로우로 표준화하여, 누구나 일관되고 재현 가능한 데이터 검증 환경을 구축할 수 있도록 하는 것을 목표로 합니다.

---

## 프로젝트 소개 (Introduction)

Truthound는 Polars 기반의 오픈소스 데이터 검증 프레임워크와 워크플로우 오케스트레이션을 제공하는 프로젝트입니다. 데이터 스키마 검증, 사용자 정의 규칙, 품질 검사, 이상 데이터 탐지 등 다양한 검증 기능을 지원하며, Airflow, Dagster, Prefect 등 주요 워크플로우 환경과 연계하여 검증 절차를 자동화할 수 있습니다. 코드 중심의 선언적 검증 방식을 통해 데이터 품질 관리의 재현성과 유지보수성을 높이고, 다양한 데이터 파이프라인에서 쉽게 활용할 수 있도록 설계되었습니다.

Truthound는 두 개의 저장소로 구성된 하나의 오픈소스 생태계입니다.

| 구성 요소 | 저장소 | 역할 |
| --- | --- | --- |
| **Truthound** | [`truthound`](https://github.com/seadonggyun4/truthound) | 검증 커널 — `th.check()`, `ValidationRunResult`, 플래너/런타임, 제로 설정 워크스페이스, 리포터, 체크포인트 |
| **Truthound Orchestration** | [`truthound-orchestration`](https://github.com/seadonggyun4/truthound-orchestration) | Airflow, Dagster, Prefect, dbt, Mage, Kestra 등 워크플로우 환경 연동 레이어 |

---

## 기대 효과 (Impact)

데이터 검증 로직을 표준화하고 자동화함으로써 반복적인 품질 관리 비용을 줄이고, 데이터 파이프라인의 신뢰성과 운영 효율성을 높일 수 있습니다. 또한 다양한 오픈소스 생태계와 연계 가능한 구조를 제공하여 데이터 분석, ETL, AI/ML 등 여러 분야에서 재사용 가능한 데이터 품질 기반을 마련할 것으로 기대합니다.

---

## 주요 특징 (Key Features)

- **Polars 우선 실행**: 플래너 기반 메트릭 집계로 검증기마다 반복 스캔하지 않고 한 번에 계산합니다.
- **제로 설정 (Zero-Configuration)**: `th.check(data)` 한 줄로 로컬 `.truthound/` 워크스페이스를 자동 생성·재사용합니다.
- **결정론적 자동 검증 스위트**: 스키마·널 허용·타입·범위·키 휴리스틱을 기준으로 필요한 검사만 선택합니다 ("전부 실행"이 아님).
- **단일 결과 모델**: `ValidationRunResult` 하나를 체크포인트, 리포터, 검증 문서, 플러그인이 공유합니다.
- **명시적 계약(Contract)**: 컨텍스트, 체크 팩토리, 백엔드, 아티팩트 생성에 대한 안정적인 인터페이스를 제공합니다.
- **워크플로우 연계**: Truthound Orchestration을 통해 스케줄러/워크플로우 환경에서 호스트 네이티브로 실행됩니다.

> AI 검토 기능은 선택적 보조 레이어로 제공됩니다. Truthound의 핵심은 데이터 품질 워크플로우이며, AI는 프롬프트 기반 검증 제안·실행 분석 등을 사람이 검토하는 형태로 보조할 뿐입니다. AI 없이도 모든 핵심 기능이 동작합니다.

---

## 벤치마크 (Benchmark)

고정 러너 기반 릴리스 등급 벤치마크에서, 비교 가능한 모든 워크로드에 대해 정확성을 유지하면서 Great Expectations 대비 더 빠른 실행 시간과 더 낮은 메모리 사용을 측정했습니다.

| 워크로드 | Truthound Warm (s) | GX Warm (s) | 속도 향상 | 메모리 비율 |
| --- | --- | --- | --- | --- |
| local-mixed-core-suite | 0.028240 | 0.075232 | 2.66x | 44.29% |
| local-null | 0.016487 | 0.024964 | 1.51x | 43.62% |
| local-range | 0.002470 | 0.013219 | 5.35x | 43.84% |
| local-schema | 0.001479 | 0.017303 | 11.70x | 35.88% |
| local-unique | 0.002023 | 0.013785 | 6.81x | 42.28% |
| sqlite-null | 0.007370 | 0.032909 | 4.47x | 48.16% |
| sqlite-range | 0.006053 | 0.022355 | 3.69x | 43.80% |
| sqlite-unique | 0.002066 | 0.015655 | 7.58x | 42.12% |

이 비교는 결정론적 핵심 검사와 SQLite 푸시다운 워크로드로 범위를 한정한 결과이며, 모든 기능 영역에 대한 일반화된 주장이 아닙니다. 검증된 벤치마크 근거는 [Latest Verified Benchmark Summary](https://github.com/seadonggyun4/truthound/blob/main/docs/releases/latest-benchmark-summary.md)에서 확인할 수 있습니다.

성능 차이의 주요 원인:
- 검증기 루프로 재스캔하지 않고 메트릭 작업을 중복 제거하는 Polars 우선 플래너/런타임
- 기본 작업을 정확하고 관련성 있게 유지하는 결정론적 자동 스위트 선택
- 무거운 프로젝트 부트스트랩 없이 베이스라인과 아티팩트를 유지하는 가벼운 제로 설정 컨텍스트
- 리포터·체크포인트·검증 문서가 공유하는 단일 결과 계약

---

## 빠른 시작 (Quick Start)

### 설치 (Installation)

```bash
pip install truthound
```

```bash
# 선택적 AI 검토 기능
pip install truthound[ai]
```

```bash
# 이 저장소에서의 개발/문서 작업
uv sync --extra dev --extra docs
```

### Python API

```python
import truthound as th
from truthound.datadocs import generate_validation_report
from truthound.reporters import get_reporter
from truthound.drift import compare

run = th.check(
    {"customer_id": [1, 2, 2], "email": ["a@example.com", None, "c@example.com"]},
)

print(run.execution_mode)
print([check.name for check in run.checks])
print(run.metadata["context_root"])

json_report = get_reporter("json").render(run)
validation_docs = generate_validation_report(run, title="Customer Quality Overview")

context = th.get_context()
schema = th.learn({"id": [1, 2], "status": ["active", "inactive"]})
masked = th.mask(
    {"email": ["a@example.com", "b@example.com"]},
    columns=["email"],
    strategy="hash",
)
drift = compare({"score": [0.1, 0.2]}, {"score": [0.1, 0.8]})
```

### CLI

```bash
truthound check data.csv --validators null,unique
truthound check --connection "sqlite:///warehouse.db" --table users --pushdown
truthound scan pii.csv
truthound profile data.csv
truthound doctor . --workspace
truthound plugins list --json
```

```bash
# 선택적 AI 검토 워크플로우
truthound ai suggest-suite data.csv --prompt "Require customer_id to be unique"
truthound ai proposals list
truthound ai explain-run --run-id <run_id>
```

---

## 제로 설정 워크플로우 (Zero-Config Workflow)

Truthound는 프로젝트 루트에 `.truthound/` 워크스페이스를 자동으로 생성합니다. 기본적으로 다음을 관리합니다.

- `.truthound/config.yaml`: 해석된 프로젝트 기본값
- `.truthound/catalog/`: 자산 핑거프린트와 소스 시그니처
- `.truthound/baselines/`: 학습된 스키마와 메트릭 히스토리
- `.truthound/runs/`: 저장된 `ValidationRunResult` 메타데이터
- `.truthound/docs/`: 생성된 검증 문서
- `.truthound/plugins/`: 해석된 플러그인 매니페스트와 신뢰 메타데이터

`th.check(data)`만 호출해도 Truthound는 다음 순서로 동작합니다.

1. 자산/백엔드 감지
2. 활성 `TruthoundContext` 해석
3. 베이스라인 로드 또는 생성
4. 자동 검증 스위트 합성
5. 검증 계획 수립 및 실행
6. (활성화 시) 실행 결과와 검증 문서 저장

`truthound doctor . --workspace`로 로컬 `.truthound/` 레이아웃과 인덱스, 베이스라인, 저장된 실행 아티팩트의 구조적 무결성을 점검할 수 있습니다.

---

## 공개 API (Public Surface)

루트 패키지는 의도적으로 작은 API만 노출합니다.

- 안정 파사드: `check`, `scan`, `mask`, `profile`, `learn`, `read`, `get_context`
- 핵심 타입: `TruthoundContext`, `ValidationSuite`, `CheckSpec`, `SchemaSpec`, `ValidationRunResult`, `CheckResult`
- `th.check()`는 `ValidationRunResult`를 직접 반환합니다.
- 체크포인트 결과: `CheckpointResult.validation_run`이 정식이며, `CheckpointResult.validation_view`는 레거시 호환 프로젝션입니다.
- 리포터 타입: `truthound.reporters.RunPresentation`, `truthound.reporters.ReporterContext`
- 검증 문서 진입점: `truthound.datadocs.ValidationDocsBuilder`, `truthound.datadocs.generate_validation_report`
- 드리프트 비교: `truthound.drift.compare`
- 고급 시스템: 네임스페이스로 임포트 (예: `truthound.ml`, `truthound.lineage`, `truthound.realtime`, `truthound.datadocs`)
- 선택적 AI 레이어: `truthound[ai]` 설치 후 `truthound.ai` 임포트

---

## 플러그인 플랫폼 (Plugin Platform)

Truthound는 단일 라이프사이클 런타임을 사용합니다.

- `PluginManager`가 정식 플러그인 매니저입니다.
- `EnterprisePluginManager`는 동일 런타임 위의 비동기·역량 기반 파사드입니다.
- 플러그인은 `register_check_factory`, `register_data_asset_provider`, `register_reporter`, `register_hook`, `register_capability` 같은 안정 포트로 등록합니다.
- 리포터 플러그인은 `ValidationRunResult`를 정식 렌더 입력으로, `RunPresentation`을 공유 렌더 프로젝션으로 사용하는 contract-v3 표면을 대상으로 합니다.

---

## 문서 (Documentation)

- 메인 문서 포털: [truthound.netlify.app](https://truthound.netlify.app/)
- 개요: [docs/index.md](https://github.com/seadonggyun4/truthound/blob/main/docs/index.md)
- 시작하기: [docs/getting-started/index.md](https://github.com/seadonggyun4/truthound/blob/main/docs/getting-started/index.md)
- 아키텍처: [docs/concepts/architecture.md](https://github.com/seadonggyun4/truthound/blob/main/docs/concepts/architecture.md)
- 제로 설정 컨텍스트: [docs/concepts/zero-config.md](https://github.com/seadonggyun4/truthound/blob/main/docs/concepts/zero-config.md)
- 가이드: [docs/guides/index.md](https://github.com/seadonggyun4/truthound/blob/main/docs/guides/index.md)
- 레퍼런스: [docs/reference/index.md](https://github.com/seadonggyun4/truthound/blob/main/docs/reference/index.md)
- AI 문서: [docs/ai/index.md](https://github.com/seadonggyun4/truthound/blob/main/docs/ai/index.md)
- 오케스트레이션: [truthound.netlify.app/orchestration/](https://truthound.netlify.app/orchestration/)
- 오케스트레이션 시작하기: [docs/orchestration/getting-started.md](https://github.com/seadonggyun4/truthound/blob/main/docs/orchestration/getting-started.md)
- 최신 검증 벤치마크 요약: [docs/releases/latest-benchmark-summary.md](https://github.com/seadonggyun4/truthound/blob/main/docs/releases/latest-benchmark-summary.md)

---

## 개발 (Development)

```bash
uv run --frozen --extra dev python -m pytest -q
uv run --frozen --extra dev python -m pytest --collect-only -q tests
uv run --frozen --extra dev python -m pytest -q -m "contract or fault or e2e" -p no:cacheprovider
uv run --frozen --extra benchmarks python -m truthound.cli benchmark parity --suite pr-fast --frameworks truthound --backend local --strict
uv run --frozen --extra benchmarks python -m truthound.cli benchmark parity --suite nightly-core --frameworks both --backend local --strict
uv run --frozen --extra dev --extra docs mkdocs build --strict
```

테스트는 실패 우선(failure-first) 레인 모델을 따릅니다.

- `contract`: 안정 공개 API와 호환성 경계
- `fault`: 결정론적 실패 주입, 타임아웃, 손상, 동시성 시나리오
- `integration`: 옵트인 백엔드·외부 서비스 커버리지
- `soak` / `stress`: 야간 전용 부하·카오스 커버리지

공식 성능 수치는 `.truthound/benchmarks/release/` 아래의 검증된 릴리스 등급 패리티 아티팩트에서만 인용합니다. 야간 출력은 추세 확인용이며 공개 벤치마크 포지셔닝 용도가 아닙니다.

---

## 라이선스 (License)

Apache License 2.0. 자세한 내용은 [LICENSE](https://github.com/seadonggyun4/truthound/blob/main/LICENSE)를 참고하세요.
