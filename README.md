<div align="center">
  <img width="500" alt="Truthound Banner" src="docs/assets/truthound_banner.png?v=9d79fbef" />
</div>

<h1 align="center">Truthound — Data Quality Workflow</h1>

<!-- mcp-name: io.github.seadonggyun4/truthound -->

<p align="center">
  <strong>Polars 기반 제로 설정 데이터 품질 프레임워크</strong> <br/>
  <strong>Zero-Configuration Data Quality Framework Powered by Polars</strong>
</p>

<p align="center">
  <em>Sniffs out bad data.</em>
</p>

[![Documentation](https://img.shields.io/badge/docs-truthound.netlify.app-blue)](https://truthound.netlify.app/)
[![PyPI](https://img.shields.io/pypi/v/truthound.svg)](https://pypi.org/project/truthound/)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-orange.svg)](https://opensource.org/licenses/Apache-2.0)
[![Powered by Polars](https://img.shields.io/badge/Powered%20by-Polars-2563EB?logo=polars&logoColor=white)](https://pola.rs/)
[![Awesome Polars](https://awesome.re/badge.svg)](https://github.com/ddotta/awesome-polars)
[![Total Downloads](https://api.pepy.tech/personalized-badge/truthound?period=total&units=abbreviation&left_color=grey&right_color=brightgreen&left_text=downloads)](https://pepy.tech/project/truthound)



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

## 플러그인 플랫폼 (Plugin Platform)

### MCP로 사용하기

Codex와 Claude Desktop에서 Truthound Local을 연결해 승인한 로컬 데이터의 결측값·중복을 검사하고, 컬럼 요약과 PII 표본 탐지를 실행할 수 있습니다.

![Truthound Local 플러그인 화면 — 로컬 데이터 폴더 설정과 CSV 검사](docs/assets/truthound-local-plugin.png)

- **[Codex 설치 안내](docs/locales/ko/guides/mcp.md#openai-로컬-플러그인과-저장소-마켓플레이스)** — 준비 사항, 저장소 마켓플레이스 추가, 플러그인 설치와 허용 폴더 설정
- **[Claude Desktop 설치 안내](docs/locales/ko/guides/mcp.md#claude-desktop-설치-파일)** — MCPB 다운로드, 확장 설치와 허용 폴더 설정

설치 후에는 사용할 데이터 폴더를 별도로 승인해야 합니다. 다른 MCP 클라이언트의 설정은 [Registry·클라이언트 연결 안내](docs/locales/ko/guides/mcp.md#registry-메타데이터와-설치)를 참고하세요.

기본 검증·프로파일링·PII 탐지 외에 Core 프로파일에서 드리프트 비교·마스킹·스키마 검증을 사용할 수 있습니다. [핵심 사용 흐름과 프로파일 선택](docs/locales/ko/guides/mcp.md#핵심-데이터-품질-작업)을 참고하세요.

---

## 개요 (Overview)

Truthound는 데이터 품질 검증(Data Validation)과 데이터 워크플로우(Data Workflow)를 위한 오픈소스 프레임워크입니다. Polars 기반의 검증 커널을 중심으로, 스키마 검증·사용자 정의 규칙·품질 검사·이상 데이터 탐지를 코드로 선언하고, 그 결과를 재현 가능하게 관리합니다.

Truthound is an open-source framework for data validation and data quality workflows, built on a Polars-first validation kernel.

`th.scan()`은 문자열 컬럼별 첫 1,000개 비결측값을 검사하는 PII 표본 탐지입니다. 전체 프레임 적재 없이 집계·제한 표본을 사용하며 발견 건수는 추정치입니다. 완전한 Parquet footer 통계가 있으면 컬럼별 표본 확보 후 읽기를 종료하고, 통계가 없으면 배치 전체를 계수합니다. [PII API](docs/python-api/core-functions.md#thscan)를 참고하세요.


**English Readme**: [English README](README.en.md) <br/>
**문서 (Documentation)**: [truthound.netlify.app](https://truthound.netlify.app/)

---

표현식 검증은 통과한 검사를 생략해도 실패 샘플·빈도·행을 해당 컬럼에 연결합니다.

3.3.0의 `ai`·`mcp`·`all` extra는 Pydantic 2.11 이상을 요구합니다.
`truthound-mcp bundle --profile core --output truthound-core.mcpb`로 Core 번들을 만듭니다. 별도 `truthound-core` 플러그인 후보는 Core 3.3.0을 고정하며 데이터·작업공간·승인 저장소·민감 결과 권한을 명시적으로 설정하고 변경 후 재연결합니다.

## 프로젝트 소개 (Introduction)

Truthound는 Polars 기반의 오픈소스 데이터 검증 프레임워크와 워크플로우 오케스트레이션을 제공하는 프로젝트입니다. 데이터 스키마 검증, 사용자 정의 규칙, 품질 검사, 이상 데이터 탐지 등 다양한 검증 기능을 지원하며, Airflow, Dagster, Prefect 등 주요 워크플로우 환경과 연계하여 검증 절차를 자동화할 수 있습니다. 코드 중심의 선언적 검증 방식을 통해 데이터 품질 관리의 재현성과 유지보수성을 높이고, 다양한 데이터 파이프라인에서 쉽게 활용할 수 있도록 설계되었습니다.

Truthound는 두 개의 저장소로 구성된 하나의 오픈소스 생태계입니다.

| 구성 요소 | 저장소 | 역할 |
| --- | --- | --- |
| **Truthound** | [`truthound`](https://github.com/seadonggyun4/truthound) | 검증 커널 — `th.check()`, `ValidationRunResult`, 플래너/런타임, 제로 설정 워크스페이스, 리포터, 체크포인트 |
| **Truthound Orchestration** | [`truthound-orchestration`](https://github.com/seadonggyun4/truthound-orchestration) | Airflow, Dagster, Prefect, dbt, Mage, Kestra 등 워크플로우 환경 연동 레이어 |

---

## 주요 특징 (Key Features)

<!--
FACT-CHECK LOCK, 2026-07-01:
The default th.check() local runtime is Polars/LazyFrame based, but it executes
validators sequentially through ValidationRuntime._execute_sequential().
ExpressionBatchExecutor and SharedMetricStore provide batched/deduplicated
metric execution utilities, but ScanPlanner does not automatically fuse all
validator metrics into one collect() on the default path.
-->
- **Polars 우선 실행**: 로컬 검증은 Polars `LazyFrame`을 기준 경로로 사용합니다. 배치 표현식·공유 메트릭 유틸리티도 제공하지만, 기본 `th.check()` 경로가 모든 검증기 메트릭을 자동으로 단일 집계로 합치는 것은 아닙니다.
- **제로 설정 (Zero-Configuration)**: `th.check(data)` 한 줄로 로컬 `.truthound/` 워크스페이스를 자동 생성·재사용합니다.
- **결정론적 자동 검증 스위트**: 스키마·널 허용·타입·범위·키 휴리스틱을 기준으로 필요한 검사만 선택합니다 ("전부 실행"이 아님).
- **단일 결과 모델**: `ValidationRunResult` 하나를 체크포인트, 리포터, 검증 문서, 플러그인이 공유합니다.
- **명시적 계약(Contract)**: 컨텍스트, 체크 팩토리, 백엔드, 아티팩트 생성에 대한 안정적인 인터페이스를 제공합니다.
- **워크플로우 연계**: Truthound Orchestration을 통해 스케줄러/워크플로우 환경에서 호스트 네이티브로 실행됩니다.

> AI 검토 기능은 선택적 보조 레이어로 제공됩니다. Truthound의 핵심은 데이터 품질 워크플로우이며, AI는 프롬프트 기반 검증 제안·실행 분석 등을 사람이 검토하는 형태로 보조할 뿐입니다. AI 없이도 모든 핵심 기능이 동작합니다.

---

## 벤치마크 (Benchmark)

아래 표는 기존에 공개된 고정 러너 release artifact set의 역사적 측정 결과입니다. 당시 동일 조건의 8개 워크로드에서 정확성을 유지하면서 Great Expectations 대비 더 빠른 실행 시간과 더 낮은 메모리 사용을 측정했으며, 이후 버전이나 변경된 측정 방식의 성능 통과를 뜻하지 않습니다.

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

이 비교는 결정론적 핵심 검사와 SQLite 푸시다운 워크로드로 범위를 한정한 결과이며, 모든 기능 영역에 대한 일반화된 주장이 아닙니다. 저장소 안의 `.truthound/benchmarks/artifacts`에는 일부 원시 observation만 남아 있을 수 있으므로, 공식 수치는 release artifact set과 [Latest Verified Benchmark Summary](https://github.com/seadonggyun4/truthound/blob/main/docs/releases/latest-benchmark-summary.md)를 기준으로 해석해야 합니다.

새 벤치마크의 기본 측정 방식은 `truthound-parity-thread-budget-v2`입니다. 두 프레임워크에 동일한 worker thread budget `1`을 적용하고, cold 1회와 warm 7회의 정확성 및 모든 warm 측정값을 보존합니다. 기존 결과는 덮어쓰지 않으며, methodology·워크로드 계약·데이터·backend·exactness가 다른 결과를 직접 성능 비교하지 않습니다. 공개 wheel을 수정된 측정 도구로 재검증할 때도 wheel 버전·digest와 도구 revision을 별도로 기록합니다. 공식 릴리스 검증에는 전체 기본 methodology와 변경하지 않은 threshold가 필요하며, 사용자 지정 methodology는 참고용입니다. 새 방식의 성능 통과는 실제 검증 artifact가 있어야 인정됩니다. 자세한 기준은 [Benchmark Methodology](docs/guides/benchmark-methodology.md)를 참고하세요.

성능 차이의 주요 원인:
- Polars `LazyFrame` 기반 로컬 실행과 SQL 푸시다운 경로
- 일부 검증기와 고급 유틸리티에서 제공되는 배치 표현식·공유 메트릭 최적화
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
validation_docs = generate_validation_report(run, title="Customer Quality Overview", locale="ko")

context = th.get_context()
schema = th.learn({"id": [1, 2], "status": ["active", "inactive"]})
masked = th.mask(
    {"email": ["a@example.com", "b@example.com"]},
    columns=["email"],
    strategy="hash",
)
drift = compare({"score": [0.1, 0.2]}, {"score": [0.1, 0.8]})
```

- **보고서**: A4 HTML/PDF를 생성하며 공개 테마는 `light`·`dark`·`minimal`입니다. 번호가 붙은 표·그림·부록과 품질 차원·진단 기준·임계값을 제공합니다.
- **PDF 배치**: 긴 표·섹션은 페이지별로 나누고, 한 페이지에 들어가는 카드는 함께 배치합니다. 긴 컬럼명과 JSON 예시는 생략 없이 줄바꿈합니다.
- **자료형·통계**: 자료형은 `inferred_type` → `physical_type` → `dtype` 순으로 표시하며, 없으면 `unknown`입니다. 정의되지 않은 왜도·첨도는 JSON `null`로 표현하고 원본 값과 다른 통계는 유지합니다.
- **고급 연동**: `ReportDocument`로 구조를 검사하며 `generate_html_report`, `HTMLReportBuilder`, `export_to_pdf` 호환성을 유지합니다.

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

Lineage sample 메타데이터는 명시적 활성화 시에만 행·바이트 제한 안에서 복사하며 원본 자산을 자동으로 샘플링하지 않습니다.

고급 Arrow 변환에 보관된 인덱스·null·시간 정책과 행·바이트 제한을 적용합니다. materialization 한계는 분산 프로파일링 가이드에 설명되어 있습니다.

파일 저장소의 suite 쿼리는 선택한 정렬 뒤 페이지를 나누며, 모호한 JSON 기록은 인덱스 재구축에서도 거부합니다.


루트 패키지는 의도적으로 작은 API만 노출합니다.
공개 함수와 타입은 첫 접근 시 로드하며 기존 Core 객체와 동작을 유지합니다.

- 안정 파사드: `check`, `scan`, `mask`, `profile`, `learn`, `read`, `get_context`
- 핵심 타입: `TruthoundContext`, `ValidationSuite`, `CheckSpec`, `SchemaSpec`, `ValidationRunResult`, `CheckResult`
- `th.check()`는 `ValidationRunResult`를 직접 반환합니다.
  - 3.1.14부터 issue 상세가 없어도 명시적으로 실패한 check를 실패로 보존합니다. 설치된 버전을 확인해야 합니다.
- 체크포인트 결과: `CheckpointResult.validation_run`이 정식이며, `CheckpointResult.validation_view`는 레거시 호환 프로젝션입니다.
- 리포터 타입: `truthound.reporters.RunPresentation`, `truthound.reporters.ReporterContext`
- 검증 문서 진입점: `truthound.datadocs.ValidationDocsBuilder`, `truthound.datadocs.generate_validation_report`
  - 3.1.13부터 `locale="en"`(기본값) 또는 `locale="ko"`는 본문 제목·표 머리글·알림을 번역합니다. 품질 수치·상태 코드·사용자 데이터는 바꾸지 않으며 미지원 locale은 `ValueError`입니다. 이전 버전에는 이 옵션이 없습니다.
- 드리프트 비교: `truthound.drift.compare`; 명시한 0 임계값을 유지하며 잘못된 설정과 유한하지 않은 숫자 입력은 오류로 보고합니다.
- ERD 감사(개발 중): [`truthound.schema_audit`](docs/python-api/schema-audit.md)의 불변 모델·JSON·digest, 표준→설계 감사 A, 설계→카탈로그 감사 B와 제한된 PG14/17 읽기 전용 수집기. Console·접속 인가·승인 권한은 별도입니다.
- 고급 시스템: 네임스페이스로 임포트 (예: `truthound.ml`, `truthound.lineage`, `truthound.realtime`, `truthound.datadocs`) 네이티브 모델 생성자는 설정을 복사하고 알 수 없는 override를 거부합니다. 학습 표본에 시드를 적용하고 내장 탐지기의 재학습 실패 시 기존 모델을 보존합니다. 개념 드리프트는 상관관계 기본값과 실제 River DDM·ADWIN·Page-Hinkley 스트림 비교를 제공합니다([계약](docs/cli/ml/drift.md)). 계보 추적은 명시적 행 위치·컨텍스트 연결·재개 가능한 자동 저장·명시적으로 제공한 제한된 표본을 지원하며 OpenLineage 전송은 결과 불명 전환을 보존하고 종료 시 소유 요청을 회수합니다([계약](docs/guides/advanced/lineage.md)).
- 선택적 AI 레이어: `truthound[ai]` 설치 후 `truthound.ai` 임포트

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

## 라이선스 (License)

Apache License 2.0. 자세한 내용은 [LICENSE](https://github.com/seadonggyun4/truthound/blob/main/LICENSE)를 참고하세요.
