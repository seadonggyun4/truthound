# Data Docs - HTML Report Generation (Phase 8)

Truthound의 Data Docs 모듈은 데이터 프로파일 결과를 아름답고 인터랙티브한 HTML 리포트로 변환합니다.

## 하위 문서

| 문서 | 설명 |
|------|------|
| [HTML Reports](html-reports.md) | 정적 HTML 리포트 생성 |
| [Themes](themes.md) | 6개 빌트인 테마 + 커스터마이징 |
| [Charts](charts.md) | ApexCharts, SVG 차트 렌더링 |
| [Sections](sections.md) | 9개 섹션 타입 구성 |
| [Versioning](versioning.md) | 리포트 버전 관리 (4개 전략) |
| [Custom Renderers](custom-renderers.md) | 커스텀 렌더러 개발 |
| [PDF Export](pdf-export.md) | PDF 내보내기 (WeasyPrint) |
| [Dashboard](dashboard.md) | Stage 2 인터랙티브 대시보드 |

---

## Overview

Data Docs는 두 단계로 구성됩니다:

| Stage | 기능 | 의존성 |
|-------|------|--------|
| **Stage 1: Static HTML Report** | Self-contained HTML 리포트 생성 | 없음 (CDN 기반) |
| **Stage 2: Interactive Dashboard** | Reflex 기반 인터랙티브 대시보드 | `truthound[dashboard]` |

### 주요 특징

- **Zero Dependencies**: npm/node 빌드 불필요, CDN에서 JS 로드
- **Self-Contained**: 단일 HTML 파일로 오프라인에서도 동작
- **6가지 빌트인 테마**: Default, Light, Dark, Professional, Minimal, Modern + Enterprise 커스터마이징
- **자동 차트 렌더링**: HTML은 ApexCharts, PDF는 SVG 자동 선택
- **반응형 디자인**: 모바일/태블릿/데스크톱 대응
- **인쇄 최적화**: Print-friendly CSS 포함
- **PDF 내보내기**: weasyprint 사용 (선택적)
- **다국어 지원**: 15개 언어 (en, ko, ja, zh, de, fr, es, pt, it, ru, ar, th, vi, id, tr)
- **리포트 버전관리**: 4개 전략 (Incremental, Semantic, Timestamp, GitLike)

---

## Installation

```bash
# 기본 설치 (Stage 1: HTML Report - Jinja2 필요)
pip install truthound[reports]

# PDF 내보내기 지원
pip install truthound[pdf]

# 대시보드 지원 (Stage 2)
pip install truthound[dashboard]

# 전체 설치
pip install truthound[all]
```

> **Note**: HTML 리포트 생성에는 Jinja2가 필요합니다. `truthound[reports]` 또는 `truthound[all]`을 설치하세요.

PDF 내보내기 시스템 의존성은 [PDF Export](pdf-export.md) 문서를 참조하세요.

---

## Quick Start

### 1. 프로파일 생성

```bash
truthound auto-profile data.csv -o profile.json
```

### 2. HTML 리포트 생성

```bash
truthound docs generate profile.json -o report.html
```

### 3. Python API

```python
from truthound.datadocs import generate_html_report

html = generate_html_report(
    profile=profile_dict,
    title="Data Quality Report",
    theme="professional",
    output_path="report.html",
)
```

자세한 내용은 [HTML Reports](html-reports.md) 문서를 참조하세요.

---

## CLI Commands

### `truthound docs generate`

```bash
truthound docs generate <profile_file> [OPTIONS]

# Options:
#   -o, --output TEXT    출력 파일 경로
#   -t, --title TEXT     리포트 제목
#   -s, --subtitle TEXT  부제목
#   --theme TEXT         테마 (light, dark, professional, minimal, modern)
#   -f, --format TEXT    출력 형식 (html, pdf)
```

### `truthound docs themes`

```bash
truthound docs themes  # 사용 가능한 테마 목록
```

### `truthound dashboard`

```bash
truthound dashboard --profile profile.json --port 8080
```

---

## Architecture

```
src/truthound/datadocs/
├── __init__.py          # 모듈 exports
├── base.py              # 타입, Enums, Protocols
├── builder.py           # HTMLReportBuilder, ProfileDataConverter
├── charts.py            # ApexChartsRenderer, SVGChartRenderer
├── sections.py          # 9개 섹션 렌더러
├── styles.py            # CSS 스타일시트
│
├── engine/              # 파이프라인 엔진
│   ├── context.py       # ReportContext, ReportData
│   ├── pipeline.py      # ReportPipeline
│   └── registry.py      # ComponentRegistry
│
├── themes/              # 테마 시스템
│   ├── base.py          # ThemeConfig, ThemeColors
│   ├── default.py       # 6개 빌트인 테마
│   ├── enterprise.py    # EnterpriseTheme
│   └── loader.py        # YAML/JSON 로더
│
├── renderers/           # 커스텀 렌더러
│   ├── base.py          # BaseRenderer
│   ├── jinja.py         # JinjaRenderer
│   └── custom.py        # String/File/Callable 렌더러
│
├── exporters/           # 출력 포맷
│   ├── base.py          # BaseExporter
│   ├── html_reporter.py # HtmlExporter
│   ├── pdf.py           # PdfExporter, OptimizedPdfExporter
│   └── markdown.py      # MarkdownExporter
│
├── versioning/          # 버전 관리
│   ├── version.py       # 4개 버전 전략
│   ├── storage.py       # InMemory, File 스토리지
│   └── diff.py          # ReportDiffer
│
├── i18n/                # 다국어 지원
│   ├── catalog.py       # 15개 언어
│   ├── plurals.py       # CLDR 복수형
│   └── formatting.py    # 숫자/날짜 포맷팅
│
└── dashboard/           # Stage 2: 대시보드
    ├── app.py           # DashboardApp
    ├── state.py         # Reflex state
    └── components.py    # UI 컴포넌트
```

### Class Hierarchy

```
BaseChartRenderer (ABC)
├── ApexChartsRenderer  # HTML reports (default)
└── SVGChartRenderer    # PDF export

BaseSectionRenderer (ABC)
├── OverviewSection
├── ColumnsSection
├── QualitySection
├── PatternsSection
├── DistributionSection
├── CorrelationsSection
├── RecommendationsSection
├── AlertsSection
└── CustomSection
```

---

## CI/CD Integration

### GitHub Actions

```yaml
name: Data Quality Report

on:
  schedule:
    - cron: '0 6 * * *'

jobs:
  report:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install truthound
      - run: truthound auto-profile data.csv -o profile.json
      - run: truthound docs generate profile.json -o report.html
      - uses: actions/upload-artifact@v4
        with:
          name: data-quality-report
          path: report.html
```

### GitLab CI

```yaml
data-quality-report:
  stage: report
  image: python:3.11
  script:
    - pip install truthound
    - truthound auto-profile data.csv -o profile.json
    - truthound docs generate profile.json -o report.html
  artifacts:
    paths:
      - report.html
```

---

## Troubleshooting

### Charts not rendering

CDN에서 JavaScript를 로드할 수 없는 환경에서는 PDF 포맷을 사용하세요:

```bash
truthound docs generate profile.json -o report.pdf --format pdf
```

### PDF export fails

시스템 라이브러리가 필요합니다. [PDF Export](pdf-export.md) 문서를 참조하세요.

### Dashboard import error

```bash
pip install truthound[dashboard]
```

---

## See Also

- [HTML Reports](html-reports.md) - 정적 HTML 리포트 생성
- [Themes](themes.md) - 테마 커스터마이징
- [Charts](charts.md) - 차트 렌더링
- [PDF Export](pdf-export.md) - PDF 내보내기
- [Dashboard](dashboard.md) - 인터랙티브 대시보드
