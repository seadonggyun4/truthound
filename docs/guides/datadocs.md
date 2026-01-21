# Data Docs - HTML Report Generation (Phase 8)

Truthound의 Data Docs 모듈은 데이터 프로파일 결과를 아름답고 인터랙티브한 HTML 리포트로 변환합니다. CI/CD 파이프라인에서 아티팩트로 저장하거나, 이메일/Slack으로 공유할 수 있는 self-contained HTML 파일을 생성합니다.

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [CLI Commands](#cli-commands)
5. [Python API](#python-api)
6. [Themes](#themes)
7. [Chart Libraries](#chart-libraries)
8. [Report Sections](#report-sections)
9. [Customization](#customization)
10. [Dashboard (Stage 2)](#dashboard-stage-2)
11. [CI/CD Integration](#cicd-integration)
12. [Architecture](#architecture)

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
- **6가지 빌트인 테마**: Default, Light, Dark, Professional, Minimal, Modern (+ Enterprise)
- **자동 차트 렌더링**: HTML은 ApexCharts, PDF는 SVG 자동 선택
- **반응형 디자인**: 모바일/태블릿/데스크톱 대응
- **인쇄 최적화**: Print-friendly CSS 포함
- **PDF 내보내기**: weasyprint 사용 (선택적)
- **다국어 지원**: 영어(en), 한국어(ko)
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

!!! warning "HTML Report Dependency"
    HTML 리포트 생성에는 Jinja2가 필요합니다. 반드시 `truthound[reports]` 또는 `truthound[all]`을 설치하세요.

### PDF Export System Dependencies

PDF 내보내기는 WeasyPrint를 사용하며, **시스템 라이브러리**가 필요합니다.
Python 패키지만으로는 동작하지 않으므로 반드시 아래 시스템 의존성을 먼저 설치하세요.

#### macOS (Homebrew)

```bash
brew install pango cairo gdk-pixbuf libffi
pip install truthound[pdf]
```

#### Ubuntu/Debian

```bash
sudo apt-get install libpango-1.0-0 libpangocairo-1.0-0 \
  libgdk-pixbuf2.0-0 libffi-dev shared-mime-info
pip install truthound[pdf]
```

#### Fedora/RHEL

```bash
sudo dnf install pango gdk-pixbuf2 libffi-devel
pip install truthound[pdf]
```

#### Alpine Linux

```bash
apk add pango gdk-pixbuf libffi-dev
pip install truthound[pdf]
```

#### Windows

Windows에서는 GTK3 런타임이 필요합니다:

1. [GTK3 for Windows](https://github.com/nickvidal/weasyprint/releases) 다운로드
2. 압축 해제 후 PATH에 추가
3. `pip install truthound[pdf]`

또는 bundled 버전 사용:
```bash
pip install weasyprint[gtk3]
```

#### Docker

Docker에서 사용할 경우:

```dockerfile
# Debian/Ubuntu 기반
FROM python:3.11-slim
RUN apt-get update && apt-get install -y \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libgdk-pixbuf2.0-0 \
    libffi-dev \
    shared-mime-info \
    && rm -rf /var/lib/apt/lists/*
RUN pip install truthound[pdf]
```

```dockerfile
# Alpine 기반
FROM python:3.11-alpine
RUN apk add --no-cache pango gdk-pixbuf libffi-dev
RUN pip install truthound[pdf]
```

> **Note**: `pip install truthound[pdf]`는 Python 패키지(weasyprint)만 설치합니다.
> 위의 시스템 라이브러리가 없으면 PDF 생성 시 `cannot load library 'libpango-1.0-0'` 에러가 발생합니다.

자세한 내용은 [WeasyPrint 설치 가이드](https://doc.courtbouillon.org/weasyprint/stable/first_steps.html#installation)를 참조하세요.

---

## Quick Start

### 1. 프로파일 생성

먼저 데이터 프로파일을 생성합니다:

```bash
truthound auto-profile data.csv -o profile.json
```

### 2. HTML 리포트 생성

```bash
truthound docs generate profile.json -o report.html
```

### 3. 브라우저에서 열기

생성된 `report.html` 파일을 브라우저에서 열면 완전한 데이터 품질 리포트를 확인할 수 있습니다.

---

## CLI Commands

### `truthound docs generate`

프로파일 JSON 파일에서 HTML 리포트를 생성합니다.

```bash
truthound docs generate <profile_file> [OPTIONS]
```

**Options:**

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--output` | `-o` | `<input>.html` | 출력 파일 경로 |
| `--title` | `-t` | "Data Profile Report" | 리포트 제목 |
| `--subtitle` | `-s` | "" | 리포트 부제목 |
| `--theme` | | "professional" | 테마 (light, dark, professional, minimal, modern) |
| `--format` | `-f` | "html" | 출력 형식 (html, pdf) |

**Examples:**

```bash
# 기본 사용
truthound docs generate profile.json -o report.html

# 커스텀 제목과 다크 테마
truthound docs generate profile.json -o report.html \
    --title "Q4 Data Quality Report" \
    --subtitle "Customer Dataset" \
    --theme dark

# PDF 내보내기 (requires weasyprint)
truthound docs generate profile.json -o report.pdf --format pdf
```

### `truthound docs themes`

사용 가능한 테마 목록을 표시합니다.

```bash
truthound docs themes
```

**Output:**

```
Available report themes:

  light          - Clean and bright, suitable for most use cases
  dark           - Dark mode with vibrant colors, easy on the eyes
  professional   - Corporate style, subdued colors (default)
  minimal        - Minimalist design with monochrome accents
  modern         - Contemporary design with vibrant gradients
```

### `truthound dashboard`

인터랙티브 대시보드를 실행합니다 (Stage 2).

```bash
truthound dashboard [OPTIONS]
```

**Options:**

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--profile` | `-p` | None | 프로파일 JSON 파일 경로 |
| `--port` | | 8080 | 서버 포트 |
| `--host` | | "localhost" | 서버 호스트 |
| `--title` | `-t` | "Truthound Dashboard" | 대시보드 제목 |
| `--debug` | | False | 디버그 모드 활성화 |

**Examples:**

```bash
# 프로파일과 함께 실행
truthound dashboard --profile profile.json

# 커스텀 포트
truthound dashboard --profile profile.json --port 3000

# 외부 접근 허용
truthound dashboard --profile profile.json --host 0.0.0.0
```

---

## Python API

### HTMLReportBuilder

세밀한 제어가 필요할 때 `HTMLReportBuilder`를 직접 사용합니다.

```python
from truthound.datadocs import (
    HTMLReportBuilder,
    ReportTheme,
    ChartLibrary,
    ReportConfig,
)

# 기본 사용
builder = HTMLReportBuilder(theme=ReportTheme.PROFESSIONAL)
html = builder.build(profile, title="My Data Report")
builder.save(html, "report.html")

# 커스텀 설정
config = ReportConfig(
    theme=ReportTheme.DARK,
    include_toc=True,
    include_timestamp=True,
    footer_text="Generated by Data Team",
)
builder = HTMLReportBuilder(config=config)
html = builder.build(profile)
```

### Convenience Functions

간단한 사용을 위한 함수들:

```python
from truthound.datadocs import (
    generate_html_report,
    generate_report_from_file,
    export_report,
    export_to_pdf,
)

# 프로파일 dict에서 직접 생성
html = generate_html_report(
    profile=profile_dict,
    title="Data Quality Report",
    theme="professional",
    output_path="report.html",
)

# 파일에서 생성
html = generate_report_from_file(
    profile_path="profile.json",
    output_path="report.html",
    title="My Report",
    theme="dark",
)

# 다양한 형식으로 내보내기
export_report(profile, "report.html", format="html")
export_report(profile, "report.pdf", format="pdf")

# PDF 직접 내보내기
export_to_pdf(profile, "report.pdf", title="PDF Report")
```

### 전체 Workflow 예시

```python
import truthound as th
from truthound.datadocs import generate_html_report

# 1. 데이터 로드
df = th.load("data.csv")

# 2. 프로파일 생성
from truthound.profiler import DataProfiler
profiler = DataProfiler()
profile = profiler.profile(df)

# 3. HTML 리포트 생성
html = generate_html_report(
    profile=profile.to_dict(),
    title="Customer Data Quality Report",
    subtitle="Q4 2025 Analysis",
    theme="professional",
    output_path="customer_report.html",
)

print(f"Report generated: {len(html):,} bytes")
```

---

## Themes

### Available Themes

| Theme | Description | Best For |
|-------|-------------|----------|
| `light` | 밝고 깔끔한 디자인 | 일반적인 사용, 인쇄 |
| `dark` | 다크 모드, 선명한 색상 | 야간 작업, 프레젠테이션 |
| `professional` | 기업 스타일, 차분한 색상 | 비즈니스 리포트 (기본값) |
| `minimal` | 미니멀리스트, 모노톤 | 간결한 문서 |
| `modern` | 현대적, 그라데이션 | 마케팅, 데모 |

### Theme Preview

#### Professional Theme (Default)
- Background: Light Gray (#fafbfc)
- Primary: Blue (#2563eb)
- Surface: White (#ffffff)
- 차분하고 전문적인 느낌

#### Dark Theme
- Background: Dark (#0f172a)
- Primary: Blue (#60a5fa)
- Surface: Dark Gray (#1e293b)
- 눈의 피로를 줄이는 다크 모드

### Custom Theme

커스텀 테마를 만들 수 있습니다:

```python
from truthound.datadocs import (
    HTMLReportBuilder,
    ReportConfig,
)
from truthound.datadocs.themes import (
    ThemeConfig,
    ThemeColors,
    ThemeTypography,
    ThemeSpacing,
    ThemeAssets,
)

# 커스텀 색상 정의
custom_colors = ThemeColors(
    background="#fafafa",
    surface="#ffffff",
    text_primary="#333333",
    primary="#ff6b6b",
    secondary="#4ecdc4",
    accent="#ffe66d",
)

# 커스텀 테마 생성 (ThemeConfig from themes module)
custom_theme = ThemeConfig(
    name="my_brand",
    display_name="My Brand Theme",
    description="Custom theme for my brand",
    colors=custom_colors,
    # 추가 옵션 (선택)
    footer_text="Generated by My Company",
    show_toc=True,
)

# 빌더에 적용
config = ReportConfig(custom_theme=custom_theme)
builder = HTMLReportBuilder(config=config)
```

---

## Chart Rendering

Truthound는 출력 형식에 따라 자동으로 최적의 차트 렌더러를 선택합니다:

| Output Format | Chart Renderer | Description |
|---------------|----------------|-------------|
| **HTML** | ApexCharts | 모던, 인터랙티브, 툴팁/애니메이션 지원 |
| **PDF** | SVG | JavaScript 불필요, PDF 렌더링 최적화 |

### 지원 차트 유형

| Chart Type | HTML (ApexCharts) | PDF (SVG) |
|------------|-------------------|-----------|
| Bar | ✅ | ✅ |
| Horizontal Bar | ✅ | ✅ |
| Line | ✅ | ✅ |
| Pie | ✅ | ✅ |
| Donut | ✅ | ✅ |
| Histogram | ✅ | ✅ (bar fallback) |
| Heatmap | ✅ | ❌ |
| Scatter | ✅ | ❌ |
| Box Plot | ✅ | ❌ |
| Gauge | ✅ | ❌ |
| Radar | ✅ | ❌ |

> **Note**: PDF 출력에서 지원되지 않는 차트 유형은 Bar 차트로 대체됩니다.

### CDN URLs

ApexCharts는 CDN에서 로드됩니다:

```python
from truthound.datadocs import CDN_URLS, ChartLibrary

# ApexCharts (HTML reports)
CDN_URLS[ChartLibrary.APEXCHARTS]
# ['https://cdn.jsdelivr.net/npm/apexcharts@3.45.1/dist/apexcharts.min.js']

# SVG (no dependencies - for PDF)
CDN_URLS[ChartLibrary.SVG]
# []
```

---

## Report Sections

생성되는 리포트는 8개의 섹션으로 구성됩니다:

### 1. Overview

데이터셋의 핵심 메트릭을 카드 형태로 표시:

- **Row Count**: 전체 행 수
- **Column Count**: 전체 컬럼 수
- **Memory**: 추정 메모리 사용량
- **Duplicates**: 중복 행 수
- **Missing**: 전체 null 셀 수
- **Quality Score**: 종합 품질 점수 (0-100)

데이터 타입 분포 차트도 포함됩니다.

### 2. Data Quality

품질 차원별 점수를 원형 게이지로 표시:

- **Completeness**: 데이터 완전성 (null 비율)
- **Uniqueness**: 유일성 (unique 비율)
- **Validity**: 유효성 (형식 일치율)
- **Consistency**: 일관성

결측치 분포 차트와 경고 목록도 포함됩니다.

### 3. Column Details

각 컬럼에 대한 상세 정보:

- **Summary Table**: 모든 컬럼의 요약 테이블
- **Column Cards**: 컬럼별 상세 카드
  - 데이터 타입 배지
  - Null/Unique 비율
  - 기술 통계 (수치형)
  - 탐지된 패턴
  - 값 분포 차트

### 4. Detected Patterns

자동으로 탐지된 데이터 패턴:

- **Pattern Name**: 패턴 유형 (Email, Phone, UUID 등)
- **Match Ratio**: 일치율
- **Sample Matches**: 샘플 값

### 5. Value Distribution

값 분포 분석:

- 유일성 분포 차트
- 상위 값 빈도
- 히스토그램

### 6. Correlations

컬럼 간 상관관계:

- 상관 계수 목록
- 강/중/약 상관관계 시각화
- 양/음의 상관관계 구분

### 7. Recommendations

자동 생성된 개선 제안:

- 추천 Validator 목록
- 데이터 품질 개선 제안
- 파이프라인 권장 사항

### 8. Alerts

데이터 품질 이슈 경고:

| Severity | Color | Example |
|----------|-------|---------|
| Info | Blue | 상수 컬럼 발견 |
| Warning | Yellow | 50% 이상 결측 |
| Error | Red | 80% 이상 결측 |
| Critical | Dark Red | 데이터 무결성 위반 |

---

## Customization

### Report Configuration

`ReportConfig`로 리포트를 커스터마이징합니다:

```python
from truthound.datadocs import ReportConfig, SectionType

config = ReportConfig(
    # 테마
    theme=ReportTheme.PROFESSIONAL,

    # 포함할 섹션 (순서대로)
    sections=[
        SectionType.OVERVIEW,
        SectionType.QUALITY,
        SectionType.COLUMNS,
        SectionType.PATTERNS,
        SectionType.DISTRIBUTION,
        SectionType.CORRELATIONS,
        SectionType.RECOMMENDATIONS,
        SectionType.ALERTS,
    ],

    # 레이아웃 옵션
    include_toc=True,           # 목차 포함
    include_header=True,        # 헤더 포함
    include_footer=True,        # 푸터 포함
    include_timestamp=True,     # 생성 시간 표시

    # 커스텀 콘텐츠
    custom_css="",              # 추가 CSS
    custom_js="",               # 추가 JavaScript
    logo_url=None,              # 로고 URL
    logo_base64=None,           # 로고 Base64
    footer_text="Generated by Truthound",

    # 로컬라이제이션
    language="en",
    date_format="%Y-%m-%d %H:%M:%S",
)
```

### Custom CSS

추가 CSS를 삽입할 수 있습니다:

```python
config = ReportConfig(
    custom_css="""
    .report-title {
        color: #ff6b6b;
    }
    .metric-card {
        border: 2px solid #4ecdc4;
    }
    """,
)
```

### Custom JavaScript

추가 JavaScript를 삽입할 수 있습니다:

```python
config = ReportConfig(
    custom_js="""
    document.addEventListener('DOMContentLoaded', function() {
        console.log('Report loaded!');
    });
    """,
)
```

### Logo

회사 로고를 추가할 수 있습니다:

```python
# URL로 로고 추가
config = ReportConfig(
    logo_url="https://example.com/logo.png",
)

# Base64로 로고 추가 (오프라인 지원)
import base64
with open("logo.png", "rb") as f:
    logo_b64 = base64.b64encode(f.read()).decode()

config = ReportConfig(
    logo_base64=f"data:image/png;base64,{logo_b64}",
)
```

---

## Dashboard (Stage 2)

Stage 2는 Reflex 기반의 인터랙티브 대시보드를 제공합니다.

### Installation

```bash
pip install truthound[dashboard]
```

### Features

- **실시간 데이터 탐색**: 필터링, 정렬, 검색
- **컬럼 드릴다운**: 상세 분석
- **라이브 프로파일링**: 실시간 데이터 분석
- **인터랙티브 차트**: 줌, 팬, 호버

### Usage

```python
from truthound.datadocs import launch_dashboard

# 프로파일과 함께 실행
launch_dashboard(
    profile_path="profile.json",
    port=8080,
    host="localhost",
    title="My Dashboard",
)
```

### CLI

```bash
truthound dashboard --profile profile.json --port 8080
```

---

## CI/CD Integration

### GitHub Actions

```yaml
name: Data Quality Report

on:
  schedule:
    - cron: '0 6 * * *'  # 매일 오전 6시

jobs:
  report:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install Truthound
        run: pip install truthound

      - name: Generate Profile
        run: truthound auto-profile data.csv -o profile.json

      - name: Generate Report
        run: |
          truthound docs generate profile.json \
            -o report.html \
            --title "Daily Data Quality Report" \
            --theme professional

      - name: Upload Report
        uses: actions/upload-artifact@v4
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
    expire_in: 30 days
```

### Jenkins Pipeline

```groovy
pipeline {
    agent any

    stages {
        stage('Generate Report') {
            steps {
                sh 'pip install truthound'
                sh 'truthound auto-profile data.csv -o profile.json'
                sh 'truthound docs generate profile.json -o report.html'
            }
        }
    }

    post {
        always {
            archiveArtifacts artifacts: 'report.html'
            publishHTML([
                reportName: 'Data Quality Report',
                reportDir: '.',
                reportFiles: 'report.html'
            ])
        }
    }
}
```

### Slack Notification

리포트 생성 후 Slack으로 알림:

```bash
# 리포트 생성
truthound docs generate profile.json -o report.html

# Slack으로 전송 (curl 사용)
curl -F file=@report.html \
     -F channels=data-quality \
     -F title="Daily Data Quality Report" \
     -H "Authorization: Bearer $SLACK_BOT_TOKEN" \
     https://slack.com/api/files.upload
```

---

## Architecture

### Module Structure

```
src/truthound/datadocs/
├── __init__.py          # 모듈 exports & lazy imports
├── base.py              # 기본 타입, Enums, Protocols, Registry
├── charts.py            # 2가지 차트 렌더러 (ApexCharts, SVG)
├── sections.py          # 8가지 섹션 렌더러
├── styles.py            # CSS 스타일시트
├── builder.py           # HTMLReportBuilder
│
├── engine/              # 파이프라인 엔진
│   ├── context.py       # ReportContext, ReportData
│   ├── pipeline.py      # ReportPipeline, PipelineBuilder
│   └── registry.py      # ComponentRegistry
│
├── themes/              # 테마 시스템
│   ├── base.py          # ThemeConfig, ThemeColors, ThemeAssets
│   ├── default.py       # 6개 빌트인 테마 (Default, Light, Dark, Minimal, Modern, Professional)
│   ├── enterprise.py    # EnterpriseTheme (화이트라벨링)
│   └── loader.py        # YAML/JSON 로더
│
├── renderers/           # 템플릿 렌더링
│   ├── jinja.py         # JinjaRenderer
│   └── custom.py        # StringTemplate, FileTemplate, Callable
│
├── exporters/           # 출력 포맷
│   ├── html.py          # HtmlExporter
│   ├── pdf.py           # OptimizedPdfExporter
│   ├── markdown.py      # MarkdownExporter
│   └── json_exporter.py # JsonExporter
│
├── versioning/          # 버전 관리
│   ├── version.py       # 4개 버전 전략
│   ├── storage.py       # InMemory, File 스토리지
│   └── diff.py          # ReportDiffer
│
├── i18n/                # 다국어 지원
│   ├── catalog.py       # 2개 언어 (en, ko)
│   ├── plurals.py       # CLDR 복수형 규칙
│   └── formatting.py    # 숫자/날짜 포맷팅
│
└── dashboard/           # Stage 2: 대시보드
    ├── state.py         # Reflex state management
    ├── components.py    # UI 컴포넌트
    └── app.py           # Reflex 앱
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

### Registry Pattern

렌더러는 데코레이터로 자동 등록됩니다:

```python
from truthound.datadocs.base import (
    register_chart_renderer,
    register_section_renderer,
)

@register_chart_renderer(ChartLibrary.APEXCHARTS)
class ApexChartsRenderer(BaseChartRenderer):
    ...

@register_section_renderer(SectionType.OVERVIEW)
class OverviewSection(BaseSectionRenderer):
    ...
```

### Extensibility

커스텀 차트 렌더러 추가:

```python
from truthound.datadocs import (
    BaseChartRenderer,
    ChartLibrary,
    ChartSpec,
    register_chart_renderer,
)

# 새로운 차트 라이브러리 등록
@register_chart_renderer(ChartLibrary.CUSTOM)
class CustomChartRenderer(BaseChartRenderer):
    library = ChartLibrary.CUSTOM

    def render(self, spec: ChartSpec) -> str:
        # 커스텀 렌더링 로직
        return "<div>Custom Chart</div>"

    def get_dependencies(self) -> list[str]:
        return ["https://example.com/chart-lib.js"]
```

---

## Troubleshooting

### Common Issues

#### 1. Charts not rendering

CDN에서 JavaScript를 로드할 수 없는 경우:

```bash
# PDF로 내보내면 SVG 차트 사용 (JavaScript 불필요)
truthound docs generate profile.json -o report.pdf --format pdf
```

> **Note**: HTML 리포트는 ApexCharts를 사용합니다. SVG 차트는 PDF 내보내기에서 자동으로 사용됩니다.

#### 2. PDF export fails

**에러: `cannot load library 'libpango-1.0-0'`**

이 에러는 시스템 라이브러리가 설치되지 않았을 때 발생합니다.
`pip install truthound[pdf]`는 Python 패키지만 설치하며, 시스템 라이브러리는 별도로 설치해야 합니다.

```bash
# macOS
brew install pango cairo gdk-pixbuf libffi

# Ubuntu/Debian
sudo apt-get install libpango-1.0-0 libpangocairo-1.0-0 \
  libgdk-pixbuf2.0-0 libffi-dev shared-mime-info

# Fedora/RHEL
sudo dnf install pango gdk-pixbuf2 libffi-devel

# Alpine
apk add pango gdk-pixbuf libffi-dev
```

시스템 라이브러리 설치 후 다시 시도하세요:

```bash
truthound docs generate profile.json -o report.pdf --format pdf
```

**에러: `ModuleNotFoundError: No module named 'weasyprint'`**

Python 패키지가 설치되지 않은 경우:

```bash
pip install truthound[pdf]
```

> **Tip**: PDF가 급하지 않다면 HTML 포맷을 먼저 사용할 수 있습니다:
> ```bash
> truthound docs generate profile.json -o report.html
> ```

#### 3. Dashboard import error

대시보드 의존성이 설치되지 않은 경우:

```bash
pip install truthound[dashboard]
```

#### 4. Large profile file

프로파일 파일이 너무 큰 경우, 샘플링을 사용하세요:

```bash
truthound auto-profile data.csv -o profile.json --sample-size 100000
```

---

## API Reference

### Enums

```python
class ReportTheme(str, Enum):
    LIGHT = "light"
    DARK = "dark"
    PROFESSIONAL = "professional"
    MINIMAL = "minimal"
    MODERN = "modern"

class ChartLibrary(str, Enum):
    APEXCHARTS = "apexcharts"  # Default for HTML
    SVG = "svg"                # Used for PDF export

class ChartType(str, Enum):
    BAR = "bar"
    HORIZONTAL_BAR = "horizontal_bar"
    LINE = "line"
    PIE = "pie"
    DONUT = "donut"
    HISTOGRAM = "histogram"
    HEATMAP = "heatmap"
    SCATTER = "scatter"
    BOX = "box"
    GAUGE = "gauge"
    RADAR = "radar"
    TABLE = "table"

class SectionType(str, Enum):
    OVERVIEW = "overview"
    COLUMNS = "columns"
    QUALITY = "quality"
    PATTERNS = "patterns"
    DISTRIBUTION = "distribution"
    CORRELATIONS = "correlations"
    RECOMMENDATIONS = "recommendations"
    ALERTS = "alerts"
    CUSTOM = "custom"

class SeverityLevel(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
```

### Data Classes

```python
@dataclass
class ReportConfig:
    theme: ReportTheme = ReportTheme.PROFESSIONAL
    custom_theme: ThemeConfig | None = None
    sections: list[SectionType] = ...
    include_toc: bool = True
    include_header: bool = True
    include_footer: bool = True
    include_timestamp: bool = True
    custom_css: str = ""
    custom_js: str = ""
    logo_url: str | None = None
    logo_base64: str | None = None
    footer_text: str = "Generated by Truthound"
    language: str = "en"
    date_format: str = "%Y-%m-%d %H:%M:%S"

@dataclass
class ChartSpec:
    chart_type: ChartType
    title: str = ""
    subtitle: str = ""
    labels: list[str] = field(default_factory=list)
    values: list[float | int] = field(default_factory=list)
    series: list[dict] | None = None
    colors: list[str] | None = None
    height: int = 300
    width: int | None = None
    show_legend: bool = True
    show_labels: bool = True
    show_grid: bool = True
    animation: bool = True
    options: dict[str, Any] = field(default_factory=dict)  # 추가 옵션

@dataclass
class AlertSpec:
    title: str
    message: str
    severity: SeverityLevel = SeverityLevel.INFO
    column: str | None = None
    suggestion: str | None = None
```

---

## See Also

- [Auto-Profiling (docs/PROFILER.md)](PROFILER.md) - 데이터 프로파일링
- [Reporters (docs/REPORTERS.md)](REPORTERS.md) - 다른 리포트 형식
- [Examples (docs/EXAMPLES.md)](EXAMPLES.md) - 사용 예시
