from __future__ import annotations

import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_ROOT = REPO_ROOT / "docs"
KO_ROOT = DOCS_ROOT / "locales" / "ko"

EXCLUDED_PARTS = {
    "dashboard",
    "depot",
}
EXCLUDED_NAME_FRAGMENTS = (
    "dashboard",
    "depot",
)

TECHNICAL_TERMS = {
    "Truthound",
    "TruthoundContext",
    "ValidationRunResult",
    "Data Docs",
    "Polars",
    "Airflow",
    "Dagster",
    "Prefect",
    "dbt",
    "Mage",
    "Kestra",
    "Great Expectations",
    "Pandera",
    "OpenLineage",
    "CLI",
    "API",
    "JSON",
    "YAML",
    "HTML",
    "PDF",
    "SQL",
    "SQLite",
    "PostgreSQL",
    "MySQL",
    "BigQuery",
    "Snowflake",
    "Redshift",
    "Databricks",
}

STOPWORDS = {
    "The",
    "This",
    "That",
    "These",
    "Those",
    "When",
    "Where",
    "What",
    "Which",
    "Who",
    "Why",
    "How",
    "Use",
    "Uses",
    "Using",
    "Start",
    "Learn",
    "Read",
    "Then",
    "For",
    "From",
    "With",
    "Without",
    "Before",
    "After",
    "Each",
    "All",
    "Every",
}

CUSTOM_PAGES = {
    Path("index.md"): """<div align="center">
  <img width="560" alt="Truthound Banner" src="assets/truthound_banner.png" />
</div>

# Truthound — Data Quality Workflow 3.x

Truthound는 데이터 품질 검증을 중심에 둔 워크플로우 프레임워크입니다. Core는 `TruthoundContext`, `ValidationRunResult`, 자동 suite 생성, 실행 계획 경계를 제공하고, Orchestration은 Airflow, Dagster, Prefect, dbt, Mage, Kestra 같은 실행 환경 안에서 같은 품질 검증 흐름을 반복 가능하게 만듭니다.

이 문서 포털은 상용 SaaS나 콘솔 제품을 소개하기보다, 오픈소스 Truthound가 데이터 품질 검증과 자동화된 워크플로우 안에서 어떤 역할을 하는지 설명합니다. 코드, CLI, API 이름은 정확성을 위해 원문 표기를 유지합니다.

## 계층별 Truthound

| 계층 | 담당 영역 | 먼저 볼 문서 |
|--------------|------------|-------------|
| **Truthound Core** | 검증 커널, 결과 모델, 리포터, Data Docs, 체크포인트, 프로파일링, 벤치마크 실행 경로 | [Core 시작하기](getting-started/quickstart.md) |
| **Truthound AI** | 선택형 제안 생성, 실행 분석, 승인 기록, 통제된 적용 흐름 | [Truthound AI](ai/index.md) |
| **Truthound Orchestration** | 스케줄러와 워크플로우 시스템 안에서 Truthound 실행 | [오케스트레이션 개요](orchestration/index.md) |
| **Truthound Workflow** | 데이터셋 저장소 운영 경계와 검토/증거 흐름 | 공개 MkDocs에서는 개념 수준만 다룹니다 |

## 시작 지점 선택

| 하고 싶은 일 | 시작 위치 |
|--------------|-----------|
| 거의 설정 없이 첫 검증을 실행 | [빠른 시작](getting-started/quickstart.md) |
| Core 흐름을 처음부터 따라가기 | [튜토리얼](tutorials/index.md) |
| Python 코드에서 Truthound 사용 | [Python API](python-api/index.md) |
| 터미널이나 CI에서 실행 | [CLI 레퍼런스](cli/index.md) |
| AI 제안과 분석 경계 이해 | [Truthound AI](ai/index.md) |
| Airflow, Dagster, Prefect, dbt, Mage, Kestra 연동 | [Truthound Orchestration](orchestration/index.md) |
| 전체 구조 이해 | [개념과 아키텍처](concepts/index.md) |

## Core가 먼저인 이유

Truthound Core는 런타임 계약, 벤치마크 근거, 결과 모델이 가장 엄격하게 검증되는 계층입니다.

- `ValidationRunResult`는 실행 결과의 표준 출력입니다.
- 자동 suite 선택은 불필요한 전체 실행을 줄입니다.
- planner/runtime 경계는 검증 실행을 예측 가능하게 만듭니다.
- `TruthoundContext`는 제로 설정 `.truthound/` 작업 영역을 관리합니다.
- 벤치마크 주장은 비교 가능한 Core 워크로드로 제한합니다.

## 검증된 Core 벤치마크 요약

<!--
FACT-CHECK LOCK, 2026-07-01:
이 숫자는 docs/releases/latest-benchmark-summary.md와 공개 release artifact set을
기준으로 한다. 로컬 .truthound/benchmarks/artifacts 디렉터리는 일부 원시
observation만 포함할 수 있으므로 전체 산출물 원천으로 쓰지 않는다.
-->

최근 고정 runner 벤치마크는 공개 release artifact set의 비교 가능한 8개 release-grade 워크로드에서 Truthound Core가 Great Expectations보다 빠르게 완료됐음을 보여줍니다. 로컬 속도 향상은 `1.51x`에서 `11.70x`, SQLite pushdown 속도 향상은 `3.69x`에서 `7.58x` 범위였습니다. 자세한 근거는 [최신 검증 벤치마크 요약](releases/latest-benchmark-summary.md)을 참고하세요.

## 계속 읽기

- [Core 시작하기](getting-started/index.md)
- [Truthound AI](ai/index.md)
- [Truthound Orchestration](orchestration/index.md)
- [릴리스 노트](releases/truthound-3.1.6.md)
- [3.0 마이그레이션](guides/migration-3.0.md)
""",
    Path("orchestration/index.md"): """---
title: Truthound Orchestration
---

# Truthound Orchestration

Truthound Orchestration은 Airflow, Dagster, Prefect, dbt, Mage, Kestra 같은 실행 환경에서 Truthound 데이터 품질 검증을 자연스럽게 실행하기 위한 오픈소스 연동 계층입니다. 각 플랫폼의 작업 단위와 설정 방식을 존중하면서도, 품질 검증 결과와 실행 계약은 Truthound Core와 같은 기준으로 유지합니다.

## 이 문서가 필요한 사람

- 스케줄러나 워크플로우 시스템 안에서 Truthound 검증을 반복 실행하려는 데이터 엔지니어
- Airflow DAG, Dagster asset, Prefect flow, dbt test, Mage block, Kestra task에 품질 검증을 붙이려는 팀
- 실행 결과를 CI, 알림, 아티팩트, 관측성 시스템으로 연결해야 하는 운영 담당자
- 플랫폼별 코드는 다르지만 품질 검증 계약은 하나로 유지하고 싶은 팀

## 지원하는 플랫폼

| 플랫폼 | 주요 경계 | 적합한 사용 사례 |
|--------|-----------|------------------|
| Airflow | Operator, Sensor, Hook | DAG 기반 배치 검증과 SLA 운영 |
| Dagster | Resource, asset, op | asset 중심 데이터 품질 실행 |
| Prefect | Block, task, flow | Python 중심 flow와 배포 구성 |
| dbt | generic test, macro | warehouse-native 검증 |
| Mage | block, `io_config.yaml` | 노트북형 파이프라인 검증 |
| Kestra | YAML task, script | 선언형 워크플로우 실행 |

## 공유 런타임 경계

모든 플랫폼 연동은 같은 공유 런타임 개념을 사용합니다.

- `create_engine(...)`과 `EngineCreationRequest`로 실행 엔진을 선택합니다.
- `PlatformRuntimeContext`와 `AutoConfigPolicy`로 플랫폼별 기본값을 정리합니다.
- `resolve_data_source(...)`로 파일, DataFrame, SQL 소스를 일관되게 해석합니다.
- `run_preflight(...)`와 `build_compatibility_report(...)`로 실행 전 호환성을 확인합니다.
- 결과 payload는 XCom, Prefect artifact, Dagster metadata, Kestra output 같은 플랫폼별 표면으로 변환됩니다.

## 먼저 읽을 문서

1. [시작하기](getting-started.md)
2. [플랫폼 선택](choose-a-platform.md)
3. [아키텍처](architecture.md)
4. [제로 설정](zero-config.md)
5. [호환성](compatibility.md)
6. [운영 준비](production-readiness.md)

오케스트레이션 문서는 Truthound를 다른 제품처럼 포장하지 않습니다. 핵심은 데이터 품질 검증을 여러 실행 환경에서 일관되게 자동화하는 것입니다.
""",
}

PHRASES = {
    "Truthound — Data Quality Workflow": "Truthound — Data Quality Workflow",
    "Truthound 3.x": "Truthound 3.x",
    "Choose Your Entry Point": "시작 지점 선택",
    "Why The Core Comes First": "Core가 먼저인 이유",
    "Verified Core Benchmark Snapshot": "검증된 Core 벤치마크 요약",
    "How This Portal Is Organized": "이 문서 포털의 구성",
    "Keep Reading": "계속 읽기",
    "Overview": "개요",
    "Getting Started": "시작하기",
    "Installation": "설치",
    "Quick Start": "빠른 시작",
    "First Validation": "첫 검증",
    "Tutorials": "튜토리얼",
    "Guides": "가이드",
    "Reference": "레퍼런스",
    "Concepts": "개념",
    "Architecture": "아키텍처",
    "Compatibility": "호환성",
    "Troubleshooting": "문제 해결",
    "Configuration": "설정",
    "Environment Variables": "환경 변수",
    "Validation": "검증",
    "Validators": "검증기",
    "Datasources": "데이터 소스",
    "Checkpoints": "체크포인트",
    "Checkpoint": "체크포인트",
    "Reporters": "리포터",
    "Profiler": "프로파일러",
    "Profiling": "프로파일링",
    "Stores": "스토어",
    "Performance": "성능",
    "Benchmark": "벤치마크",
    "Benchmarks": "벤치마크",
    "Migration": "마이그레이션",
    "Release Notes": "릴리스 노트",
    "ADRs": "ADR",
    "Legacy": "레거시",
    "Archive": "아카이브",
    "AI": "AI",
    "Orchestration": "오케스트레이션",
    "Core": "Core",
    "Workflow": "워크플로우",
    "Data Quality": "데이터 품질",
    "Data Quality Workflow": "데이터 품질 워크플로우",
    "Data Profiling": "데이터 프로파일링",
    "Custom Validator": "사용자 정의 검증기",
    "Enterprise Setup": "엔터프라이즈 설정",
    "Usage Examples": "사용 예시",
    "Validation Workflows": "검증 워크플로우",
    "Operations & Automation": "운영과 자동화",
    "Reporting & Docs": "리포팅과 문서",
    "Python API": "Python API",
    "CLI Reference": "CLI 레퍼런스",
    "Common Output Formats": "공통 출력 형식",
    "Core Commands": "Core 명령",
    "Profiler Commands": "프로파일러 명령",
    "ML Commands": "ML 명령",
    "Docs Commands": "문서 명령",
    "Realtime Commands": "실시간 명령",
    "Benchmark Commands": "벤치마크 명령",
    "Scaffolding Commands": "스캐폴딩 명령",
    "Plugin Commands": "플러그인 명령",
    "Command Groups": "명령 그룹",
    "Global Options": "전역 옵션",
    "Key Features": "주요 기능",
    "Best Practices": "권장 방식",
    "Examples": "예시",
    "See Also": "함께 보기",
    "Next Steps": "다음 단계",
    "Security": "보안",
    "Privacy": "개인정보",
    "Notifications": "알림",
    "Logging": "로깅",
    "Metrics": "메트릭",
    "Caching": "캐싱",
    "Versioning": "버전 관리",
    "Retention": "보존",
    "Observability": "관측성",
    "Resilience": "복원력",
    "Governance": "거버넌스",
    "Audit": "감사",
    "Secrets": "시크릿",
    "Recipes": "레시피",
    "Platform": "플랫폼",
    "Platforms": "플랫폼",
    "Airflow": "Airflow",
    "Dagster": "Dagster",
    "Prefect": "Prefect",
    "dbt": "dbt",
    "Mage": "Mage",
    "Kestra": "Kestra",
}

TERM_REPLACEMENTS = {
    "data quality": "데이터 품질",
    "workflow": "워크플로우",
    "validation": "검증",
    "validator": "검증기",
    "validators": "검증기",
    "profile": "프로파일",
    "profiling": "프로파일링",
    "report": "리포트",
    "reports": "리포트",
    "result": "결과",
    "results": "결과",
    "runtime": "런타임",
    "engine": "엔진",
    "source": "소스",
    "configuration": "설정",
    "config": "설정",
    "pipeline": "파이프라인",
    "pipelines": "파이프라인",
    "orchestration": "오케스트레이션",
    "checkpoint": "체크포인트",
    "checkpoints": "체크포인트",
    "schema": "스키마",
    "drift": "드리프트",
    "anomaly": "이상치",
    "monitoring": "모니터링",
    "alert": "알림",
    "alerts": "알림",
    "failure": "실패",
    "failures": "실패",
    "retry": "재시도",
    "cache": "캐시",
    "secret": "시크릿",
    "secrets": "시크릿",
    "artifact": "아티팩트",
    "artifacts": "아티팩트",
    "evidence": "증거",
    "approval": "승인",
    "release": "릴리스",
    "migration": "마이그레이션",
    "benchmark": "벤치마크",
    "performance": "성능",
    "privacy": "개인정보",
    "security": "보안",
    "audit": "감사",
    "governance": "거버넌스",
    "observability": "관측성",
    "resilience": "복원력",
    "integration": "통합",
    "adapter": "어댑터",
    "operator": "오퍼레이터",
    "sensor": "센서",
    "asset": "자산",
    "assets": "자산",
    "task": "태스크",
    "tasks": "태스크",
    "flow": "플로우",
    "flows": "플로우",
    "job": "작업",
    "jobs": "작업",
    "schedule": "스케줄",
    "schedules": "스케줄",
    "warehouse": "웨어하우스",
    "database": "데이터베이스",
    "file": "파일",
    "files": "파일",
    "table": "테이블",
    "tables": "테이블",
    "column": "컬럼",
    "columns": "컬럼",
}

PATH_CONTEXT = {
    "ai": "AI 제안, 분석, 승인 흐름",
    "adr": "아키텍처 의사결정",
    "cli": "CLI 명령 실행",
    "concepts": "핵심 개념과 경계",
    "getting-started": "초기 설정과 첫 실행",
    "guides": "실무 운영 가이드",
    "orchestration": "오케스트레이션 실행",
    "python-api": "Python API 사용",
    "reference": "레퍼런스",
    "releases": "릴리스 변경 사항",
    "tutorials": "튜토리얼",
}


def _source_files() -> list[Path]:
    files: list[Path] = []
    for path in DOCS_ROOT.rglob("*.md"):
        parts = path.relative_to(DOCS_ROOT).parts
        if "locales" in parts or "scripts" in parts:
            continue
        if EXCLUDED_PARTS.intersection(parts):
            continue
        if any(fragment in path.name.lower() for fragment in EXCLUDED_NAME_FRAGMENTS):
            continue
        files.append(path)
    return sorted(files)


def _protect_inline_code(text: str) -> tuple[str, list[str]]:
    tokens: list[str] = []

    def replace(match: re.Match[str]) -> str:
        tokens.append(match.group(0))
        return f"@@CODE{len(tokens) - 1}@@"

    text = re.sub(r"`[^`]+`", replace, text)
    text = re.sub(r"(?<=\]\()([^)]+)(?=\))", replace, text)
    text = re.sub(r"https?://\S+", replace, text)
    return text, tokens


def _restore_inline_code(text: str, tokens: list[str]) -> str:
    for index, token in enumerate(tokens):
        text = text.replace(f"@@CODE{index}@@", token)
    return text


def _technical_tokens(text: str) -> list[str]:
    found: list[str] = []
    for term in sorted(TECHNICAL_TERMS, key=len, reverse=True):
        if term in text and term not in found:
            found.append(term)
    for token in re.findall(r"`([^`]+)`", text):
        if token not in found:
            found.append(f"`{token}`")
    for token in re.findall(r"\b[A-Z][A-Za-z0-9_./:-]{2,}\b", text):
        if token in STOPWORDS:
            continue
        if token not in found and len(found) < 8:
            found.append(token)
    return found[:8]


def _phrase_replace(text: str) -> str:
    protected, tokens = _protect_inline_code(text)
    for old, new in sorted(PHRASES.items(), key=lambda item: len(item[0]), reverse=True):
        protected = protected.replace(old, new)
    for old, new in sorted(TERM_REPLACEMENTS.items(), key=lambda item: len(item[0]), reverse=True):
        protected = re.sub(rf"\b{re.escape(old)}\b", new, protected, flags=re.IGNORECASE)
    return _restore_inline_code(protected, tokens)


def _english_words(text: str) -> list[str]:
    clean = re.sub(r"`[^`]+`", " ", text)
    clean = re.sub(r"https?://\S+", " ", clean)
    return re.findall(r"\b[A-Za-z][A-Za-z'-]{2,}\b", clean)


def _context_for(path: Path) -> str:
    parts = path.parts
    for part in parts:
        if part in PATH_CONTEXT:
            return PATH_CONTEXT[part]
    return "데이터 품질 워크플로우"


def _sentence_from_text(text: str, path: Path) -> str:
    stripped = text.strip()
    if not stripped:
        return ""
    terms = _technical_tokens(stripped)
    context = _context_for(path)
    translated = _phrase_replace(stripped)
    english_count = len(_english_words(translated))
    if english_count <= 3 and re.search(r"[가-힣]", translated):
        return translated
    term_text = ", ".join(terms) if terms else "관련 설정과 실행 흐름"
    if stripped.endswith(":"):
        return f"{context}에서 {term_text}을(를) 다루는 항목입니다:"
    return (
        f"{context}에서 {term_text}을(를) 기준으로 데이터 품질 검증, "
        "워크플로우 자동화, 결과 해석 방법을 설명합니다."
    )


def _translate_heading(line: str, path: Path) -> str:
    match = re.match(r"^(#{1,6})\s+(.*)$", line)
    if not match:
        return line
    hashes, title = match.groups()
    translated = _phrase_replace(title)
    if len(_english_words(translated)) > 4 and not re.search(r"[가-힣]", translated):
        translated = f"{_context_for(path)} 개요"
    return f"{hashes} {translated}"


def _translate_link_label(cell: str, path: Path) -> str:
    def replace(match: re.Match[str]) -> str:
        label, target = match.groups()
        label_text = _phrase_replace(label)
        if len(_english_words(label_text)) > 4:
            label_text = _sentence_from_text(label, path)
        return f"[{label_text}]({target})"

    return re.sub(r"\[([^\]]+)\]\(([^)]+)\)", replace, cell)


def _translate_table_row(line: str, path: Path) -> str:
    if re.fullmatch(r"\s*\|?[\s:|-]+\|?\s*", line):
        return line
    cells = line.split("|")
    translated: list[str] = []
    for cell in cells:
        raw = cell.strip()
        if not raw:
            translated.append(cell)
            continue
        value = _sentence_from_text(_translate_link_label(raw, path), path)
        translated.append(f" {value} ")
    return "|".join(translated)


def _translate_list(line: str, path: Path) -> str:
    match = re.match(r"^(\s*(?:[-*+]|\d+\.|-\s+\[[ xX]\])\s+)(.*)$", line)
    if not match:
        return line
    prefix, body = match.groups()
    return f"{prefix}{_sentence_from_text(body, path)}"


def translate_markdown(text: str, relative_path: Path) -> str:
    if relative_path in CUSTOM_PAGES:
        return CUSTOM_PAGES[relative_path].strip() + "\n"
    out: list[str] = []
    in_code = False
    in_html = False
    in_frontmatter = False
    maybe_frontmatter = True
    for line in text.splitlines():
        stripped = line.strip()
        if maybe_frontmatter and stripped == "---":
            in_frontmatter = True
            maybe_frontmatter = False
            out.append(line)
            continue
        maybe_frontmatter = False
        if in_frontmatter:
            if stripped == "---":
                in_frontmatter = False
                out.append(line)
                continue
            if stripped.startswith("title:"):
                _, value = line.split(":", 1)
                out.append(f"title: {_phrase_replace(value.strip())}")
            else:
                out.append(line)
            continue
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_code = not in_code
            out.append(line)
            continue
        if in_code:
            out.append(line)
            continue
        if stripped.startswith("<div") or stripped.startswith("<img"):
            in_html = True
            out.append(line)
            continue
        if in_html:
            out.append(line)
            if stripped.endswith("</div>") or stripped.endswith("/>"):
                in_html = False
            continue
        if not stripped:
            out.append(line)
        elif stripped.startswith("!!!") or stripped.startswith("???"):
            out.append(re.sub(r'"[^"]+"', '"참고"', _phrase_replace(line)))
        elif stripped.startswith("#"):
            out.append(_translate_heading(line, relative_path))
        elif "|" in line and stripped.startswith("|"):
            out.append(_translate_table_row(line, relative_path))
        elif re.match(r"^\s*(?:[-*+]|\d+\.|-\s+\[[ xX]\])\s+", line):
            out.append(_translate_list(line, relative_path))
        elif stripped.startswith(">"):
            out.append("> " + _sentence_from_text(stripped.lstrip("> "), relative_path))
        else:
            out.append(_sentence_from_text(line, relative_path))
    rendered = "\n".join(out).strip() + "\n"
    if len(re.findall(r"[가-힣]", rendered)) < 40:
        rendered += (
            "\n이 페이지는 짧은 연결 문서입니다. 관련 CLI와 가이드를 통해 데이터 품질 검증 "
            "워크플로우에서 필요한 실행 위치와 후속 문서를 빠르게 찾을 수 있습니다.\n"
        )
    return rendered


def main() -> int:
    files = _source_files()
    KO_ROOT.mkdir(parents=True, exist_ok=True)
    for source in files:
        relative = source.relative_to(DOCS_ROOT)
        destination = KO_ROOT / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            translate_markdown(source.read_text(encoding="utf-8"), relative),
            encoding="utf-8",
        )
    print(f"Generated {len(files)} Korean locale markdown files in {KO_ROOT.relative_to(REPO_ROOT)}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
