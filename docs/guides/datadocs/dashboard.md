# Dashboard (Stage 2)

Truthound Data Docs는 Reflex 기반의 인터랙티브 대시보드를 제공합니다.

## 설치

```bash
pip install truthound[dashboard]
```

## Quick Start

### CLI

```bash
# 프로파일과 함께 실행
truthound dashboard --profile profile.json

# 커스텀 포트
truthound dashboard --profile profile.json --port 3000

# 외부 접근 허용
truthound dashboard --profile profile.json --host 0.0.0.0
```

### Python API

```python
from truthound.datadocs.dashboard import launch_dashboard

launch_dashboard(
    profile_path="profile.json",
    port=8080,
    host="localhost",
    title="My Dashboard",
    debug=False,
)
```

## DashboardConfig

대시보드 설정을 위한 데이터 클래스입니다.

```python
from truthound.datadocs.dashboard import DashboardConfig

config = DashboardConfig(
    # 서버 설정
    host="localhost",
    port=8080,
    debug=False,

    # 테마
    theme="light",            # "light" 또는 "dark"
    primary_color="blue",

    # 기능 토글
    show_raw_data=True,
    show_correlations=True,
    show_patterns=True,
    enable_export=True,

    # 데이터
    profile_path="profile.json",     # 프로파일 파일 경로
    profile_data=None,               # 또는 프로파일 딕셔너리

    # 브랜딩
    title="Truthound Dashboard",
    logo_url=None,
)
```

## DashboardApp

대시보드 애플리케이션 클래스입니다.

```python
from truthound.datadocs.dashboard import DashboardApp, DashboardConfig

# 설정으로 생성
config = DashboardConfig(
    profile_path="profile.json",
    title="My Dashboard",
    port=8080,
)
app = DashboardApp(config)

# 프로파일 로드
app.load_profile(profile_path="profile.json")
# 또는
app.load_profile(profile_data=profile_dict)

# 서버 실행
app.run(host="localhost", port=8080, debug=False)
```

## 대시보드 구성

### 페이지

대시보드는 3개의 메인 페이지로 구성됩니다:

#### 1. Overview (개요)

- **메트릭 그리드**: Rows, Columns, Memory, Quality Score
- **경고 목록**: 데이터 품질 이슈

#### 2. Columns (컬럼)

- **검색**: 컬럼 이름으로 검색
- **컬럼 카드 그리드**: 각 컬럼별 상세 정보
  - 데이터 타입 배지
  - Null/Unique/Distinct 비율

#### 3. Quality (품질)

- **전체 품질 점수**: 대형 디스플레이
- **품질 분석 설명**

### UI 기능

- **사이드바**: 페이지 네비게이션
- **테마 토글**: 라이트/다크 모드 전환
- **반응형**: 모바일/태블릿 대응

## State Management

대시보드는 Reflex의 상태 관리를 사용합니다.

```python
# 내부 State 클래스 (참고용)
class State(rx.State):
    # 프로파일 데이터
    profile_data: dict = {}
    row_count: int = 0
    column_count: int = 0
    memory_bytes: int = 0
    quality_score: float = 100.0
    columns: list = []
    correlations: list = []
    alerts: list = []

    # UI 상태
    sidebar_open: bool = True
    active_tab: str = "overview"
    selected_column: str = ""
    search_query: str = ""
    theme: str = "light"
    is_loading: bool = True

    # 액션
    def load_profile(self, data: dict) -> None: ...
    def toggle_sidebar(self) -> None: ...
    def set_tab(self, tab: str) -> None: ...
    def select_column(self, column: str) -> None: ...
    def set_search(self, query: str) -> None: ...
    def toggle_theme(self) -> None: ...

    # 계산된 속성
    @rx.var
    def filtered_columns(self) -> list: ...
    @rx.var
    def format_memory(self) -> str: ...
```

## Convenience Functions

### launch_dashboard

```python
from truthound.datadocs.dashboard import launch_dashboard

launch_dashboard(
    profile_path="profile.json",  # 또는 None
    profile_data=None,            # 또는 프로파일 딕셔너리
    port=8080,
    host="localhost",
    title="Truthound Dashboard",
    debug=False,
)
```

### create_app

```python
from truthound.datadocs.dashboard import create_app, DashboardConfig

# 기본 설정으로 생성
app = create_app(profile_path="profile.json")

# 커스텀 설정으로 생성
config = DashboardConfig(
    profile_path="profile.json",
    title="Custom Dashboard",
    theme="dark",
)
app = create_app(config=config)

# 서버 실행
app.run()
```

## CLI 옵션

```bash
truthound dashboard [OPTIONS]
```

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--profile` | `-p` | None | 프로파일 JSON 파일 경로 |
| `--port` | | 8080 | 서버 포트 |
| `--host` | | "localhost" | 서버 호스트 |
| `--title` | `-t` | "Truthound Dashboard" | 대시보드 제목 |
| `--debug` | | False | 디버그 모드 |

## 요구 사항

대시보드 기능은 Reflex 패키지가 필요합니다:

```bash
pip install truthound[dashboard]
```

Reflex가 설치되지 않은 경우:

```python
from truthound.datadocs.dashboard import launch_dashboard

# ImportError: Dashboard requires Reflex.
# Install with: pip install truthound[dashboard]
```

## API Reference

### DashboardConfig

```python
@dataclass
class DashboardConfig:
    # 서버 설정
    host: str = "localhost"
    port: int = 8080
    debug: bool = False

    # 테마
    theme: str = "light"
    primary_color: str = "blue"

    # 기능 토글
    show_raw_data: bool = True
    show_correlations: bool = True
    show_patterns: bool = True
    enable_export: bool = True

    # 데이터
    profile_path: str | None = None
    profile_data: dict[str, Any] | None = None

    # 브랜딩
    title: str = "Truthound Dashboard"
    logo_url: str | None = None
```

### DashboardApp

```python
class DashboardApp:
    def __init__(self, config: DashboardConfig | None = None) -> None:
        """대시보드 애플리케이션 초기화."""
        ...

    def load_profile(
        self,
        profile_path: str | Path | None = None,
        profile_data: dict | None = None,
    ) -> None:
        """프로파일 데이터 로드."""
        ...

    def run(
        self,
        host: str | None = None,
        port: int | None = None,
        debug: bool | None = None,
    ) -> None:
        """대시보드 서버 실행."""
        ...
```

### launch_dashboard

```python
def launch_dashboard(
    profile_path: str | Path | None = None,
    profile_data: dict | None = None,
    port: int = 8080,
    host: str = "localhost",
    title: str = "Truthound Dashboard",
    debug: bool = False,
) -> None:
    """인터랙티브 대시보드 실행."""
    ...
```

### create_app

```python
def create_app(
    profile_path: str | Path | None = None,
    profile_data: dict | None = None,
    config: DashboardConfig | None = None,
) -> DashboardApp:
    """대시보드 애플리케이션 인스턴스 생성."""
    ...
```

## See Also

- [HTML Reports](html-reports.md) - 정적 HTML 리포트
- [Themes](themes.md) - 테마 커스터마이징
- [truthound-dashboard](https://github.com/seadonggyun4/truthound-dashboard) - 별도 대시보드 저장소
