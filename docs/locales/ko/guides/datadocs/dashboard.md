# 선택형 Data Docs 프로파일 뷰어

Core에는 프로파일 데이터를 조회하는 Reflex 기반 로컬 뷰어가 있습니다. HTML·PDF
리포트 생성에 필요한 UI는 아닙니다. 필요한 Reflex 의존성은 Truthound의 `dashboard`
extra로 설치합니다.

```python
from truthound.datadocs import launch_dashboard

launch_dashboard(
    profile_path="profile.json",
    host="localhost",
    port=8080,
    title="Local profile inspection",
)
```

launcher 인자는 `profile_path`, `profile_data`, `port`, `host`, `title`, `debug`입니다.
경로를 전달하면 파일 입력이 우선합니다. UI에는 프로파일·컬럼·패턴·상관관계 조회와
프로필 JSON 다운로드가 있습니다.

서버를 바로 시작하지 않고 wrapper를 구성할 수도 있습니다.

```python
from truthound.datadocs import create_dashboard_app, get_dashboard_config

Config = get_dashboard_config()
viewer = create_dashboard_app(config=Config(
    host="localhost", port=8080, profile_path="profile.json",
))
# viewer.run() starts the optional server.
```

이 함수는 report HTML이 아닌 `DashboardApp` wrapper를 반환합니다. 개발 소스의
`run()`은 비공개 임시 프로젝트를 준비하고 컴파일된 frontend와 네이티브 backend를
하나의 HTTP 포트에서 실행합니다. host·port·debug override를 전달하며 debug는 서버
로깅을 선택합니다. 인증 경계가 없으므로 로컬에서 사용하세요.

## 개발 소스의 명시적 준비와 관리형 수명주기

checkout에서 `pip install '.[dashboard]'`로 설치하고 Node/npm을 준비합니다.
지원 Reflex 범위는 `>=0.8.25,<0.9`이며 로컬 실행은 0.8.25로 확인합니다.
extra는 SQLAlchemy asyncio 지원과 WebSocket을 포함한 Uvicorn ASGI 서버를 명시적으로
설치합니다. 로컬 확인에는 Node 20.20.2/npm 10.8.2를 사용했습니다. 준비 단계는 npm으로
frontend 패키지를 설치하고 네이티브 앱을 빌드하며 서버 시작·원격 배포와 별개입니다.
생성한 프로젝트에서 Reflex telemetry를 끕니다. 제목을 포함한 설정은 JSON 데이터입니다.

```python
from pathlib import Path
from truthound.datadocs.dashboard import DashboardConfig, ReflexDashboardServer

# 부모 디렉터리는 있어야 하며 기존 프로젝트에는 덮어쓰지 않습니다.
server = ReflexDashboardServer(Path("private-viewer-project"), DashboardConfig(
    host="127.0.0.1", port=8080, profile_path="profile.json",
))
server.prepare()  # 명시적 빌드, 기본 제한 300초·최대 600초
try:
    server.start()  # 기본 제한 10초·최대 30초, 의존성 설치 없음
    print(server.status())  # running, ready, url, exit_code
    server.wait()
finally:
    server.close()
```

운영자가 선택한 상대 profile 경로는 서버 생성 시 절대 경로로 해석합니다. 자식은 비공개
프로젝트에서 실행합니다. `stop()` 후 같은 준비 설정으로 재시작할 수 있습니다. 설정·프로필
파일 변경은 거부하며 새 프로젝트가 필요합니다. 소유자가 프로세스 그룹을 정리하고 부모 강제 종료도
네이티브 자식을 종료합니다. 준비 프로젝트에는 설정·컴파일 산출물의 프로필 값이 있으므로
비공개로 보관하고 저장소·패키지에 포함하지 마세요. `show_raw_data`·`show_correlations`·
`show_patterns`·`enable_export`는 표시·다운로드 컨트롤을 제어하며 승인된 브라우저에서
값을 보호하는 기능은 아닙니다. 실제 페이지 컴파일·프로필 상태·HTTP 준비·재시작·MCP 해제·
부모 종료를 로컬 테스트합니다. 원격 배포는 미검증입니다.

독립적인 리포트 산출물은 `HTMLReportBuilder`, `generate_html_report`,
`generate_validation_report`로 생성합니다. [HTML 리포트](html-reports.md)와
[Core 인터페이스 개요](../../dashboard/index.md)의 별도 Streamlit UI 설명을 참고하세요.
