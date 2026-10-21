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
경로를 전달하면 파일 입력이 우선합니다. UI에는 프로파일·컬럼·패턴·상관관계 조회 화면이
있지만 소스 존재만으로 최신 Reflex에서 모든 컨트롤이 정상 동작한다고 보장하지 않습니다.

서버를 바로 시작하지 않고 wrapper를 구성할 수도 있습니다.

```python
from truthound.datadocs import create_dashboard_app, get_dashboard_config

Config = get_dashboard_config()
viewer = create_dashboard_app(config=Config(
    host="localhost", port=8080, profile_path="profile.json",
))
# viewer.run() starts the optional server.
```

이 함수는 report HTML이 아닌 `DashboardApp` wrapper를 반환합니다. 현재 서버 호출에
`debug`는 전달되지 않습니다. 설정 필드가 있다는 이유만으로 인증·내보내기 제어가 모두
구현됐다고 가정하지 않습니다. UI를 외부에 공개하거나 인증이 강제된다고 가정하지 마세요.
이번 문서 변경에서 GUI 시작·의존성 호환성을 실행 검증하지는 않았습니다.

독립적인 리포트 산출물은 `HTMLReportBuilder`, `generate_html_report`,
`generate_validation_report`로 생성합니다. [HTML 리포트](html-reports.md)와
[Core 인터페이스 개요](../../dashboard/index.md)의 별도 Streamlit UI 설명을 참고하세요.
