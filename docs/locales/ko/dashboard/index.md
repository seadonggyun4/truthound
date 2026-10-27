# Core의 선택형 대화형 인터페이스

Truthound Core에는 프로파일 결과를 탐색하는 선택형 로컬 UI가 있습니다.
HTML·PDF 리포트를 만들거나 MCP 어댑터를 실행하는 데 필요한 기능은 아닙니다.

| 인터페이스 | 의존성 | 진입점 |
| --- | --- | --- |
| Data Docs 프로파일 뷰어 | `dashboard` extra로 설치되는 Reflex | `truthound dashboard` |
| 대화형 profiler | 개발 소스의 `dashboard-streamlit` extra 또는 별도 Streamlit | `truthound.profiler.dashboard.run_dashboard` |

## Data Docs 프로파일 뷰어

```bash
pip install "truthound[dashboard]==3.2.0"
```

[CLI 설명](../cli/dashboard.md)과 [Python 가이드](../guides/datadocs/dashboard.md)를
참고하세요. 기본 주소는 `localhost`, 기본 포트는 8080입니다.

## 대화형 profiler

`dashboard` extra는 Reflex용이며 Streamlit을 설치하지 않습니다. 별도 UI를 사용하려면:

```bash
pip install "truthound==3.2.0" streamlit
```

```python
from truthound.profiler.dashboard import DashboardConfig, run_dashboard

run_dashboard(DashboardConfig(host="localhost", port=8501))
```

파일 선택·업로드, 프로파일 조회와 내보내기 컨트롤이 구현되어 있습니다. 현재 launcher는
모든 dataclass 설정을 비공개 JSON으로 정적 앱에 전달합니다. 제목은 실행 코드로 삽입하지 않습니다.
설정 클래스에 필드가 존재한다는 것만으로 실제 강제 동작이 구현됐다는 뜻은 아닙니다.

## 제약

개발 소스의 로컬 테스트는 Streamlit 시작·정리·제목 렌더링과 Reflex 0.8.25 페이지
컴파일·실제 HTTP 수명주기·소유자 정리를 검증합니다. 원격 배포 호환성은 미검증이며
기본적으로 localhost에만 연결하세요. 표시 값·업로드
파일은 민감할 수 있습니다. `require_auth` 설정 필드만으로 인증 경계를 보장하지 않습니다.
정적 리포트 생성은 이 UI 없이 별도로 사용할 수 있습니다.

## 개발 소스의 관리형 프로세스

개발 checkout에서 `pip install '.[dashboard-streamlit]'`로 선택 extra를 설치합니다. `truthound.profiler.dashboard.server.DashboardServer`는 `start()`·`status()`·
`stop()`·`close()`로 실제 Streamlit 프로세스를 관리합니다. 준비 상태와 프로세스 생존을
분리하며 부모 연결 종료 시 자식도 종료합니다. start의 기본 제한 시간은 10초이고
최대 30초입니다. 설정은 정적 진입점에 JSON으로 전달하며 native UI의 인증 보장은
추가하지 않습니다. MCP 제어는 [가이드](../guides/mcp.md#네이티브-loopback-대시보드-수명주기)를
참조하세요. 이 기능은 게시된 3.2.0에 아직 포함되지 않습니다. 로컬 실제 서버·렌더링
회귀가 있습니다. `truthound.datadocs.dashboard.ReflexDashboardServer`는 준비된
비공개 Reflex 프로젝트도 하나의 HTTP 포트에서 관리합니다. 명시적 `prepare()`는
Node/npm과 개발 `dashboard` extra가 필요하며 준비된 서버 시작은 의존성을 설치하지
않습니다. [준비 가이드](../guides/datadocs/dashboard.md)를 참고하세요.
