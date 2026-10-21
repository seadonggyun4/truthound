# Core의 선택형 대화형 인터페이스

Truthound Core에는 프로파일 결과를 탐색하는 선택형 로컬 UI가 있습니다.
HTML·PDF 리포트를 만들거나 MCP 어댑터를 실행하는 데 필요한 기능은 아닙니다.

| 인터페이스 | 의존성 | 진입점 |
| --- | --- | --- |
| Data Docs 프로파일 뷰어 | `dashboard` extra로 설치되는 Reflex | `truthound dashboard` |
| 대화형 profiler | 별도로 설치하는 Streamlit | `truthound.profiler.dashboard.run_dashboard` |

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
host·port·theme·title을 생성한 앱에 전달하지만 나머지 설정을 모두 전달하지는 않습니다.
설정 클래스에 필드가 존재한다는 것만으로 실제 강제 동작이 구현됐다는 뜻은 아닙니다.

## 제약

이번 문서 변경은 현재 Reflex·Streamlit 버전에서의 GUI 시작·호환성을 검증한 것이 아닙니다.
대상 환경에서 확인한 뒤 사용하고 기본적으로 localhost에만 연결하세요. 표시 값·업로드
파일은 민감할 수 있습니다. `require_auth` 설정 필드만으로 인증 경계를 보장하지 않습니다.
정적 리포트 생성은 이 UI 없이 별도로 사용할 수 있습니다.
