# 로컬 프로파일 뷰어 CLI

`truthound dashboard`는 Core의 선택형 Reflex 프로파일 뷰어를 실행합니다.
정적 리포트 생성 명령이나 MCP 서버 실행 명령이 아닙니다. 명령을 실행하는 환경에
Truthound의 `dashboard` extra를 설치해야 합니다.

```bash
truthound dashboard --profile profile.json --host localhost --port 8080
```

| 옵션 | 기본값 | 의미 |
| --- | --- | --- |
| `--profile`, `-p` | 파일 없음 | 기존 프로파일 JSON 입력 |
| `--host` | `localhost` | 요청할 연결 주소 |
| `--port` | `8080` | 요청할 포트 |
| `--title`, `-t` | `Truthound Dashboard` | 페이지 제목 |
| `--debug` | `false` | launcher 설정에 저장 |

이 명령은 `truthound.datadocs.launch_dashboard`를 호출합니다. 선택 의존성이 없으면
오류가 발생합니다. 현재 launcher는 `debug`를 저장하지만 실제 서버 호출에 전달하지
않으므로 유효한 서버 디버그 전환이라고 설명해서는 안 됩니다.

정적 출력에는 별도의 `truthound docs generate`를 사용하세요. 프로그램에서 호출하려면
[Python 가이드](../guides/datadocs/dashboard.md)를 참고하세요. 로컬 주소를 유지하고
비민감 입력으로 실행 환경을 먼저 확인해야 합니다. 명령 signature는 소스로 확인했지만
이번 문서 변경으로 GUI 실행·최신 의존성 호환성이 검증된 것은 아닙니다.
