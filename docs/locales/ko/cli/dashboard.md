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
| `--debug` | `false` | 개발 launcher에서 서버 debug 로깅 선택 |

이 명령은 `truthound.datadocs.launch_dashboard`를 호출합니다. 선택 의존성이 없으면
오류가 발생합니다. 개발 소스의 launcher는 Node/npm으로 비공개 임시 Reflex 프로젝트를
빌드하고 frontend와 backend를 선택한 포트에서 실행합니다. `debug`는 Uvicorn 서버
로깅을 선택합니다. 게시된 3.2.0에는 이 수정이 없으므로 checkout에서
`pip install '.[dashboard]'`로 설치합니다.

정적 출력에는 별도의 `truthound docs generate`를 사용하세요. 프로그램에서 호출하려면
[Python 가이드](../guides/datadocs/dashboard.md)를 참고하세요. 로컬 주소를 유지하고
비민감 입력으로 실행 환경을 먼저 확인해야 합니다. 로컬 Reflex 0.8.25 컴파일과
네이티브 HTTP 수명주기를 테스트했으며 원격 배포는 미검증입니다.
