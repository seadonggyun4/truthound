# Truthound 3.3.1

이 패치는 로컬 MCP tool 설명과 JSON Schema 필드 설명을 개선합니다.
`check_file`, `profile_file`, `scan_pii` 이름은 유지합니다. 검증 기능,
디렉터리 승인, 원문 공개 권한, 네트워크 전송을 추가하지 않습니다.
결측·중복 검사, 타입·집계 프로파일링, 표본 기반 개인정보 탐지 범위는 같습니다.

## 설치와 게시

2026-11-02 Core 소스 `80f8092d30918c825154e3ff8e5ed3ca4c328b17`에서 게시했습니다. [같은 소스 전체 검증](https://github.com/seadonggyun4/truthound/actions/runs/41371907329)과 [PyPI 게시](https://github.com/seadonggyun4/truthound/actions/runs/41378219376)가 통과했습니다. [PyPI 3.3.1](https://pypi.org/project/truthound/3.3.1/)과 [공식 Registry의 active/latest 기록](https://registry.modelcontextprotocol.io/v0.1/servers/io.github.seadonggyun4%2Ftruthound/versions/latest)을 따로 확인했습니다. 정식 wheel/sdist 해시는 PyPI와 일치하며 GitHub v3.3.1 릴리스에도 같은 파일을 보존합니다.

```bash
python -m pip install "truthound[mcp]==3.3.1"
```

Local plugin 0.1.7은 게시된 Core 3.3.1을 frozen lock으로 고정합니다.
별도 기존 Core plugin은 0.2.1입니다. Core 산출물의 소스 커밋과 plugin 의존성
갱신 커밋을 구분해 기록합니다. Styxis 공개 catalog는 Local `truthound` 하나이며
새 제품이나 공급자 directory listing을 추가하지 않습니다.

## 검증과 한계

같은 커밋의 전체 테스트, 네이티브 호환성, 한영 문서, wheel/sdist 출처 검증을
유지합니다. metadata 개선만으로 Agent의 실제 선택 정확성이 증명되지는 않습니다.
client/model 관측을 따로 기록하며 모든 Agent의 발견을 보장하지 않습니다.
Local summary tool 설치는 데이터 접근을 승인하지 않고 원문을 반환하지 않습니다.
경로·열 이름·집계는 host 모델에 전달될 수 있습니다.

실제 Agent 선택 검증은 기존 Codex·Claude CLI로 시작을 시도했습니다.
Codex 모델 요청은 `model_not_found`, Claude는 조직의 subscription 접근 금지로
거절됐습니다. 선택 관측은 0건이고 정확도·오선택률은 미측정입니다.
이는 도구 선택 오류가 아닌 접근 제한이며 키·계정 추가나 정책 우회는 하지 않았습니다.

최초 macOS/Python 3.13 CI 시도의 native AI 제안 검증 오류는 로컬에서 재현되지 않았으며 원인은 미확정입니다. 같은 소스의 제한된 재실행이 기존 전체 게이트를 통과한 뒤 게시했습니다. 최초 실패 ZIP과 재실행 기록은 [배포 검증 기록](https://github.com/seadonggyun4/styxis-plugin/blob/main/evaluation/results/release-3.3.1/ci-retry.json)에 보존합니다. 테스트·시간 제한·권한 검사를 완화하지 않았습니다.
