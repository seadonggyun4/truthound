# Truthound 3.1.7 릴리스 노트

## 핵심 변경

Truthound 3.1.7은 Truthound Depot의 mixed-asset 운영 gate에서 발견된 structured
file 계약 결함을 복구합니다. 완전한 JSON 문서는 root API와 `FileDataSource`에서
동일한 의미로 처리되고, 중첩 JSON 열은 hashability 오류 없이 자동 drift 비교에
참여합니다.

## 완전한 JSON 문서 계약

`.json`은 첫 문자로 NDJSON 여부를 추정하지 않고 하나의 완전한 문서로 파싱합니다.

- 객체 배열은 객체 하나당 한 행을 생성합니다.
- 최상위 객체는 필드를 포함하는 한 행을 생성합니다.
- 스칼라 배열은 `value` 열의 여러 행을 생성합니다.
- 최상위 스칼라는 `value` 열의 한 행을 생성합니다.
- `.ndjson`, `.jsonl`은 별도의 line-delimited lazy scan 형식을 유지합니다.

이 변경으로 `{`로 시작하는 정상 최상위 객체를 NDJSON scanner로 보내던 오류를
제거했습니다. JSON 배열을 읽을 때 관리되지 않는 임시 NDJSON 파일도 생성하지
않습니다.

## 중첩 Drift 비교

`truthound.drift.compare(..., method="auto")`는 `Struct`, `List` 값을 categorical
detector와 통계 계산 전에 결정론적 JSON 범주로 정규화합니다. 객체 key 순서를
정규화하고 null은 null로 유지하며, `ColumnDrift.dtype`에는 원래 Polars dtype을
계속 제공합니다.

`ks`, `psi` 같은 명시적 수치 방식은 여전히 수치 열만 지원합니다.

## 소비자 업그레이드 Gate

Truthound Depot 같은 소비자는 공개 배포된 3.1.7 wheel을 설치하고
`truthound.__version__`을 확인한 뒤 profile, validation, drift, anomaly,
serialization 또는 재진입, mixed-asset lifecycle을 다시 검증해야 합니다. source
checkout이나 배포되지 않은 wheel은 소비자 인증 증거가 아닙니다.
