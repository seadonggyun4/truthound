# 파일 시스템·메모리 저장소

`FileSystemStore`는 선택한 로컬 디렉터리에 검증 결과를 보존합니다.
`MemoryStore`는 Python 인스턴스 하나의 canonical record mapping을 보관합니다.
여러 thread·process가 같은 파일에 쓰면
`truthound.stores.backends.concurrent_filesystem`의
`ConcurrentFileSystemStore`·`ConcurrentFileSystemExpectationStore`를 사용합니다.
해당 초기화 lock은 프로세스 간 파일 lock을 대체하지 않으므로 그 용도에서는
기본 `auto` 전략을 유지합니다.

## 파일 시스템 저장소

### 기본 사용법

```python
from datetime import datetime
from tempfile import TemporaryDirectory

from truthound.stores.backends.filesystem import FileSystemStore
from truthound.stores.results import ResultStatus, ValidationResult

with TemporaryDirectory() as directory:
    store = FileSystemStore(base_path=directory, compression=True)
    item = ValidationResult(
        run_id="run-123", run_time=datetime(2026, 1, 1),
        data_asset="customers", status=ResultStatus.SUCCESS,
    )
    store.save(item)
    assert store.get("run-123").data_asset == "customers"
```

결과 경로는 `{base_path}/{namespace}/{prefix}`이며 생성자 기본값은
`.truthound/store`·`default`·`validations`입니다. Namespace/prefix는 상대 경로여야
하며 상위 경로 이동·drive 경로는 거부합니다. Run ID는 경로 구성 요소 하나여야
하고 `_index`는 예약되어 있습니다. 읽기는 파일 내부 ID도 확인하며 결과·index
파일의 심볼릭 링크를 허용하지 않습니다.

### 팩토리 사용

```python
from truthound.stores import get_store

store = get_store("filesystem", base_path=".truthound/results")
```

### 설정

개별 생성자 인자 또는 복사해 보관하는 `FileSystemConfig` 중 하나를 사용하고
두 형식을 섞지 않습니다. 알 수 없는 설정은 무시하지 않고 실패합니다. Config의
metadata는 기본값이며 각 결과의 metadata가 우선합니다. 원래 config 객체를 나중에
편집해도 저장소가 보관한 설정을 바꾸지 않습니다.

```python
from truthound.stores.backends.filesystem import FileSystemConfig, FileSystemStore

config = FileSystemConfig(
    base_path=".truthound/results", namespace="production", prefix="validations",
    serialization_format="yaml", file_extension=".yaml",
    use_compression=True, metadata={"environment": "production"},
)
store = FileSystemStore(config=config)
```

| 설정 | 기본값 | 동작 |
|------|--------|------|
| `serialization_format` | `json` | JSON 또는 safe YAML, pickle 거부 |
| `file_extension` | `.json` | 파일 접미사, format을 생략하면 생성자가 `.yaml`/`.yml`에서 YAML 추론 |
| `create_dirs` | `True` | 선택 디렉터리 생성, false이면 기존 디렉터리 필요 |
| `pretty_print` | `True` | JSON 들여쓰기 |
| `compression` | `None` | `gzip`·`zstd` 등 네이티브 코덱 이름 |
| `use_compression` | `False` | 구형 flag, true이고 코덱 이름이 없으면 gzip 선택 |
| `metadata` | `{}` | 결과 metadata 기본값, suite에는 `meta`로 적용 |

Config dataclass의 상속된 `prefix` 기본값은 빈 문자열이므로 config 객체를 쓰면
명시적으로 설정합니다. 각 저장소 생성자의 기본값은 `validations`·`expectations`
입니다. YAML은 safe loader/dumper를 사용하고 지원하지 않는 metadata 값은 쓰기
오류가 됩니다. `file_extension`은 접미사이며 명시한 `serialization_format`이 실제
내용 형식을 결정합니다.

### 압축

`FileSystemStore(compression=True)`는 gzip·`.gz` 접미사를 선택합니다.
`compression="zstd"`는 실제 zstd provider와 `.zst`를 선택하며 그 외 네이티브
코덱은 이름을 추가 접미사로 사용합니다. 선택 provider가 설치되어 있어야 합니다.
읽기는 자동으로 압축을 해제하고 JSON/YAML을 파싱합니다. 다시 열 때
같은 format/codec 설정이 필요하며 설정 변경 자체는 형식 migration이 아닙니다.

### 디렉터리 구조

```text
{base_path}/{namespace}/{prefix}/
    _index.json
    run-123.json
    run-456.yaml.gz
```

서로 다른 설정의 파일 예시이며 각 저장소는 선택한 접미사 하나를 사용합니다.
다른 설정은 별도 디렉터리에 두십시오.

### 인덱스 관리

결과 파일·`_index.json`은 각각 비공개 형제 임시 파일을 flush·sync한 뒤 원자적으로
교체합니다. 두 파일을 한 번에 확정하는 transaction은 아닙니다. 새 index는 metadata와
결과 SHA-256을 포함합니다. 해시 없는 구형 index도 metadata가 맞으면 읽습니다.
초기화는 결과·index 일치를 검증하므로 결과 파일도 읽고 index만 로드하는 것보다
비용이 듭니다.

손상된 결과·index, 누락된 결과, ID 불일치·내용 변경은 오류입니다. Query는 손상된
결과를 조용히 제외하지 않습니다. 결과/index 쓰기가 확인되지 않으면 해당 소유자의
후속 변경을 차단합니다. 새 소유자는 실제 일치를 확인합니다. 복구는 명시적입니다.

```python
from truthound.stores.backends.filesystem import FileSystemStore

store = FileSystemStore(base_path=".truthound/results")
# index 자체가 손상되었다면 initialize부터 호출하지 않습니다.
indexed = store.rebuild_index()
```

`rebuild_index`는 선택된 모든 결과를 검증한 뒤 index를 교체합니다. 결과가 손상되면
복구를 멈추고 이전 index를 보존합니다. 해당 결과를 운영자가 통제하는 절차로
수정·복원한 뒤 재시도하십시오. 재구축은 현재 파일을 조정할 뿐 이미 교체된 결과를
rollback하거나 exactly-once 효과를 증명하지 않습니다. 일반 파일 저장소는 공동
writer를 조정하지 않으므로 그 용도에는 concurrent backend를 사용합니다.

### Expectation 저장소

`FileSystemExpectationStore`도 같은 config 객체·safe format·압축·원자적 게시를
지원합니다. Suite 이름을 파일명 구성 요소로 인코딩하여 `a/b`·`a_b`를 구별합니다.
단순한 구형 이름은 경로를 유지합니다. `/`를 `_`로 바꾼 구형 파일은 내부 suite
이름이 일치할 때만 읽으며, 충돌하는 구형 파일명은 명시적으로 migration해야 합니다.

```python
from truthound.stores.backends.filesystem import FileSystemExpectationStore
from truthound.stores.expectations import ExpectationSuite

store = FileSystemExpectationStore(
    base_path=".truthound/suites", use_compression=True,
)
store.save(ExpectationSuite.create("customer_suite", "customers"))
assert store.get("customer_suite").data_asset == "customers"
```

## 메모리 저장소

### 기본 사용법

```python
from datetime import datetime
from truthound.stores.backends.memory import MemoryStore
from truthound.stores.results import ResultStatus, ValidationResult

store = MemoryStore(max_items=1)
for identity in ("first", "second"):
    store.save(ValidationResult(identity, datetime(2026, 1, 1), "customers", ResultStatus.SUCCESS))
assert store.list_ids() == ["second"]
```

### 설정

`MemoryConfig(max_items=0, deep_copy=True, metadata={})`를
`MemoryStore(config=config)`·`MemoryExpectationStore(config=config)`로 전달할 수
있습니다. 용량 0은 무제한이며 음수·boolean 용량, boolean이 아닌 복사 flag와 알 수
없는 설정은 실패합니다. 메모리 저장소는 canonical mapping을 보관하며 byte 압축·
비 JSON 직렬화 설정을 거부하고 pickle을 사용하지 않습니다. 각 인스턴스가 데이터를
소유하며 context 종료가 다른 위치에 데이터를 보존하지는 않습니다.

### 메모리 관리

용량이 차면 새 ID 삽입 시 기존 결과 중 run time이 가장 이른 항목을 제거합니다.
Suite는 update time을 사용하고 동률은 ID로 결정합니다. 기존 ID 갱신은 다른 결과를
제거하지 않습니다. 인스턴스 lock으로 작업을 보호하되 프로세스끼리 상태를 공유하지
않습니다. `clear_all()`은 모든 항목을 명시적으로 제거합니다.

### Deep copy 동작

`deep_copy=True`는 결과 본문과 필터용 metadata를 같은 복사본으로 보관합니다.
원본이나 조회한 결과의 중첩 태그 변경은 이후 query를 바꾸지 않습니다.
`deep_copy=False`의 재구성된 결과는 보관 mapping의 중첩 값을 공유할 수 있어 이를
편집하면 이후 읽기가 바뀔 수 있습니다. 결과 객체 자체는 재구성되므로 임의의 최상위
속성 대입이 보관 mapping을 바꾼다는 뜻은 아닙니다.

### Expectation 저장소

`MemoryExpectationStore`는 이름 있는 `ExpectationSuite`에 같은 용량·config·복사·
metadata·필터·페이지 처리를 제공합니다. 데이터 수명도 인스턴스에 한정됩니다.

## 공통 작업

검증 저장소 둘 다 save/get/exists/delete/list/query와 상속된 history/latest 기능을
제공합니다. `StoreQuery`는 data asset·status·tags·시간 필터, 선택 `order_by`
(`run_time`, `data_asset`, `status`)·방향·offset·limit을 적용합니다. `limit=0`은
빈 결과이며 잘못된 페이지·정렬 설정은 실패합니다. Suite query는 사용 가능한
metadata와 페이지 처리를 적용합니다.

## 오류 처리

`StoreNotFoundError`는 없는 결과, `StoreReadError`는 잘못된 파일/index,
`StoreWriteError`는 게시 실패·필수 조정을 나타냅니다. 실패 파일을 보존하고
불명확한 쓰기를 조사하십시오. 오류 반환만으로 직전 파일 교체가 없었다고 증명할
수는 없습니다. 미배포 MCP Core 프로파일에서는 승인된 `StoreRuntime` 팩토리가
설정을 소유하며 `rebuild_index`로 네이티브 복구를 선택합니다. 새 메모리 소유자는
이전 소유자의 결과를 갖지 않으며 민감 정보 공개 정책은 계속 적용됩니다.

## 저장소 선택

임시 작업·테스트는 memory, 단일 writer 영구 결과는 filesystem, 조정된 공동 쓰기는
concurrent filesystem 저장소를 사용합니다.

## 다음 단계

- [클라우드 저장소](cloud-storage.md)
- [버전 관리](versioning.md)
- [캐싱](caching.md)

Suite 쿼리는 `run_time`(`updated_at` 사용), `data_asset`, `name` 정렬과 방향을 적용한 뒤 페이지를 나눕니다. 쿼리가 있으면 기본값은 최근 수정 순이며, 쿼리가 없으면 이름 오름차순입니다. JSON 기록과 인덱스는 중복 키와 유한하지 않은 숫자를 거부하며 명시적 인덱스 재구축에도 같은 검증을 적용합니다.

JSON과 safe YAML은 모두 중복 매핑 키와 유한하지 않은 숫자를 거부합니다. 게시할 때는 기록을 교체하기 전에 순환 컨테이너 메타데이터와 64단계를 넘는 중첩을 거부합니다. 읽기 실패 시 조사할 원본 파일을 유지하며 명시적 인덱스 재구축도 모호한 기록을 받아들이지 않습니다. 중복 매핑을 만드는 YAML merge 키도 거부합니다.

## 동시 파일 저장소 설정

두 concurrent 저장소는 중첩 동시성 설정과 메타데이터를 복사합니다. 알 수 없는
생성자 옵션, 잘못된 모드 플래그·경로·잠금 제한 시간은 무시하지 않고 거부합니다.
`config` 속성도 복사본을 반환합니다. `create_dirs=False`이면 선택 디렉터리 전체가
이미 존재해야 하며 인덱스 초기화를 통해 간접 생성하지 않습니다. 기대값 저장소도
검증 결과 저장소와 동일한 `auto`, `fcntl`, `filelock`, `portalocker`, `none`
잠금 전략을 적용합니다. `none`에는 프로세스 간 상호 배제가 없습니다.

두 저장소는 일반 파일 저장소의 JSON·안전한 YAML·압축 코덱을 공유합니다.
YAML/gzip은 `serialization_format="yaml"`, `file_extension=".yaml"`과 함께
기대값 저장소의 `use_compression=True` 또는 검증 결과 저장소의
`compression=True`로 선택합니다. `compression`으로 등록된 코덱 이름도
선택할 수 있습니다. 설정 메타데이터는 검증 결과의 `metadata`, suite의 `meta`
기본값이며 항목 값이 우선합니다. 비유한 값·순환 값은 레코드 교체 전에 거부합니다.
중복 JSON/YAML 키와 잘못된 레코드는 읽기·조회 오류로 보존합니다. 잘못된 레코드가
있으면 인덱스 재구축을 중단하고 이전 인덱스를 유지합니다. 인덱스·WAL의 JSON도
중복 키와 비유한 수를 거부합니다.

이름은 단일 파일명으로 percent-encoding하므로 `a/b`, `a_b`, `a%b`가 충돌하거나
선택 디렉터리를 벗어나지 않습니다. 내부 인덱스·배치 소유자·복구 이름 접두사는
예약되며 레코드 심볼릭 링크는 거부됩니다. 기존 중첩 검증 파일명과 밑줄 치환
suite 파일명은 레코드 내부 식별자를 확인한 뒤 인코딩된 파일명으로 명시적으로
이전해야 합니다. 추측하여 자동 채택하지 않습니다. 일반적인 변경 없는 이름은
기존 파일명을 유지하고 명시적 작업 전까지 기존 파일을 변경하지 않습니다.

조회는 정렬·페이지 처리를 한 번 적용하며 `limit=0`은 빈 결과입니다. 검증 결과는
`run_time`, `data_asset`, `status`, suite는 `run_time`(갱신 시각), `data_asset`,
`name` 정렬을 지원합니다. 갱신 시각이 없는 기존 suite 인덱스는 다시 저장하기
전까지 기록된 생성 시각을 사용합니다. 레코드와 인덱스 게시는 여전히 별개이며
다중 파일 배치가 crash-atomic 트랜잭션으로 바뀌지는 않습니다. 불명확한 커밋은
명시적 배치 복구 절차로 처리하세요.


모든 인덱스 작업을 첫 WAL 기록 전에 검증합니다. 뒤쪽 메타데이터 오류가 앞쪽 작업만 복구 가능한 부분 기록으로 남기지 않습니다. 실제 쓰기 장애는 여전히 결과 불명 처리가 필요합니다.

동시 인덱스와 WAL은 선택 디렉터리 안의 일반 파일이어야 하며 존재하거나 대상이 없는 심볼릭 링크를 거부합니다. `O_NOFOLLOW` 지원 플랫폼에서는 WAL 추가 기록에도 적용합니다. 잠금은 협력하는 writer를 조정하며 관련 없는 경로를 변경할 권한을 부여하지 않습니다.
