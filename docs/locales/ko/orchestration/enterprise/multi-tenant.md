---
title: Multi-Tenant
---

# Multi-Tenant

`truthound.multitenancy`는 공유 배포의 tenant 관리, 명시적 문맥, quota 추적과
데이터 격리 전략을 제공합니다.

## Main Components

- `TenantStore`·resolver·quota tracker를 사용하는 `TenantManager`
- `TenantContext`와 `TenantManager.context(tenant_id)`
- `IsolationManager`·`SharedIsolation`·`RowLevelIsolation`
- schema/database namespace 보조 기능과 middleware 연결

## When To Use It

하나의 배포에서 여러 고객·조직을 처리하며 소유권·quota·데이터 소스 경계가 필요할 때
사용합니다. tenant metadata나 문맥에 들어간 사실만으로 다른 소스를 읽을 권한이 생기지 않습니다.

## Recommended Operating Model

선택한 저장소에서 tenant 정의를 관리하고 작업 주위에 문맥을 설정하며 실제 행 필터에
활성 tenant를 전달합니다. callback 정책은 신뢰하는 애플리케이션 코드로 구성하고
정책 이름은 중복하지 않습니다.

```python
import polars as pl
from truthound.multitenancy import (
    IsolationLevel, IsolationManager, RowLevelIsolation, RowLevelPolicy, TenantManager,
)

rows = RowLevelIsolation()
rows.add_policy(RowLevelPolicy("positive", lambda row, tenant: row["value"] > 0))
isolation = IsolationManager()
isolation.register(IsolationLevel.ROW_LEVEL, rows)
manager = TenantManager(isolation_manager=isolation)
tenant = manager.create("Example", tenant_id="example")
frame = pl.DataFrame({"tenant_id": ["example", "other", "example"], "value": [-1, 9, 2]})
filtered = manager.apply_isolation(frame, tenant).select("value").collect()
assert filtered["value"].to_list() == [2]
```

## Isolation Levels

shared 격리는 tenant ID 컬럼으로 거릅니다. row-level 격리는 먼저 네이티브 tenant
조건을 적용한 뒤 활성 정책을 우선순위 내림차순으로 평가합니다. 정책은 실제 bool을
반환해야 하며 truthy 문자열·정수는 접근을 허용하지 않습니다. 기본
`fail_closed=True`에서는 tenant 컬럼이 없으면 빈 결과를 반환하고 정책 오류는 해당
행을 거부합니다. `fail_closed=False`에서는 tenant 컬럼이 없으면 그대로 통과시키며
정책 오류는 원문을 제외한 `TenantIsolationError`로 전달합니다.

행 정책 평가는 지연 실행하며 schema를 보존합니다. projection·slice pushdown이 정책에
필요한 컬럼을 없애거나 정책 평가 전에 행을 제한하지 않습니다. batch를 수집할 때 tenant
상태를 다시 확인하며 필터 계획마다 정책 설정을 복사합니다. `policies`도 설정 복사본을 반환합니다.

schema·database 전략은 애플리케이션이 이미 올바른 tenant 소스를 골랐다고 가정합니다.
namespace·문맥 검사만으로 임의의 혼합 tenant dataframe 출처를 입증하지는 않습니다.
instance 격리는 이 모듈의 runtime 범위 밖입니다.

## Practical Guidance

선택형 MCP tenant 서비스는 별도로 승인된 자원 handle의 shared/row-level 필터와
factory 정책 별칭의 목록·추가·제거·활성화·우선순위 편집을 제공합니다. 모델 요청에서
Python 식·실제 tenant ID·schema/database 연결을 받지 않습니다. 임의 입력 자원의
schema/database 격리는 지원하지 않습니다. 행 정책 편집은 소유자 내부 설정이며 tenant
저장소에 영속화되는 데이터가 아닙니다. 입력 자원을 해제하면 해당 서비스와 파생 결과도 철회합니다.

로컬 검사는 실제 Polars 행 필터와 SQLite tenant metadata를 확인합니다. 원격 저장소·
resolver·분산 quota tracker는 별도 검증이 필요합니다.

## Related Pages

- [MCP tenant 작업](../../guides/mcp.md)
- [시크릿](secrets.md)
- [CI/CD와 운영 배포](ci-cd-production.md)
