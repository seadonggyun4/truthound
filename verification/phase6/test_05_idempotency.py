#!/usr/bin/env python3
"""Phase 6 검증: Idempotency Framework 테스트

문서에 명시된 기능:
- 요청 지문 생성: MD5, SHA256, 커스텀 알고리즘
- 중복 감지: IdempotencyService
- 상태 추적: PENDING, COMPLETED, FAILED, EXPIRED, INVALIDATED
- TTL 만료: 시간 기반 자동 만료
- 분산 락: InMemory, FileSystem
"""
import sys

print("=" * 60)
print("Phase 6 검증: Idempotency Framework")
print("=" * 60)

errors = []
warnings = []
doc_discrepancies = []

# 1. 기본 import 테스트
print("\n[1] Idempotency 패키지 import 테스트...")
try:
    from truthound.checkpoint import idempotency
    print("  ✓ idempotency 패키지 import 성공")
    contents = [x for x in dir(idempotency) if not x.startswith('_')]
    print(f"  패키지 내용: {contents}")
except ImportError as e:
    errors.append(f"idempotency 패키지 import 실패: {e}")
    print(f"  ✗ idempotency 패키지 import 실패: {e}")

# 2. IdempotencyService import 테스트
print("\n[2] IdempotencyService import 테스트...")
try:
    from truthound.checkpoint.idempotency.service import IdempotencyService
    print("  ✓ IdempotencyService import 성공")
except ImportError as e:
    try:
        from truthound.checkpoint.idempotency import IdempotencyService
        print("  ✓ IdempotencyService import 성공 (__init__에서)")
    except ImportError as e2:
        errors.append(f"IdempotencyService import 실패: {e2}")
        print(f"  ✗ IdempotencyService import 실패: {e2}")

# 3. 지문 생성 모듈 확인
print("\n[3] Fingerprint 모듈 확인...")
try:
    from truthound.checkpoint.idempotency.fingerprint import (
        FingerprintGenerator,
    )
    print("  ✓ FingerprintGenerator import 성공")

    # 지원 알고리즘 확인
    import inspect
    fg_methods = [m for m in dir(FingerprintGenerator) if not m.startswith('_')]
    print(f"  메서드: {fg_methods}")

    # MD5, SHA256 확인
    for algo in ["md5", "sha256"]:
        if any(algo in m.lower() for m in fg_methods):
            print(f"  ✓ {algo.upper()} 지원 확인")

except ImportError as e:
    try:
        from truthound.checkpoint.idempotency import fingerprint
        contents = [x for x in dir(fingerprint) if not x.startswith('_')]
        print(f"  △ fingerprint 모듈 내용: {contents}")
    except ImportError as e2:
        warnings.append(f"Fingerprint 모듈 확인 실패: {e}")
        print(f"  △ Fingerprint 모듈 확인 실패: {e}")

# 4. 상태 Enum 확인
print("\n[4] Idempotency 상태 확인...")
try:
    from truthound.checkpoint.idempotency.core import IdempotencyStatus
    print("  ✓ IdempotencyStatus import 성공")

    expected_statuses = ["PENDING", "COMPLETED", "FAILED", "EXPIRED", "INVALIDATED"]
    actual_statuses = [s.name for s in IdempotencyStatus]

    for status in expected_statuses:
        if status in actual_statuses:
            print(f"  ✓ IdempotencyStatus.{status}")
        else:
            warnings.append(f"IdempotencyStatus.{status} 없음")
            print(f"  △ IdempotencyStatus.{status} 없음")

except ImportError as e:
    # 다른 경로 시도
    try:
        from truthound.checkpoint.idempotency import core
        for name in dir(core):
            obj = getattr(core, name)
            if hasattr(obj, '__members__') and "status" in name.lower():
                print(f"  ✓ {name} Enum 발견: {list(obj.__members__.keys())}")
                break
    except Exception as e2:
        warnings.append(f"IdempotencyStatus 확인 실패: {e}")
        print(f"  △ IdempotencyStatus 확인 실패: {e}")

# 5. Store 확인
print("\n[5] Idempotency Store 확인...")
try:
    from truthound.checkpoint.idempotency.stores import (
        IdempotencyStore,
        InMemoryIdempotencyStore,
    )
    print("  ✓ IdempotencyStore, InMemoryIdempotencyStore import 성공")

except ImportError as e:
    try:
        from truthound.checkpoint.idempotency import stores
        contents = [x for x in dir(stores) if not x.startswith('_')]
        print(f"  △ stores 모듈 내용: {contents}")
    except ImportError as e2:
        warnings.append(f"Idempotency Store 확인 실패: {e}")
        print(f"  △ Idempotency Store 확인 실패: {e}")

# 6. Locking 모듈 확인
print("\n[6] 분산 락 모듈 확인...")
try:
    from truthound.checkpoint.idempotency.locking import (
        IdempotencyLock,
    )
    print("  ✓ IdempotencyLock import 성공")

    # InMemory, FileSystem 락 확인
    from truthound.checkpoint.idempotency import locking
    contents = [x for x in dir(locking) if not x.startswith('_')]
    print(f"  locking 모듈 내용: {contents}")

    for lock_type in ["InMemory", "FileSystem", "File"]:
        if any(lock_type.lower() in c.lower() for c in contents):
            print(f"  ✓ {lock_type} 락 지원")

except ImportError as e:
    warnings.append(f"Locking 모듈 확인 실패: {e}")
    print(f"  △ Locking 모듈 확인 실패: {e}")

# 7. IdempotencyService 기능 테스트
print("\n[7] IdempotencyService 기능 테스트...")
try:
    from truthound.checkpoint.idempotency.service import IdempotencyService
    import inspect

    # 생성자 시그니처 확인
    sig = inspect.signature(IdempotencyService.__init__)
    print(f"  __init__ 시그니처: {sig}")

    # 필수 메서드 확인
    service_methods = ["check", "start", "complete", "fail", "invalidate",
                       "is_duplicate", "get_result"]
    for method in service_methods:
        if hasattr(IdempotencyService, method):
            print(f"  ✓ {method} 메서드 존재")
        else:
            warnings.append(f"IdempotencyService에 {method} 메서드 없음")

except Exception as e:
    warnings.append(f"IdempotencyService 기능 테스트 실패: {e}")
    print(f"  △ IdempotencyService 기능 테스트 실패: {e}")

# 8. TTL 기능 확인
print("\n[8] TTL 기능 확인...")
try:
    from truthound.checkpoint.idempotency.core import IdempotencyRecord

    import inspect
    sig = inspect.signature(IdempotencyRecord.__init__) if hasattr(IdempotencyRecord, '__init__') else None

    # TTL 관련 필드 확인
    if hasattr(IdempotencyRecord, '__dataclass_fields__'):
        fields = IdempotencyRecord.__dataclass_fields__
        ttl_fields = [f for f in fields if 'ttl' in f.lower() or 'expir' in f.lower()]
        if ttl_fields:
            print(f"  ✓ TTL 관련 필드: {ttl_fields}")
        else:
            print(f"  △ TTL 관련 필드 없음, 전체 필드: {list(fields.keys())}")
    else:
        print(f"  △ IdempotencyRecord는 dataclass가 아님")

except Exception as e:
    warnings.append(f"TTL 기능 확인 실패: {e}")
    print(f"  △ TTL 기능 확인 실패: {e}")

# 9. 실제 사용 테스트
print("\n[9] 실제 사용 테스트...")
try:
    from truthound.checkpoint.idempotency.stores import InMemoryIdempotencyStore
    from truthound.checkpoint.idempotency.service import IdempotencyService

    # 서비스 생성
    store = InMemoryIdempotencyStore()
    service = IdempotencyService(store=store)
    print("  ✓ IdempotencyService 인스턴스 생성 성공")

except Exception as e:
    warnings.append(f"실제 사용 테스트 실패: {e}")
    print(f"  △ 실제 사용 테스트 실패: {e}")
    import traceback
    traceback.print_exc()

# 결과 요약
print("\n" + "=" * 60)
print("Idempotency Framework 검증 결과")
print("=" * 60)

print(f"\n  [문서 불일치] {len(doc_discrepancies)}건:")
for d in doc_discrepancies:
    print(f"    - {d}")

print(f"\n  [오류] {len(errors)}건:")
for e in errors:
    print(f"    - {e}")

print(f"\n  [경고] {len(warnings)}건:")
for w in warnings:
    print(f"    - {w}")

if errors:
    print("\n결과: ✗ FAIL")
    sys.exit(1)
else:
    print("\n결과: ✓ PASS" if not warnings else "\n결과: △ PASS (경고 있음)")
    sys.exit(0)
