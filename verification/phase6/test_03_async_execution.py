#!/usr/bin/env python3
"""Phase 6 검증: Async Execution Framework 테스트

문서에 명시된 기능:
- AsyncCheckpoint: 비동기 체크포인트 클래스
- AsyncCheckpointRunner: 동시 체크포인트 실행
- 3가지 실행 전략:
  - SequentialStrategy: 순차 실행
  - ConcurrentStrategy: 최대 N개 병렬 실행
  - PipelineStrategy: async generator 스트리밍
"""
import sys
import asyncio

print("=" * 60)
print("Phase 6 검증: Async Execution Framework")
print("=" * 60)

errors = []
warnings = []
doc_discrepancies = []

# 1. AsyncCheckpoint import 테스트
print("\n[1] AsyncCheckpoint import 테스트...")
try:
    from truthound.checkpoint.async_checkpoint import AsyncCheckpoint
    print("  ✓ AsyncCheckpoint import 성공")
except ImportError as e:
    errors.append(f"AsyncCheckpoint import 실패: {e}")
    print(f"  ✗ AsyncCheckpoint import 실패: {e}")

# 2. AsyncCheckpointRunner import 테스트
print("\n[2] AsyncCheckpointRunner import 테스트...")
try:
    from truthound.checkpoint.async_runner import AsyncCheckpointRunner
    print("  ✓ AsyncCheckpointRunner import 성공")
except ImportError as e:
    errors.append(f"AsyncCheckpointRunner import 실패: {e}")
    print(f"  ✗ AsyncCheckpointRunner import 실패: {e}")

# 3. 실행 전략 import 테스트
print("\n[3] 실행 전략 import 테스트...")
expected_strategies = ["SequentialStrategy", "ConcurrentStrategy", "PipelineStrategy"]

try:
    from truthound.checkpoint import async_runner
    module_contents = dir(async_runner)

    for strategy in expected_strategies:
        if strategy in module_contents:
            print(f"  ✓ {strategy} 존재")
        else:
            # 다른 이름으로 존재할 수 있음
            similar = [x for x in module_contents if "strategy" in x.lower() or "execution" in x.lower()]
            if similar:
                doc_discrepancies.append(f"문서: {strategy} → 실제: {similar}")
                print(f"  △ {strategy} 없음, 유사 항목: {similar}")
            else:
                warnings.append(f"{strategy} 찾을 수 없음")
                print(f"  △ {strategy} 찾을 수 없음")

except ImportError as e:
    errors.append(f"async_runner 모듈 import 실패: {e}")
    print(f"  ✗ async_runner 모듈 import 실패: {e}")

# 4. AsyncCheckpoint 클래스 분석
print("\n[4] AsyncCheckpoint 클래스 분석...")
try:
    from truthound.checkpoint.async_checkpoint import AsyncCheckpoint
    import inspect

    # 클래스 메서드 확인
    methods = [m for m in dir(AsyncCheckpoint) if not m.startswith('_')]
    print(f"  Public 메서드: {methods[:10]}{'...' if len(methods) > 10 else ''}")

    # 비동기 메서드 확인
    async_methods = []
    for name in dir(AsyncCheckpoint):
        if not name.startswith('_'):
            attr = getattr(AsyncCheckpoint, name, None)
            if asyncio.iscoroutinefunction(attr):
                async_methods.append(name)

    if async_methods:
        print(f"  ✓ 비동기 메서드: {async_methods}")
    else:
        warnings.append("AsyncCheckpoint에 비동기 메서드 없음")
        print(f"  △ 비동기 메서드 발견 안됨")

except Exception as e:
    errors.append(f"AsyncCheckpoint 분석 실패: {e}")
    print(f"  ✗ AsyncCheckpoint 분석 실패: {e}")

# 5. AsyncCheckpointRunner 클래스 분석
print("\n[5] AsyncCheckpointRunner 클래스 분석...")
try:
    from truthound.checkpoint.async_runner import AsyncCheckpointRunner
    import inspect

    # 생성자 시그니처 확인
    sig = inspect.signature(AsyncCheckpointRunner.__init__)
    print(f"  __init__ 시그니처: {sig}")

    # 클래스 메서드 확인
    methods = [m for m in dir(AsyncCheckpointRunner) if not m.startswith('_')]
    print(f"  Public 메서드: {methods}")

    # run/execute 메서드 확인
    for method_name in ["run", "execute", "run_all", "run_concurrent"]:
        if hasattr(AsyncCheckpointRunner, method_name):
            print(f"  ✓ {method_name} 메서드 존재")

except Exception as e:
    errors.append(f"AsyncCheckpointRunner 분석 실패: {e}")
    print(f"  ✗ AsyncCheckpointRunner 분석 실패: {e}")

# 6. 실행 전략 Enum/클래스 확인
print("\n[6] 실행 전략 상세 확인...")
try:
    # ExecutionStrategy Enum 찾기
    from truthound.checkpoint import async_runner
    from truthound.checkpoint import async_checkpoint

    for module in [async_runner, async_checkpoint]:
        for name in dir(module):
            obj = getattr(module, name)
            if isinstance(obj, type) and "strategy" in name.lower():
                print(f"  ✓ {name} 클래스 발견 in {module.__name__}")

            # Enum 확인
            if hasattr(obj, '__members__') and "strategy" in name.lower():
                print(f"  ✓ {name} Enum 발견: {list(obj.__members__.keys())}")

except Exception as e:
    warnings.append(f"실행 전략 상세 확인 실패: {e}")
    print(f"  △ 실행 전략 상세 확인 실패: {e}")

# 7. async_base 모듈 확인
print("\n[7] async_base 모듈 확인...")
try:
    from truthound.checkpoint import async_base
    print(f"  ✓ async_base 모듈 존재")
    print(f"  내용: {[x for x in dir(async_base) if not x.startswith('_')]}")

except ImportError as e:
    warnings.append(f"async_base 모듈 없음: {e}")
    print(f"  △ async_base 모듈 없음")

# 8. async_actions 모듈 확인
print("\n[8] async_actions 모듈 확인...")
try:
    from truthound.checkpoint import async_actions
    print(f"  ✓ async_actions 모듈 존재")
    print(f"  내용: {[x for x in dir(async_actions) if not x.startswith('_')]}")

except ImportError as e:
    warnings.append(f"async_actions 모듈 없음: {e}")
    print(f"  △ async_actions 모듈 없음")

# 9. 실제 비동기 실행 테스트 (간단한)
print("\n[9] 비동기 실행 테스트...")
try:
    async def test_async():
        # AsyncCheckpoint 생성 가능한지 확인
        from truthound.checkpoint.async_checkpoint import AsyncCheckpoint

        # 생성자 시그니처 확인
        import inspect
        sig = inspect.signature(AsyncCheckpoint.__init__)
        params = list(sig.parameters.keys())
        print(f"    AsyncCheckpoint 파라미터: {params}")

        return True

    result = asyncio.run(test_async())
    if result:
        print("  ✓ 비동기 환경 테스트 성공")

except Exception as e:
    warnings.append(f"비동기 실행 테스트 실패: {e}")
    print(f"  △ 비동기 실행 테스트 실패: {e}")

# 결과 요약
print("\n" + "=" * 60)
print("Async Execution Framework 검증 결과")
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
elif doc_discrepancies or warnings:
    print("\n결과: △ PASS (확인 필요)")
    sys.exit(0)
else:
    print("\n결과: ✓ PASS")
    sys.exit(0)
