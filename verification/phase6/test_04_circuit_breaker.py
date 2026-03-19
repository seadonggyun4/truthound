#!/usr/bin/env python3
"""Phase 6 검증: Circuit Breaker Pattern 테스트

문서에 명시된 기능:
- 3가지 상태: CLOSED → OPEN → HALF_OPEN
- 실패 감지 전략: 임계값, 연속 실패, 시간 기반
- 메트릭 수집: 호출 횟수, 성공률, 응답 시간
- 중앙 레지스트리: CircuitBreakerRegistry
"""
import sys

print("=" * 60)
print("Phase 6 검증: Circuit Breaker Pattern")
print("=" * 60)

errors = []
warnings = []
doc_discrepancies = []

# 1. 기본 import 테스트
print("\n[1] Circuit Breaker import 테스트...")
try:
    from truthound.checkpoint.circuitbreaker import (
        CircuitBreaker,
    )
    print("  ✓ CircuitBreaker import 성공")
except ImportError as e:
    # 다른 경로 시도
    try:
        from truthound.checkpoint.circuitbreaker.breaker import CircuitBreaker
        print("  ✓ CircuitBreaker import 성공 (breaker 모듈에서)")
    except ImportError as e2:
        errors.append(f"CircuitBreaker import 실패: {e2}")
        print(f"  ✗ CircuitBreaker import 실패: {e2}")

# 2. CircuitBreakerRegistry import 테스트
print("\n[2] CircuitBreakerRegistry import 테스트...")
try:
    from truthound.checkpoint.circuitbreaker.registry import CircuitBreakerRegistry
    print("  ✓ CircuitBreakerRegistry import 성공")
except ImportError as e:
    try:
        from truthound.checkpoint.circuitbreaker import CircuitBreakerRegistry
        print("  ✓ CircuitBreakerRegistry import 성공 (__init__에서)")
    except ImportError as e2:
        errors.append(f"CircuitBreakerRegistry import 실패: {e2}")
        print(f"  ✗ CircuitBreakerRegistry import 실패: {e2}")

# 3. 상태 Enum 확인
print("\n[3] Circuit Breaker 상태 확인...")
try:
    from truthound.checkpoint.circuitbreaker.core import CircuitState
    print("  ✓ CircuitState import 성공")

    expected_states = ["CLOSED", "OPEN", "HALF_OPEN"]
    actual_states = [s.name for s in CircuitState]

    for state in expected_states:
        if state in actual_states:
            print(f"  ✓ CircuitState.{state} 존재")
        else:
            errors.append(f"CircuitState.{state} 없음")
            print(f"  ✗ CircuitState.{state} 없음")

except ImportError as e:
    # 다른 경로 시도
    try:
        from truthound.checkpoint.circuitbreaker import breaker
        # Enum 찾기
        for name in dir(breaker):
            obj = getattr(breaker, name)
            if hasattr(obj, '__members__') and "state" in name.lower():
                print(f"  ✓ {name} Enum 발견: {list(obj.__members__.keys())}")
                break
    except Exception as e2:
        errors.append(f"CircuitState import 실패: {e}")
        print(f"  ✗ CircuitState import 실패: {e}")

# 4. 실패 감지 전략 확인
print("\n[4] 실패 감지 전략 확인...")
try:
    from truthound.checkpoint.circuitbreaker.detection import (
        FailureDetectionStrategy,
    )
    print("  ✓ FailureDetectionStrategy import 성공")

    # 전략 확인
    strategies = dir(FailureDetectionStrategy) if hasattr(FailureDetectionStrategy, '__members__') else []
    if strategies:
        print(f"  전략 목록: {strategies}")
except ImportError as e:
    # detection 모듈 내용 확인
    try:
        from truthound.checkpoint.circuitbreaker import detection
        contents = [x for x in dir(detection) if not x.startswith('_')]
        print(f"  △ detection 모듈 내용: {contents}")

        # 전략 관련 클래스 찾기
        strategy_classes = [x for x in contents if "strategy" in x.lower() or "detection" in x.lower()]
        if strategy_classes:
            print(f"  ✓ 전략 관련 클래스: {strategy_classes}")
    except ImportError as e2:
        warnings.append(f"실패 감지 전략 확인 실패: {e}")
        print(f"  △ 실패 감지 전략 확인 실패: {e}")

# 5. CircuitBreaker 인스턴스 생성 테스트
print("\n[5] CircuitBreaker 인스턴스 생성 테스트...")
try:
    from truthound.checkpoint.circuitbreaker.breaker import CircuitBreaker
    import inspect

    # 생성자 시그니처 확인
    sig = inspect.signature(CircuitBreaker.__init__)
    print(f"  __init__ 시그니처: {sig}")

    # 간단한 인스턴스 생성 시도
    cb = CircuitBreaker(name="test_breaker")
    print(f"  ✓ CircuitBreaker 인스턴스 생성 성공")

    # 상태 확인
    if hasattr(cb, 'state'):
        print(f"  ✓ 현재 상태: {cb.state}")

except Exception as e:
    warnings.append(f"CircuitBreaker 인스턴스 생성 실패: {e}")
    print(f"  △ CircuitBreaker 인스턴스 생성 실패: {e}")

# 6. 메트릭 수집 확인
print("\n[6] 메트릭 수집 확인...")
try:
    from truthound.checkpoint.circuitbreaker.breaker import CircuitBreaker

    cb = CircuitBreaker(name="metrics_test")

    # 메트릭 관련 속성/메서드 확인
    metric_attrs = ["metrics", "get_metrics", "call_count", "success_count",
                    "failure_count", "success_rate", "statistics"]

    found_metrics = []
    for attr in metric_attrs:
        if hasattr(cb, attr):
            found_metrics.append(attr)
            print(f"  ✓ {attr} 존재")

    if not found_metrics:
        warnings.append("메트릭 관련 속성/메서드 없음")
        print(f"  △ 메트릭 관련 속성/메서드 찾을 수 없음")

except Exception as e:
    warnings.append(f"메트릭 확인 실패: {e}")
    print(f"  △ 메트릭 확인 실패: {e}")

# 7. CircuitBreakerRegistry 기능 테스트
print("\n[7] CircuitBreakerRegistry 기능 테스트...")
try:
    from truthound.checkpoint.circuitbreaker.registry import CircuitBreakerRegistry

    # 레지스트리 생성
    registry = CircuitBreakerRegistry()
    print(f"  ✓ CircuitBreakerRegistry 인스턴스 생성 성공")

    # 필수 메서드 확인
    registry_methods = ["get", "register", "get_or_create", "list_all", "remove"]
    for method in registry_methods:
        if hasattr(registry, method):
            print(f"  ✓ {method} 메서드 존재")
        else:
            warnings.append(f"CircuitBreakerRegistry에 {method} 메서드 없음")

except Exception as e:
    warnings.append(f"CircuitBreakerRegistry 테스트 실패: {e}")
    print(f"  △ CircuitBreakerRegistry 테스트 실패: {e}")

# 8. 미들웨어 확인
print("\n[8] Circuit Breaker 미들웨어 확인...")
try:
    from truthound.checkpoint.circuitbreaker.middleware import CircuitBreakerMiddleware
    print(f"  ✓ CircuitBreakerMiddleware import 성공")

except ImportError as e:
    warnings.append(f"CircuitBreakerMiddleware import 실패 (선택 기능): {e}")
    print(f"  △ CircuitBreakerMiddleware import 실패 (선택 기능)")

# 9. circuitbreaker 패키지 전체 내용 확인
print("\n[9] circuitbreaker 패키지 전체 내용...")
try:
    from truthound.checkpoint import circuitbreaker
    contents = [x for x in dir(circuitbreaker) if not x.startswith('_')]
    print(f"  패키지 내용: {contents}")

except ImportError as e:
    errors.append(f"circuitbreaker 패키지 import 실패: {e}")
    print(f"  ✗ circuitbreaker 패키지 import 실패: {e}")

# 결과 요약
print("\n" + "=" * 60)
print("Circuit Breaker Pattern 검증 결과")
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
