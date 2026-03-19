#!/usr/bin/env python3
"""Phase 6 검증: Distributed Checkpoint Orchestration 테스트

문서에 명시된 4개 백엔드:
- LocalBackend: 로컬 멀티프로세스/스레드 (개발/테스트)
- CeleryBackend: Redis/RabbitMQ, 우선순위, 재시도
- RayBackend: Actor 기반, 자동 스케일링
- KubernetesBackend: Pod 기반, 클라우드 네이티브

Orchestrator 기능:
- Rate Limiting (토큰 버킷)
- Circuit Breaker
- Task Scheduling (지연/반복 실행)
- Group Submission
- Metrics Collection
"""
import sys

print("=" * 60)
print("Phase 6 검증: Distributed Checkpoint Orchestration")
print("=" * 60)

errors = []
warnings = []
doc_discrepancies = []

# 1. distributed 패키지 import 테스트
print("\n[1] distributed 패키지 import 테스트...")
try:
    from truthound.checkpoint import distributed
    print("  ✓ distributed 패키지 import 성공")
    contents = [x for x in dir(distributed) if not x.startswith('_')]
    print(f"  패키지 내용: {contents}")
except ImportError as e:
    errors.append(f"distributed 패키지 import 실패: {e}")
    print(f"  ✗ distributed 패키지 import 실패: {e}")

# 2. get_orchestrator 함수 확인 (문서에 명시)
print("\n[2] get_orchestrator 함수 확인...")
try:
    from truthound.checkpoint.distributed import get_orchestrator
    print("  ✓ get_orchestrator import 성공")

    # 시그니처 확인
    import inspect
    sig = inspect.signature(get_orchestrator)
    print(f"  시그니처: {sig}")

except ImportError as e:
    # 다른 방식 확인
    try:
        from truthound.checkpoint.distributed.orchestrator import DistributedCheckpointOrchestrator
        print("  △ get_orchestrator 없음, DistributedCheckpointOrchestrator 직접 사용")
        doc_discrepancies.append("문서: get_orchestrator() → 실제: DistributedCheckpointOrchestrator 직접 사용")
    except ImportError as e2:
        errors.append(f"Orchestrator import 실패: {e2}")
        print(f"  ✗ Orchestrator import 실패: {e2}")

# 3. 백엔드 import 테스트
print("\n[3] 백엔드 import 테스트...")
backends = {
    "LocalBackend": "local_backend",
    "CeleryBackend": "celery_backend",
    "RayBackend": "ray_backend",
    "KubernetesBackend": "kubernetes_backend",
}

for backend_class, backend_module in backends.items():
    try:
        module = __import__(
            f"truthound.checkpoint.distributed.backends.{backend_module}",
            fromlist=[backend_class]
        )
        if hasattr(module, backend_class):
            print(f"  ✓ {backend_class} import 성공")
        else:
            # 다른 이름으로 존재할 수 있음
            module_contents = [x for x in dir(module) if "backend" in x.lower()]
            if module_contents:
                print(f"  △ {backend_class} 없음, 발견된 클래스: {module_contents}")
            else:
                warnings.append(f"{backend_class} 클래스 없음")
                print(f"  △ {backend_class} 클래스 없음")
    except ImportError as e:
        warnings.append(f"{backend_class} import 실패 (선택적 의존성): {e}")
        print(f"  △ {backend_class} import 실패 (선택적 의존성일 수 있음)")

# 4. DistributedBackendProtocol 확인
print("\n[4] DistributedBackendProtocol 확인...")
try:
    from truthound.checkpoint.distributed.protocols import DistributedBackendProtocol
    print("  ✓ DistributedBackendProtocol import 성공")

    # 프로토콜 메서드 확인
    protocol_methods = [m for m in dir(DistributedBackendProtocol) if not m.startswith('_')]
    print(f"  프로토콜 메서드: {protocol_methods}")

except ImportError as e:
    try:
        from truthound.checkpoint.distributed import protocols
        contents = [x for x in dir(protocols) if not x.startswith('_')]
        print(f"  △ protocols 모듈 내용: {contents}")
    except ImportError as e2:
        warnings.append(f"DistributedBackendProtocol import 실패: {e}")
        print(f"  △ DistributedBackendProtocol import 실패: {e}")

# 5. Orchestrator 기능 확인
print("\n[5] Orchestrator 기능 확인...")
try:
    from truthound.checkpoint.distributed.orchestrator import DistributedCheckpointOrchestrator
    import inspect

    # 클래스 메서드 확인
    methods = [m for m in dir(DistributedCheckpointOrchestrator) if not m.startswith('_')]
    print(f"  Orchestrator 메서드: {methods}")

    # 문서에 명시된 기능 확인
    expected_features = {
        "submit": "Task Submission",
        "rate_limit": "Rate Limiting",
        "circuit_breaker": "Circuit Breaker",
        "schedule": "Task Scheduling",
        "submit_group": "Group Submission",
        "metrics": "Metrics Collection",
    }

    for feature, desc in expected_features.items():
        if any(feature in m.lower() for m in methods):
            print(f"  ✓ {desc} 기능 존재")
        else:
            warnings.append(f"Orchestrator에 {desc} 기능 확인 안됨")

except ImportError as e:
    errors.append(f"Orchestrator import 실패: {e}")
    print(f"  ✗ Orchestrator import 실패: {e}")

# 6. LocalBackend 상세 테스트
print("\n[6] LocalBackend 상세 테스트...")
try:
    from truthound.checkpoint.distributed.backends.local_backend import LocalBackend

    # 인스턴스 생성
    backend = LocalBackend()
    print("  ✓ LocalBackend 인스턴스 생성 성공")

    # 필수 메서드 확인
    backend_methods = ["submit", "get_result", "cancel", "status"]
    for method in backend_methods:
        if hasattr(backend, method):
            print(f"  ✓ {method} 메서드 존재")
        else:
            warnings.append(f"LocalBackend에 {method} 메서드 없음")

except Exception as e:
    warnings.append(f"LocalBackend 상세 테스트 실패: {e}")
    print(f"  △ LocalBackend 상세 테스트 실패: {e}")

# 7. Registry 확인
print("\n[7] Backend Registry 확인...")
try:
    from truthound.checkpoint.distributed.registry import BackendRegistry
    print("  ✓ BackendRegistry import 성공")

    # 등록된 백엔드 확인
    if hasattr(BackendRegistry, 'list_backends'):
        backends = BackendRegistry.list_backends()
        print(f"  등록된 백엔드: {backends}")

except ImportError as e:
    warnings.append(f"BackendRegistry import 실패: {e}")
    print(f"  △ BackendRegistry import 실패: {e}")

# 8. Context Manager 지원 확인 (문서 예시: with orchestrator:)
print("\n[8] Context Manager 지원 확인...")
try:
    from truthound.checkpoint.distributed.orchestrator import DistributedCheckpointOrchestrator

    # __enter__, __exit__ 메서드 확인
    if hasattr(DistributedCheckpointOrchestrator, '__enter__') and \
       hasattr(DistributedCheckpointOrchestrator, '__exit__'):
        print("  ✓ Context Manager 지원 (with 문 사용 가능)")
    else:
        doc_discrepancies.append("문서: with orchestrator: → 실제: Context Manager 미지원")
        print("  △ Context Manager 미지원")

except Exception as e:
    warnings.append(f"Context Manager 확인 실패: {e}")
    print(f"  △ Context Manager 확인 실패: {e}")

# 결과 요약
print("\n" + "=" * 60)
print("Distributed Checkpoint Orchestration 검증 결과")
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
    print("\n결과: ✓ PASS" if not (warnings or doc_discrepancies) else "\n결과: △ PASS (확인 필요)")
    sys.exit(0)
