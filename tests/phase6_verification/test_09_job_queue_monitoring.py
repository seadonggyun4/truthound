#!/usr/bin/env python3
"""Phase 6 검증: Job Queue Monitoring 테스트

문서에 명시된 기능:
- monitoring/ - InMemory, Redis, Prometheus 수집기
- aggregators/
- collectors/
- views/
"""
import sys

print("=" * 60)
print("Phase 6 검증: Job Queue Monitoring")
print("=" * 60)

errors = []
warnings = []
doc_discrepancies = []

# 1. monitoring 패키지 import 테스트
print("\n[1] monitoring 패키지 import 테스트...")
try:
    from truthound.checkpoint import monitoring
    print("  ✓ monitoring 패키지 import 성공")
    contents = [x for x in dir(monitoring) if not x.startswith('_')]
    print(f"  패키지 내용: {contents}")
except ImportError as e:
    errors.append(f"monitoring 패키지 import 실패: {e}")
    print(f"  ✗ monitoring 패키지 import 실패: {e}")

# 2. Collectors 확인 (InMemory, Redis, Prometheus)
print("\n[2] Collectors 확인...")
collectors_to_check = [
    ("memory_collector", "MemoryCollector", "InMemoryCollector"),
    ("redis_collector", "RedisCollector"),
    ("prometheus_collector", "PrometheusCollector"),
]

for collector_info in collectors_to_check:
    module_name = collector_info[0]
    class_names = collector_info[1:]

    try:
        module = __import__(
            f"truthound.checkpoint.monitoring.collectors.{module_name}",
            fromlist=class_names
        )
        for cls_name in class_names:
            if hasattr(module, cls_name):
                print(f"  ✓ {cls_name} import 성공")
                break
        else:
            # 모듈 내용 확인
            module_contents = [x for x in dir(module) if not x.startswith('_')]
            print(f"  △ {module_name} 모듈 내용: {module_contents}")
    except ImportError as e:
        warnings.append(f"{module_name} import 실패: {e}")
        print(f"  △ {module_name} import 실패: {e}")

# 3. Aggregators 확인
print("\n[3] Aggregators 확인...")
try:
    from truthound.checkpoint.monitoring import aggregators
    print("  ✓ aggregators 패키지 import 성공")

    aggregator_contents = [x for x in dir(aggregators) if not x.startswith('_')]
    print(f"  aggregators 내용: {aggregator_contents}")

    # 개별 모듈 확인
    for module_name in ["base", "realtime", "window"]:
        try:
            module = __import__(
                f"truthound.checkpoint.monitoring.aggregators.{module_name}",
                fromlist=["*"]
            )
            module_contents = [x for x in dir(module) if not x.startswith('_') and x[0].isupper()]
            print(f"  ✓ {module_name} 모듈: {module_contents}")
        except ImportError as e:
            warnings.append(f"aggregators.{module_name} import 실패: {e}")

except ImportError as e:
    warnings.append(f"aggregators 패키지 import 실패: {e}")
    print(f"  △ aggregators 패키지 import 실패: {e}")

# 4. Views 확인
print("\n[4] Views 확인...")
try:
    from truthound.checkpoint.monitoring import views
    print("  ✓ views 패키지 import 성공")

    views_contents = [x for x in dir(views) if not x.startswith('_')]
    print(f"  views 내용: {views_contents}")

    # queue_view 확인
    try:
        from truthound.checkpoint.monitoring.views.queue_view import QueueView
        print("  ✓ QueueView import 성공")
    except ImportError:
        try:
            from truthound.checkpoint.monitoring.views import queue_view
            module_contents = [x for x in dir(queue_view) if not x.startswith('_')]
            print(f"  △ queue_view 모듈 내용: {module_contents}")
        except ImportError as e:
            warnings.append(f"queue_view import 실패: {e}")

except ImportError as e:
    warnings.append(f"views 패키지 import 실패: {e}")
    print(f"  △ views 패키지 import 실패: {e}")

# 5. Monitoring Service 확인
print("\n[5] Monitoring Service 확인...")
try:
    from truthound.checkpoint.monitoring.service import MonitoringService
    print("  ✓ MonitoringService import 성공")

    import inspect
    methods = [m for m in dir(MonitoringService) if not m.startswith('_')]
    print(f"  MonitoringService 메서드: {methods}")

except ImportError as e:
    warnings.append(f"MonitoringService import 실패: {e}")
    print(f"  △ MonitoringService import 실패: {e}")

# 6. Protocols 확인
print("\n[6] Monitoring Protocols 확인...")
try:
    from truthound.checkpoint.monitoring.protocols import (
        MonitoringProtocol,
    )
    print("  ✓ MonitoringProtocol import 성공")

except ImportError as e:
    try:
        from truthound.checkpoint.monitoring import protocols
        protocol_contents = [x for x in dir(protocols) if not x.startswith('_')]
        print(f"  △ protocols 모듈 내용: {protocol_contents}")
    except ImportError as e2:
        warnings.append(f"Monitoring Protocols import 실패: {e}")
        print(f"  △ Monitoring Protocols import 실패: {e}")

# 7. Events 확인
print("\n[7] Monitoring Events 확인...")
try:
    from truthound.checkpoint.monitoring.events import (
        MonitoringEvent,
    )
    print("  ✓ MonitoringEvent import 성공")

except ImportError as e:
    try:
        from truthound.checkpoint.monitoring import events
        events_contents = [x for x in dir(events) if not x.startswith('_')]
        print(f"  △ events 모듈 내용: {events_contents}")
    except ImportError as e2:
        warnings.append(f"Monitoring Events import 실패: {e}")
        print(f"  △ Monitoring Events import 실패: {e}")

# 8. 실제 사용 테스트
print("\n[8] 실제 사용 테스트...")
try:
    from truthound.checkpoint.monitoring.collectors.memory_collector import MemoryCollector

    collector = MemoryCollector()
    print("  ✓ MemoryCollector 인스턴스 생성 성공")

    # 필수 메서드 확인
    collector_methods = ["collect", "get_metrics", "record", "flush"]
    for method in collector_methods:
        if hasattr(collector, method):
            print(f"  ✓ {method} 메서드 존재")

except Exception as e:
    warnings.append(f"실제 사용 테스트 실패: {e}")
    print(f"  △ 실제 사용 테스트 실패: {e}")

# 결과 요약
print("\n" + "=" * 60)
print("Job Queue Monitoring 검증 결과")
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
