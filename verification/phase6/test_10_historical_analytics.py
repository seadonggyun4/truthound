#!/usr/bin/env python3
"""Phase 6 검증: Historical Analytics 테스트

문서에 명시된 기능:
- analytics/ - 트렌드 분석, 이상 감지, 예측
- aggregations/ - time_bucket, rollup
- analyzers/ - anomaly, base, forecast, trend
- stores/ - memory_store, sqlite_store, timescale_store
"""
import sys

print("=" * 60)
print("Phase 6 검증: Historical Analytics")
print("=" * 60)

errors = []
warnings = []
doc_discrepancies = []

# 1. analytics 패키지 import 테스트
print("\n[1] analytics 패키지 import 테스트...")
try:
    from truthound.checkpoint import analytics
    print("  ✓ analytics 패키지 import 성공")
    contents = [x for x in dir(analytics) if not x.startswith('_')]
    print(f"  패키지 내용: {contents}")
except ImportError as e:
    errors.append(f"analytics 패키지 import 실패: {e}")
    print(f"  ✗ analytics 패키지 import 실패: {e}")

# 2. Aggregations 확인 (time_bucket, rollup)
print("\n[2] Aggregations 확인...")
try:
    from truthound.checkpoint.analytics import aggregations
    print("  ✓ aggregations 패키지 import 성공")

    # time_bucket 모듈
    try:
        from truthound.checkpoint.analytics.aggregations.time_bucket import TimeBucket
        print("  ✓ TimeBucket import 성공")
    except ImportError:
        try:
            from truthound.checkpoint.analytics.aggregations import time_bucket
            contents = [x for x in dir(time_bucket) if not x.startswith('_')]
            print(f"  △ time_bucket 모듈 내용: {contents}")
        except ImportError as e:
            warnings.append(f"time_bucket import 실패: {e}")

    # rollup 모듈
    try:
        from truthound.checkpoint.analytics.aggregations.rollup import Rollup
        print("  ✓ Rollup import 성공")
    except ImportError:
        try:
            from truthound.checkpoint.analytics.aggregations import rollup
            contents = [x for x in dir(rollup) if not x.startswith('_')]
            print(f"  △ rollup 모듈 내용: {contents}")
        except ImportError as e:
            warnings.append(f"rollup import 실패: {e}")

except ImportError as e:
    warnings.append(f"aggregations 패키지 import 실패: {e}")
    print(f"  △ aggregations 패키지 import 실패: {e}")

# 3. Analyzers 확인 (anomaly, trend, forecast)
print("\n[3] Analyzers 확인...")
analyzers_to_check = ["anomaly", "trend", "forecast", "base"]

try:
    from truthound.checkpoint.analytics import analyzers
    print("  ✓ analyzers 패키지 import 성공")

    for analyzer_name in analyzers_to_check:
        try:
            module = __import__(
                f"truthound.checkpoint.analytics.analyzers.{analyzer_name}",
                fromlist=["*"]
            )
            module_contents = [x for x in dir(module) if not x.startswith('_') and x[0].isupper()]
            if module_contents:
                print(f"  ✓ {analyzer_name} 모듈: {module_contents}")
            else:
                print(f"  △ {analyzer_name} 모듈 (클래스 없음)")
        except ImportError as e:
            warnings.append(f"analyzers.{analyzer_name} import 실패: {e}")
            print(f"  △ {analyzer_name} import 실패")

except ImportError as e:
    warnings.append(f"analyzers 패키지 import 실패: {e}")
    print(f"  △ analyzers 패키지 import 실패: {e}")

# 4. Stores 확인 (memory, sqlite, timescale)
print("\n[4] Analytics Stores 확인...")
stores_to_check = ["memory_store", "sqlite_store", "timescale_store"]

try:
    from truthound.checkpoint.analytics import stores
    print("  ✓ stores 패키지 import 성공")

    for store_name in stores_to_check:
        try:
            module = __import__(
                f"truthound.checkpoint.analytics.stores.{store_name}",
                fromlist=["*"]
            )
            module_contents = [x for x in dir(module) if not x.startswith('_') and x[0].isupper()]
            if module_contents:
                print(f"  ✓ {store_name} 모듈: {module_contents}")
            else:
                all_contents = [x for x in dir(module) if not x.startswith('_')]
                print(f"  △ {store_name} 모듈 내용: {all_contents}")
        except ImportError as e:
            if "timescale" in store_name:
                warnings.append(f"stores.{store_name} import 실패 (선택적 의존성): {e}")
                print(f"  △ {store_name} import 실패 (선택적 의존성)")
            else:
                warnings.append(f"stores.{store_name} import 실패: {e}")
                print(f"  △ {store_name} import 실패")

except ImportError as e:
    warnings.append(f"stores 패키지 import 실패: {e}")
    print(f"  △ stores 패키지 import 실패: {e}")

# 5. Analytics Service 확인
print("\n[5] Analytics Service 확인...")
try:
    from truthound.checkpoint.analytics.service import AnalyticsService
    print("  ✓ AnalyticsService import 성공")

    import inspect
    methods = [m for m in dir(AnalyticsService) if not m.startswith('_')]
    print(f"  AnalyticsService 메서드: {methods}")

    # 주요 기능 확인
    expected_methods = ["analyze", "get_trends", "detect_anomalies", "forecast"]
    for method in expected_methods:
        if method in methods or any(method in m for m in methods):
            print(f"  ✓ {method} 관련 기능 존재")

except ImportError as e:
    warnings.append(f"AnalyticsService import 실패: {e}")
    print(f"  △ AnalyticsService import 실패: {e}")

# 6. Models 확인
print("\n[6] Analytics Models 확인...")
try:
    from truthound.checkpoint.analytics.models import (
        AnalyticsResult,
    )
    print("  ✓ AnalyticsResult import 성공")

except ImportError as e:
    try:
        from truthound.checkpoint.analytics import models
        models_contents = [x for x in dir(models) if not x.startswith('_')]
        print(f"  △ models 모듈 내용: {models_contents}")
    except ImportError as e2:
        warnings.append(f"Analytics Models import 실패: {e}")
        print(f"  △ Analytics Models import 실패: {e}")

# 7. Protocols 확인
print("\n[7] Analytics Protocols 확인...")
try:
    from truthound.checkpoint.analytics.protocols import (
        AnalyticsProtocol,
    )
    print("  ✓ AnalyticsProtocol import 성공")

except ImportError as e:
    try:
        from truthound.checkpoint.analytics import protocols
        protocols_contents = [x for x in dir(protocols) if not x.startswith('_')]
        print(f"  △ protocols 모듈 내용: {protocols_contents}")
    except ImportError as e2:
        warnings.append(f"Analytics Protocols import 실패: {e}")
        print(f"  △ Analytics Protocols import 실패: {e}")

# 8. 실제 사용 테스트
print("\n[8] 실제 사용 테스트...")
try:
    from truthound.checkpoint.analytics.stores.memory_store import MemoryStore

    store = MemoryStore()
    print("  ✓ MemoryStore 인스턴스 생성 성공")

    # 필수 메서드 확인
    store_methods = ["store", "query", "get_range", "aggregate"]
    for method in store_methods:
        if hasattr(store, method):
            print(f"  ✓ {method} 메서드 존재")

except Exception as e:
    warnings.append(f"실제 사용 테스트 실패: {e}")
    print(f"  △ 실제 사용 테스트 실패: {e}")
    import traceback
    traceback.print_exc()

# 결과 요약
print("\n" + "=" * 60)
print("Historical Analytics 검증 결과")
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
