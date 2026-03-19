#!/usr/bin/env python3
"""Phase 6 검증: Saga Pattern 테스트

문서에 명시된 기능:
- SagaBuilder: Fluent API로 Saga 구성
- SagaRunner: 실행, 재개, 일시중지, 중단
- 8가지 보상 정책: Backward, Forward, Semantic, Pivot, Countermeasure, Parallel, Selective, Best-effort
- Event Store: 이벤트 저장 및 리플레이
"""
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

print("=" * 60)
print("Phase 6 검증: Saga Pattern")
print("=" * 60)

errors = []
warnings = []
doc_discrepancies = []

# 1. Import 테스트
print("\n[1] Import 테스트...")
try:
    from truthound.checkpoint.transaction.saga import (
        SagaBuilder,
        SagaRunner,
        SagaDefinition,
        SagaStepDefinition,
        StepBuilder,
    )
    print("  ✓ 기본 클래스 import 성공")
except ImportError as e:
    errors.append(f"기본 클래스 import 실패: {e}")
    print(f"  ✗ 기본 클래스 import 실패: {e}")

# SagaState 확인 (SagaStatus 대신)
try:
    from truthound.checkpoint.transaction.saga import SagaState
    print("  ✓ SagaState import 성공")
    doc_discrepancies.append("문서: SagaStatus → 실제: SagaState")
except ImportError as e:
    errors.append(f"SagaState import 실패: {e}")
    print(f"  ✗ SagaState import 실패: {e}")

# 2. 보상 정책 import 테스트
print("\n[2] 보상 정책 import 테스트...")
try:
    from truthound.checkpoint.transaction.saga.strategies import (
        CompensationPolicy,
    )
    print("  ✓ CompensationPolicy import 성공")

    # 8가지 보상 정책 확인
    expected_policies = [
        "BACKWARD", "FORWARD", "SEMANTIC", "PIVOT",
        "COUNTERMEASURE", "PARALLEL", "SELECTIVE", "BEST_EFFORT"
    ]
    actual_policies = [p.name for p in CompensationPolicy]

    missing = set(expected_policies) - set(actual_policies)
    if missing:
        errors.append(f"누락된 보상 정책: {missing}")
        print(f"  ✗ 누락된 보상 정책: {missing}")
    else:
        print(f"  ✓ 8가지 보상 정책 모두 존재: {actual_policies}")
except ImportError as e:
    errors.append(f"CompensationPolicy import 실패: {e}")
    print(f"  ✗ CompensationPolicy import 실패: {e}")
except Exception as e:
    errors.append(f"보상 정책 검증 실패: {e}")
    print(f"  ✗ 보상 정책 검증 실패: {e}")

# 3. Event Store import 테스트
print("\n[3] Event Store import 테스트...")
try:
    from truthound.checkpoint.transaction.saga import (
        InMemorySagaEventStore,
        FileSagaEventStore,
        SagaEventStore,
    )
    print("  ✓ Event Store import 성공")
    doc_discrepancies.append("문서: FileSystemEventStore → 실제: FileSagaEventStore")
except ImportError as e:
    errors.append(f"Event Store import 실패: {e}")
    print(f"  ✗ Event Store import 실패: {e}")

# 4. SagaBuilder 실제 API 테스트
print("\n[4] SagaBuilder 실제 API 테스트...")
try:
    from truthound.checkpoint.actions.base import BaseAction, ActionConfig, ActionResult, ActionStatus

    # 테스트용 액션 구현
    @dataclass
    class TestActionConfig(ActionConfig):
        test_value: str = "test"

    class TestAction(BaseAction[TestActionConfig]):
        action_type = "test_action"

        def __init__(self, name_val: str = "test"):
            super().__init__()
            self._name_val = name_val

        @classmethod
        def _default_config(cls):
            return TestActionConfig()

        def _execute(self, checkpoint_result):
            return ActionResult(
                action_name=self._name_val,
                action_type=self.action_type,
                status=ActionStatus.SUCCESS,
                message=f"Executed {self._name_val}"
            )

    # SagaBuilder 사용 - 실제 API 패턴
    builder = SagaBuilder("test_saga")
    print("  ✓ SagaBuilder 생성 성공")

    # step() 메서드 확인 - StepBuilder 반환
    step_builder = builder.step("load_data")
    print(f"  ✓ step() 반환 타입: {type(step_builder).__name__}")

    if isinstance(step_builder, StepBuilder):
        print("  ✓ StepBuilder 반환 확인")
    else:
        warnings.append(f"step()이 StepBuilder가 아닌 {type(step_builder)}를 반환")

    # StepBuilder 체이닝 메서드 확인
    step_methods = ["action", "compensate_with", "depends_on", "with_timeout",
                    "with_retry", "as_pivot", "end_step"]
    for method in step_methods:
        if hasattr(step_builder, method):
            print(f"  ✓ StepBuilder.{method} 존재")
        else:
            errors.append(f"StepBuilder에 {method} 메서드 없음")
            print(f"  ✗ StepBuilder.{method} 없음")

    # 문서와 API 불일치 기록
    doc_discrepancies.append("문서: .add_step(name, action, compensate=...) → 실제: .step(name).action(...).compensate_with(...).end_step()")

except Exception as e:
    errors.append(f"SagaBuilder API 테스트 실패: {e}")
    print(f"  ✗ SagaBuilder API 테스트 실패: {e}")
    import traceback
    traceback.print_exc()

# 5. 실제 Saga 빌드 테스트
print("\n[5] 실제 Saga 빌드 테스트...")
try:
    # 실제 API 사용
    saga = (
        SagaBuilder("execution_test")
        .step("step1")
            .action(TestAction("step1"))
            .compensate_with(TestAction("step1_comp"))
        .end_step()
        .step("step2")
            .action(TestAction("step2"))
            .depends_on("step1")
        .end_step()
        .with_timeout(60)
        .with_policy("backward")
        .build()
    )

    print(f"  ✓ Saga 빌드 성공: {type(saga).__name__}")
    print(f"  ✓ Steps 수: {len(saga.steps)}")

    # 각 스텝 정보 확인
    for step in saga.steps:
        print(f"    - Step: {step.step_id}, Action: {step.action}, Compensation: {step.compensation_action}")

    doc_discrepancies.append("문서: SagaRunner(saga, event_store=...) → 실제: SagaRunner(), runner.execute(saga, context)")

except Exception as e:
    errors.append(f"Saga 빌드 테스트 실패: {e}")
    print(f"  ✗ Saga 빌드 테스트 실패: {e}")
    import traceback
    traceback.print_exc()

# 6. SagaRunner 메서드 확인
print("\n[6] SagaRunner 메서드 확인...")
try:
    runner = SagaRunner()
    print(f"  ✓ SagaRunner 생성 성공")

    runner_methods = {
        "execute": "필수",
        "resume": "필수 (문서)",
        "pause": "선택 (문서: 일시중지)",
        "suspend": "선택",
        "abort": "선택 (문서: 중단)",
    }

    for method, importance in runner_methods.items():
        if hasattr(runner, method):
            print(f"  ✓ SagaRunner.{method} 존재 ({importance})")
        else:
            if "필수" in importance:
                errors.append(f"SagaRunner에 {method} 메서드 없음 ({importance})")
                print(f"  ✗ SagaRunner.{method} 없음 ({importance})")
            else:
                warnings.append(f"SagaRunner에 {method} 메서드 없음 ({importance})")
                print(f"  △ SagaRunner.{method} 없음 ({importance})")

    # runner.execute 시그니처 확인
    import inspect
    sig = inspect.signature(runner.execute)
    print(f"  ✓ SagaRunner.execute 시그니처: {sig}")

except Exception as e:
    errors.append(f"SagaRunner 메서드 확인 실패: {e}")
    print(f"  ✗ SagaRunner 메서드 확인 실패: {e}")

# 7. SagaState 상태값 확인
print("\n[7] SagaState 상태값 확인...")
try:
    actual_states = [s.name for s in SagaState]
    print(f"  실제 상태값: {actual_states}")

    for state in ["COMPLETED", "FAILED"]:
        if state in actual_states:
            print(f"  ✓ 핵심 상태값 '{state}' 존재")
        else:
            errors.append(f"핵심 상태값 '{state}' 누락")
            print(f"  ✗ 핵심 상태값 '{state}' 누락")

except Exception as e:
    errors.append(f"SagaState 검증 실패: {e}")
    print(f"  ✗ SagaState 검증 실패: {e}")

# 8. Event Store 기능 테스트
print("\n[8] Event Store 기능 테스트...")
try:
    # InMemory Event Store
    event_store = InMemorySagaEventStore()
    print(f"  ✓ InMemorySagaEventStore 생성 성공")

    # 필수 메서드 확인
    store_methods = ["append", "get_events", "save_snapshot"]
    for method in store_methods:
        if hasattr(event_store, method):
            print(f"  ✓ {method} 메서드 존재")
        else:
            warnings.append(f"InMemorySagaEventStore에 {method} 메서드 없음")
            print(f"  △ {method} 메서드 없음")

    # FileSagaEventStore 테스트
    with tempfile.TemporaryDirectory() as tmpdir:
        file_store = FileSagaEventStore(tmpdir)
        print(f"  ✓ FileSagaEventStore 생성 성공")

except Exception as e:
    errors.append(f"Event Store 테스트 실패: {e}")
    print(f"  ✗ Event Store 테스트 실패: {e}")

# 9. 고급 패턴 테스트
print("\n[9] 고급 Saga 패턴 테스트...")
try:
    from truthound.checkpoint.transaction.saga import (
        ChainedSagaPattern,
        NestedSagaPattern,
        ParallelSagaPattern,
        ChoreographySagaPattern,
        OrchestratorSagaPattern,
    )
    print("  ✓ 고급 패턴 import 성공")

    patterns = [
        ("ChainedSagaPattern", ChainedSagaPattern),
        ("NestedSagaPattern", NestedSagaPattern),
        ("ParallelSagaPattern", ParallelSagaPattern),
        ("ChoreographySagaPattern", ChoreographySagaPattern),
        ("OrchestratorSagaPattern", OrchestratorSagaPattern),
    ]

    for name, cls in patterns:
        print(f"  ✓ {name} 사용 가능")

except ImportError as e:
    warnings.append(f"고급 패턴 import 실패: {e}")
    print(f"  △ 고급 패턴 import 실패: {e}")

# 10. 테스팅 유틸리티 확인
print("\n[10] 테스팅 유틸리티 확인...")
try:
    from truthound.checkpoint.transaction.saga import (
        SagaTestHarness,
        SagaScenario,
        FailureInjector,
        SagaAssertion,
        ScenarioBuilder,
    )
    print("  ✓ 테스팅 유틸리티 import 성공")

except ImportError as e:
    warnings.append(f"테스팅 유틸리티 import 실패: {e}")
    print(f"  △ 테스팅 유틸리티 import 실패: {e}")

# 결과 요약
print("\n" + "=" * 60)
print("Saga Pattern 검증 결과")
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
elif doc_discrepancies:
    print("\n결과: △ PASS (문서 업데이트 필요)")
    sys.exit(0)
else:
    print("\n결과: ✓ PASS")
    sys.exit(0)
