#!/usr/bin/env python3
"""Phase 6 전체 검증 실행 스크립트"""
import subprocess
import sys
from pathlib import Path

def run_test(test_file: str) -> tuple[bool, str]:
    """테스트 실행 및 결과 반환"""
    result = subprocess.run(
        [sys.executable, test_file],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent.parent
    )
    return result.returncode == 0, result.stdout + result.stderr

def main():
    print("=" * 70)
    print("Phase 6 Checkpoint & CI/CD 전체 검증")
    print("=" * 70)

    test_dir = Path(__file__).parent
    test_files = sorted(test_dir.glob("test_*.py"))

    results = {}
    all_outputs = []

    for test_file in test_files:
        if test_file.name == "run_all_tests.py":
            continue

        print(f"\n실행 중: {test_file.name}...")
        success, output = run_test(str(test_file))
        results[test_file.name] = success
        all_outputs.append((test_file.name, output))

        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  결과: {status}")

    # 종합 결과
    print("\n" + "=" * 70)
    print("Phase 6 검증 종합 결과")
    print("=" * 70)

    passed = sum(1 for v in results.values() if v)
    failed = len(results) - passed

    print(f"\n통과: {passed}/{len(results)}")
    print(f"실패: {failed}/{len(results)}")

    print("\n개별 결과:")
    for name, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {name}")

    # 상세 출력 저장
    report_path = test_dir / "verification_report.txt"
    with open(report_path, "w") as f:
        f.write("Phase 6 Checkpoint & CI/CD 검증 상세 보고서\n")
        f.write("=" * 70 + "\n\n")

        for name, output in all_outputs:
            f.write(f"\n{'=' * 70}\n")
            f.write(f"{name}\n")
            f.write("=" * 70 + "\n")
            f.write(output)
            f.write("\n")

        f.write("\n\n종합 결과\n")
        f.write("=" * 70 + "\n")
        f.write(f"통과: {passed}/{len(results)}\n")
        f.write(f"실패: {failed}/{len(results)}\n")

    print(f"\n상세 보고서 저장됨: {report_path}")

    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
