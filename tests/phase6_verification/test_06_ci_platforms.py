#!/usr/bin/env python3
"""Phase 6 검증: CI/CD Platform Integration 테스트

문서에 명시된 12개 플랫폼:
GitHub Actions, GitLab CI, Jenkins, CircleCI, Travis CI, Azure DevOps,
Bitbucket Pipelines, TeamCity, Buildkite, Drone, AWS CodeBuild, Google Cloud Build

- 자동 감지: CIEnvironment (branch, commit, PR 정보)
"""
import sys

print("=" * 60)
print("Phase 6 검증: CI/CD Platform Integration")
print("=" * 60)

errors = []
warnings = []
doc_discrepancies = []

# 문서에 명시된 12개 플랫폼
EXPECTED_PLATFORMS = [
    "github_actions", "gitlab_ci", "jenkins", "circleci", "travis_ci",
    "azure_devops", "bitbucket_pipelines", "teamcity", "buildkite",
    "drone", "aws_codebuild", "google_cloud_build"
]

# 1. CI 패키지 import 테스트
print("\n[1] CI 패키지 import 테스트...")
try:
    from truthound.checkpoint import ci
    print("  ✓ ci 패키지 import 성공")
    contents = [x for x in dir(ci) if not x.startswith('_')]
    print(f"  패키지 내용: {contents}")
except ImportError as e:
    errors.append(f"ci 패키지 import 실패: {e}")
    print(f"  ✗ ci 패키지 import 실패: {e}")

# 2. CI Detection 함수 import 테스트
print("\n[2] CI Detection 기능 import 테스트...")
try:
    from truthound.checkpoint.ci import (
        detect_ci_platform,
        get_ci_environment,
        is_ci_environment,
        CIEnvironment,
        CIPlatform,
    )
    print("  ✓ detect_ci_platform, get_ci_environment, CIEnvironment import 성공")
    doc_discrepancies.append("문서: CIDetector 클래스 → 실제: detect_ci_platform, get_ci_environment 함수")
except ImportError as e:
    errors.append(f"CI Detection 기능 import 실패: {e}")
    print(f"  ✗ CI Detection 기능 import 실패: {e}")

# 3. CIEnvironment 속성 확인
print("\n[3] CIEnvironment 속성 확인...")
try:
    from truthound.checkpoint.ci.detector import CIEnvironment
    import inspect

    # dataclass 필드 또는 속성 확인
    if hasattr(CIEnvironment, '__dataclass_fields__'):
        fields = list(CIEnvironment.__dataclass_fields__.keys())
    else:
        fields = [m for m in dir(CIEnvironment) if not m.startswith('_')]

    print(f"  CIEnvironment 필드: {fields}")

    # 필수 속성 확인
    expected_attrs = ["branch", "commit", "pr", "platform", "build_id"]
    for attr in expected_attrs:
        if attr in fields or any(attr in f.lower() for f in fields):
            print(f"  ✓ {attr} 속성 존재")
        else:
            warnings.append(f"CIEnvironment에 {attr} 속성 없음")
            print(f"  △ {attr} 속성 없음")

except Exception as e:
    warnings.append(f"CIEnvironment 속성 확인 실패: {e}")
    print(f"  △ CIEnvironment 속성 확인 실패: {e}")

# 4. CI 플랫폼 감지 기능 확인
print("\n[4] CI 플랫폼 감지 기능...")
try:
    from truthound.checkpoint.ci import detect_ci_platform, CIPlatform

    # CIPlatform Enum 확인
    platforms = [p.name for p in CIPlatform]
    print(f"  ✓ 지원 플랫폼 (CIPlatform): {platforms}")

    # detect_ci_platform 함수 테스트
    detected = detect_ci_platform()
    print(f"  ✓ detect_ci_platform() 호출 성공: {detected}")

except Exception as e:
    warnings.append(f"CI 플랫폼 감지 기능 확인 실패: {e}")
    print(f"  △ CI 플랫폼 감지 기능 확인 실패: {e}")

# 5. 플랫폼별 구현 확인
print("\n[5] 플랫폼별 구현 확인...")
try:
    from truthound.checkpoint.ci import detector
    module_contents = dir(detector)

    # 플랫폼 관련 클래스/함수 찾기
    platform_items = [x for x in module_contents
                      if any(p in x.lower() for p in ["github", "gitlab", "jenkins",
                                                       "circle", "travis", "azure",
                                                       "bitbucket", "teamcity", "buildkite",
                                                       "drone", "codebuild", "cloudbuild"])]

    if platform_items:
        print(f"  플랫폼 관련 항목: {platform_items}")
    else:
        print(f"  △ 명시적 플랫폼 클래스 없음 (단일 Detector에서 처리)")

    # Detector 소스 확인
    import inspect
    source = inspect.getsource(detector)

    platforms_found = []
    for platform in EXPECTED_PLATFORMS:
        # 다양한 형태로 검색
        platform_variants = [
            platform,
            platform.upper(),
            platform.replace("_", ""),
            platform.replace("_", "-"),
        ]
        if any(v in source.lower() for v in platform_variants):
            platforms_found.append(platform)

    print(f"  소스에서 발견된 플랫폼: {len(platforms_found)}/12")
    for p in platforms_found:
        print(f"    ✓ {p}")

    missing = set(EXPECTED_PLATFORMS) - set(platforms_found)
    if missing:
        for p in missing:
            warnings.append(f"플랫폼 {p} 구현 확인 안됨")
            print(f"    △ {p} (확인 안됨)")

except Exception as e:
    warnings.append(f"플랫폼별 구현 확인 실패: {e}")
    print(f"  △ 플랫폼별 구현 확인 실패: {e}")

# 6. CI Reporter 확인
print("\n[6] CI Reporter 확인...")
try:
    from truthound.checkpoint.ci.reporter import CIReporter
    print("  ✓ CIReporter import 성공")

    # 리포터 메서드 확인
    reporter_methods = [m for m in dir(CIReporter) if not m.startswith('_')]
    print(f"  CIReporter 메서드: {reporter_methods}")

except ImportError as e:
    warnings.append(f"CIReporter import 실패: {e}")
    print(f"  △ CIReporter import 실패: {e}")

# 7. CI Templates 확인
print("\n[7] CI Templates 확인...")
try:
    from truthound.checkpoint.ci.templates import (
        get_template,
    )
    print("  ✓ templates 모듈 import 성공")

except ImportError as e:
    try:
        from truthound.checkpoint.ci import templates
        contents = [x for x in dir(templates) if not x.startswith('_')]
        print(f"  △ templates 모듈 내용: {contents}")
    except ImportError as e2:
        warnings.append(f"templates 모듈 없음: {e}")
        print(f"  △ templates 모듈 없음")

# 8. 실제 감지 테스트 (로컬 환경)
print("\n[8] 실제 CI 환경 감지 테스트...")
try:
    from truthound.checkpoint.ci import get_ci_environment, is_ci_environment

    # 현재 환경 감지 시도
    is_ci = is_ci_environment()
    print(f"  ✓ is_ci_environment() = {is_ci}")

    env = get_ci_environment()
    if env:
        print(f"  ✓ CI 환경 감지됨: platform={env.platform}, is_ci={env.is_ci}")
    else:
        print(f"  △ CI 환경 아님 (로컬 환경)")

except Exception as e:
    print(f"  △ CI 환경 감지 테스트: {e}")

# 결과 요약
print("\n" + "=" * 60)
print("CI/CD Platform Integration 검증 결과")
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
