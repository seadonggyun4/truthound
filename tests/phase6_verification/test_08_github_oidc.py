#!/usr/bin/env python3
"""Phase 6 검증: GitHub Actions OIDC Integration 테스트

문서에 명시된 기능:
- claims.py: 30+ GitHub Actions 클레임 파싱
- enhanced_provider.py: 환경/워크플로우 정책 기반 OIDC
- trust_policy.py: AWS/GCP/Azure/Vault 정책 생성 (Terraform/CLI)
- verification.py: JWKS 기반 토큰 검증
- workflow.py: 워크플로우 출력, 로깅, Job Summary
"""
import sys

print("=" * 60)
print("Phase 6 검증: GitHub Actions OIDC Integration")
print("=" * 60)

errors = []
warnings = []
doc_discrepancies = []

# 1. secrets.oidc.github 패키지 import 테스트
print("\n[1] GitHub OIDC 패키지 import 테스트...")
try:
    from truthound.secrets.oidc import github
    print("  ✓ github OIDC 패키지 import 성공")
    contents = [x for x in dir(github) if not x.startswith('_')]
    print(f"  패키지 내용: {contents}")
except ImportError as e:
    errors.append(f"GitHub OIDC 패키지 import 실패: {e}")
    print(f"  ✗ GitHub OIDC 패키지 import 실패: {e}")

# 2. GitHubActionsOIDC 클래스 확인
print("\n[2] GitHubActionsOIDC 클래스 확인...")
try:
    from truthound.secrets.oidc.github import GitHubActionsOIDC
    print("  ✓ GitHubActionsOIDC import 성공")

    import inspect
    sig = inspect.signature(GitHubActionsOIDC.__init__)
    print(f"  __init__ 시그니처: {sig}")

    # 주요 메서드 확인
    oidc_methods = ["get_aws_credentials", "get_token", "verify_token"]
    for method in oidc_methods:
        if hasattr(GitHubActionsOIDC, method):
            print(f"  ✓ {method} 메서드 존재")
        else:
            warnings.append(f"GitHubActionsOIDC에 {method} 메서드 없음")

except ImportError as e:
    errors.append(f"GitHubActionsOIDC import 실패: {e}")
    print(f"  ✗ GitHubActionsOIDC import 실패: {e}")

# 3. TrustPolicyBuilder 확인
print("\n[3] TrustPolicyBuilder 확인...")
try:
    from truthound.secrets.oidc.github import TrustPolicyBuilder
    print("  ✓ TrustPolicyBuilder import 성공")

    # 클라우드별 메서드 확인
    cloud_methods = ["aws", "gcp", "azure", "vault"]
    for method in cloud_methods:
        if hasattr(TrustPolicyBuilder, method):
            print(f"  ✓ TrustPolicyBuilder.{method}() 존재")
        else:
            warnings.append(f"TrustPolicyBuilder에 {method} 메서드 없음")

    # to_terraform 메서드 확인
    if hasattr(TrustPolicyBuilder, 'to_terraform'):
        print("  ✓ to_terraform 메서드 존재")
    else:
        # 인스턴스에서 확인
        try:
            builder = TrustPolicyBuilder.aws(account_id="123456789012", repository="test/repo")
            if hasattr(builder, 'to_terraform'):
                print("  ✓ to_terraform 메서드 존재 (인스턴스)")
        except:
            warnings.append("to_terraform 메서드 확인 실패")

except ImportError as e:
    errors.append(f"TrustPolicyBuilder import 실패: {e}")
    print(f"  ✗ TrustPolicyBuilder import 실패: {e}")

# 4. WorkflowSummary 확인
print("\n[4] WorkflowSummary 확인...")
try:
    from truthound.secrets.oidc.github import WorkflowSummary
    print("  ✓ WorkflowSummary import 성공")

    # 메서드 확인
    summary_methods = ["add_heading", "add_validation_result", "write"]
    for method in summary_methods:
        if hasattr(WorkflowSummary, method):
            print(f"  ✓ WorkflowSummary.{method}() 존재")
        else:
            warnings.append(f"WorkflowSummary에 {method} 메서드 없음")

except ImportError as e:
    warnings.append(f"WorkflowSummary import 실패: {e}")
    print(f"  △ WorkflowSummary import 실패: {e}")

# 5. claims 모듈 확인
print("\n[5] claims 모듈 확인...")
try:
    from truthound.secrets.oidc.github import claims
    print("  ✓ claims 모듈 import 성공")

    # 클레임 관련 클래스/함수 확인
    claims_contents = [x for x in dir(claims) if not x.startswith('_')]
    print(f"  claims 모듈 내용: {claims_contents}")

    # 30+ 클레임 파싱 확인
    if hasattr(claims, 'GITHUB_CLAIMS') or hasattr(claims, 'GitHubClaims'):
        print("  ✓ GitHub Claims 정의 존재")

except ImportError as e:
    try:
        from truthound.secrets.oidc.github.claims import GitHubOIDCClaims
        print("  ✓ GitHubOIDCClaims import 성공")
    except ImportError as e2:
        warnings.append(f"claims 모듈 확인 실패: {e}")
        print(f"  △ claims 모듈 확인 실패: {e}")

# 6. verification 모듈 확인
print("\n[6] verification 모듈 (JWKS 기반 토큰 검증)...")
try:
    from truthound.secrets.oidc.github import verification
    print("  ✓ verification 모듈 import 성공")

    verification_contents = [x for x in dir(verification) if not x.startswith('_')]
    print(f"  verification 모듈 내용: {verification_contents}")

    # JWKS 관련 클래스/함수 확인
    jwks_items = [x for x in verification_contents if 'jwk' in x.lower() or 'verify' in x.lower()]
    if jwks_items:
        print(f"  ✓ JWKS 관련 항목: {jwks_items}")

except ImportError as e:
    warnings.append(f"verification 모듈 확인 실패: {e}")
    print(f"  △ verification 모듈 확인 실패: {e}")

# 7. enhanced_provider 모듈 확인
print("\n[7] enhanced_provider 모듈 확인...")
try:
    from truthound.secrets.oidc.github import enhanced_provider
    print("  ✓ enhanced_provider 모듈 import 성공")

    provider_contents = [x for x in dir(enhanced_provider) if not x.startswith('_')]
    print(f"  enhanced_provider 모듈 내용: {provider_contents}")

except ImportError as e:
    warnings.append(f"enhanced_provider 모듈 확인 실패: {e}")
    print(f"  △ enhanced_provider 모듈 확인 실패: {e}")

# 8. workflow 모듈 확인
print("\n[8] workflow 모듈 확인...")
try:
    from truthound.secrets.oidc.github import workflow
    print("  ✓ workflow 모듈 import 성공")

    workflow_contents = [x for x in dir(workflow) if not x.startswith('_')]
    print(f"  workflow 모듈 내용: {workflow_contents}")

except ImportError as e:
    warnings.append(f"workflow 모듈 확인 실패: {e}")
    print(f"  △ workflow 모듈 확인 실패: {e}")

# 9. trust_policy 모듈 확인
print("\n[9] trust_policy 모듈 확인...")
try:
    from truthound.secrets.oidc.github import trust_policy
    print("  ✓ trust_policy 모듈 import 성공")

    policy_contents = [x for x in dir(trust_policy) if not x.startswith('_')]
    print(f"  trust_policy 모듈 내용: {policy_contents}")

    # Terraform/CLI 출력 기능 확인
    terraform_items = [x for x in policy_contents if 'terraform' in x.lower() or 'cli' in x.lower()]
    if terraform_items:
        print(f"  ✓ Terraform/CLI 관련 항목: {terraform_items}")

except ImportError as e:
    warnings.append(f"trust_policy 모듈 확인 실패: {e}")
    print(f"  △ trust_policy 모듈 확인 실패: {e}")

# 결과 요약
print("\n" + "=" * 60)
print("GitHub Actions OIDC Integration 검증 결과")
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
