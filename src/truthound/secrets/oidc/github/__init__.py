"""GitHub Actions OIDC Integration Module.

This module provides enhanced GitHub Actions OIDC support including:
- Advanced claims parsing and validation
- Trust policy builders for AWS/GCP/Azure
- Deployment environment integration
- Reusable workflow support
- JWKS-based token verification
- Workflow output integration

Example:
    >>> from truthound.secrets.oidc.github import (
    ...     GitHubActionsOIDC,
    ...     TrustPolicyBuilder,
    ...     GitHubClaimsValidator,
    ... )
    >>>
    >>> # Get credentials with enhanced features
    >>> oidc = GitHubActionsOIDC(
    ...     audience="sts.amazonaws.com",
    ...     require_environment="production",
    ... )
    >>> credentials = oidc.get_aws_credentials(
    ...     role_arn="arn:aws:iam::123456789012:role/my-role",
    ... )
    >>>
    >>> # Generate trust policy for AWS
    >>> policy = TrustPolicyBuilder.aws(
    ...     account_id="123456789012",
    ...     repository="owner/repo",
    ...     branches=["main", "develop"],
    ...     environments=["production"],
    ... )
"""

from truthound.secrets.oidc.github.claims import (
    GitHubActionsClaims,
    GitHubActionsContext,
    parse_github_claims,
    validate_claims,
)

from truthound.secrets.oidc.github.enhanced_provider import (
    GitHubActionsOIDC,
    GitHubActionsOIDCConfig,
    EnvironmentPolicy,
    WorkflowPolicy,
)

from truthound.secrets.oidc.github.trust_policy import (
    TrustPolicyBuilder,
    AWSTrustPolicy,
    GCPWorkloadIdentityPolicy,
    AzureFederatedCredentialPolicy,
    VaultJWTRolePolicy,
)

from truthound.secrets.oidc.github.verification import (
    JWKSVerifier,
    TokenVerifier,
    GitHubActionsJWKS,
    verify_token,
)

from truthound.secrets.oidc.github.workflow import (
    GitHubActionsOutput,
    set_output,
    set_env,
    add_mask,
    set_failed,
    set_warning,
    log_group,
    create_summary,
    WorkflowSummary,
)


__all__ = [
    # Claims
    "GitHubActionsClaims",
    "GitHubActionsContext",
    "parse_github_claims",
    "validate_claims",
    # Enhanced Provider
    "GitHubActionsOIDC",
    "GitHubActionsOIDCConfig",
    "EnvironmentPolicy",
    "WorkflowPolicy",
    # Trust Policy
    "TrustPolicyBuilder",
    "AWSTrustPolicy",
    "GCPWorkloadIdentityPolicy",
    "AzureFederatedCredentialPolicy",
    "VaultJWTRolePolicy",
    # Verification
    "JWKSVerifier",
    "TokenVerifier",
    "GitHubActionsJWKS",
    "verify_token",
    # Workflow
    "GitHubActionsOutput",
    "set_output",
    "set_env",
    "add_mask",
    "set_failed",
    "set_warning",
    "log_group",
    "create_summary",
    "WorkflowSummary",
]
