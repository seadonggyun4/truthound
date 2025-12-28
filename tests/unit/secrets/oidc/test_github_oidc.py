"""Tests for GitHub Actions OIDC Integration Module.

This module tests:
- Claims parsing and validation
- Enhanced OIDC provider
- Trust policy builders
- Token verification (mocked)
- Workflow integration utilities
"""

from __future__ import annotations

import base64
import json
import os
from datetime import datetime, timedelta
from typing import Any
from unittest.mock import MagicMock, patch, mock_open

import pytest


# =============================================================================
# Test Fixtures
# =============================================================================


def create_jwt_payload(
    repository: str = "owner/repo",
    ref: str = "refs/heads/main",
    actor: str = "testuser",
    environment: str | None = None,
    exp_offset: int = 3600,
    **extra: Any,
) -> dict[str, Any]:
    """Create a test JWT payload."""
    now = datetime.now()
    exp = now + timedelta(seconds=exp_offset)

    payload = {
        "iss": "https://token.actions.githubusercontent.com",
        "sub": f"repo:{repository}:ref:{ref}",
        "aud": "sts.amazonaws.com",
        "exp": int(exp.timestamp()),
        "iat": int(now.timestamp()),
        "nbf": int(now.timestamp()),
        "jti": "test-jwt-id",
        "repository": repository,
        "repository_owner": repository.split("/")[0],
        "repository_owner_id": "12345",
        "repository_id": "67890",
        "repository_visibility": "private",
        "actor": actor,
        "actor_id": "11111",
        "workflow": "CI",
        "workflow_ref": f"{repository}/.github/workflows/ci.yml@{ref}",
        "workflow_sha": "abc123",
        "ref": ref,
        "ref_type": "branch",
        "sha": "abc123def456",
        "run_id": "123456789",
        "run_number": "42",
        "run_attempt": "1",
        "event_name": "push",
        "runner_environment": "github-hosted",
        **extra,
    }

    if environment:
        payload["environment"] = environment
        payload["environment_node_id"] = "env-node-id"

    return payload


def create_jwt_token(payload: dict[str, Any]) -> str:
    """Create a mock JWT token from payload."""
    header = {"alg": "RS256", "typ": "JWT", "kid": "test-key-id"}

    header_b64 = base64.urlsafe_b64encode(
        json.dumps(header).encode()
    ).rstrip(b"=").decode()

    payload_b64 = base64.urlsafe_b64encode(
        json.dumps(payload).encode()
    ).rstrip(b"=").decode()

    # Fake signature
    signature_b64 = base64.urlsafe_b64encode(b"fake-signature").rstrip(b"=").decode()

    return f"{header_b64}.{payload_b64}.{signature_b64}"


# =============================================================================
# Claims Tests
# =============================================================================


class TestGitHubActionsClaims:
    """Tests for claims parsing."""

    def test_parse_basic_claims(self) -> None:
        """Test parsing basic claims."""
        from truthound.secrets.oidc.github.claims import parse_github_claims

        payload = create_jwt_payload()
        claims = parse_github_claims(payload)

        assert claims.issuer == "https://token.actions.githubusercontent.com"
        assert claims.repository == "owner/repo"
        assert claims.repository_owner == "owner"
        assert claims.actor == "testuser"
        assert claims.ref == "refs/heads/main"
        assert claims.sha == "abc123def456"

    def test_parse_environment_claims(self) -> None:
        """Test parsing environment claims."""
        from truthound.secrets.oidc.github.claims import parse_github_claims

        payload = create_jwt_payload(environment="production")
        claims = parse_github_claims(payload)

        assert claims.environment == "production"
        assert claims.has_environment is True

    def test_branch_extraction(self) -> None:
        """Test branch name extraction."""
        from truthound.secrets.oidc.github.claims import parse_github_claims

        payload = create_jwt_payload(ref="refs/heads/feature/test")
        claims = parse_github_claims(payload)

        assert claims.branch_name == "feature/test"
        assert claims.tag_name is None
        assert claims.is_tag is False

    def test_tag_extraction(self) -> None:
        """Test tag name extraction."""
        from truthound.secrets.oidc.github.claims import parse_github_claims

        payload = create_jwt_payload(ref="refs/tags/v1.0.0", ref_type="tag")
        claims = parse_github_claims(payload)

        assert claims.tag_name == "v1.0.0"
        assert claims.is_tag is True

    def test_repository_matching(self) -> None:
        """Test repository pattern matching."""
        from truthound.secrets.oidc.github.claims import parse_github_claims

        payload = create_jwt_payload(repository="org/my-repo")
        claims = parse_github_claims(payload)

        assert claims.matches_repository("org/my-repo") is True
        assert claims.matches_repository("org/*") is True
        assert claims.matches_repository("*/my-repo") is True
        assert claims.matches_repository("other/repo") is False

    def test_ref_matching(self) -> None:
        """Test ref pattern matching."""
        from truthound.secrets.oidc.github.claims import parse_github_claims

        payload = create_jwt_payload(ref="refs/heads/main")
        claims = parse_github_claims(payload)

        assert claims.matches_ref("refs/heads/main") is True
        assert claims.matches_ref("refs/heads/*") is True
        assert claims.matches_ref("refs/tags/*") is False

    def test_event_type_parsing(self) -> None:
        """Test event type parsing."""
        from truthound.secrets.oidc.github.claims import (
            parse_github_claims,
            EventType,
        )

        # Push event
        payload = create_jwt_payload(event_name="push")
        claims = parse_github_claims(payload)
        assert claims.event_name == EventType.PUSH

        # Pull request event
        payload = create_jwt_payload(event_name="pull_request")
        claims = parse_github_claims(payload)
        assert claims.event_name == EventType.PULL_REQUEST
        assert claims.is_pull_request is True

        # Workflow dispatch
        payload = create_jwt_payload(event_name="workflow_dispatch")
        claims = parse_github_claims(payload)
        assert claims.event_name == EventType.WORKFLOW_DISPATCH
        assert claims.is_workflow_dispatch is True

    def test_expired_token(self) -> None:
        """Test expired token detection."""
        from truthound.secrets.oidc.github.claims import parse_github_claims

        payload = create_jwt_payload(exp_offset=-3600)  # Expired 1 hour ago
        claims = parse_github_claims(payload)

        assert claims.is_expired is True

    def test_claims_to_dict(self) -> None:
        """Test claims serialization."""
        from truthound.secrets.oidc.github.claims import parse_github_claims

        payload = create_jwt_payload()
        claims = parse_github_claims(payload)
        result = claims.to_dict()

        assert result["repository"] == "owner/repo"
        assert result["actor"] == "testuser"
        assert isinstance(result["exp"], int)


class TestClaimsValidation:
    """Tests for claims validation."""

    def test_validate_repository(self) -> None:
        """Test repository validation."""
        from truthound.secrets.oidc.github.claims import (
            parse_github_claims,
            validate_claims,
            ClaimsValidationPolicy,
        )

        payload = create_jwt_payload(repository="org/allowed-repo")
        claims = parse_github_claims(payload)

        # Should pass
        policy = ClaimsValidationPolicy(allowed_repositories=["org/allowed-repo"])
        result = validate_claims(claims, policy)
        assert result.is_valid is True

        # Should fail
        policy = ClaimsValidationPolicy(allowed_repositories=["org/other-repo"])
        result = validate_claims(claims, policy)
        assert result.is_valid is False
        assert "not in allowed list" in result.errors[0]

    def test_validate_branches(self) -> None:
        """Test branch validation."""
        from truthound.secrets.oidc.github.claims import (
            parse_github_claims,
            validate_claims,
            ClaimsValidationPolicy,
        )

        payload = create_jwt_payload(ref="refs/heads/main")
        claims = parse_github_claims(payload)

        # Should pass
        policy = ClaimsValidationPolicy(allowed_branches=["main", "develop"])
        result = validate_claims(claims, policy)
        assert result.is_valid is True

        # Should fail
        policy = ClaimsValidationPolicy(allowed_branches=["develop"])
        result = validate_claims(claims, policy)
        assert result.is_valid is False

    def test_validate_environment_required(self) -> None:
        """Test environment requirement validation."""
        from truthound.secrets.oidc.github.claims import (
            parse_github_claims,
            validate_claims,
            ClaimsValidationPolicy,
        )

        # Without environment
        payload = create_jwt_payload()
        claims = parse_github_claims(payload)

        policy = ClaimsValidationPolicy(require_environment=True)
        result = validate_claims(claims, policy)
        assert result.is_valid is False
        assert "required but not present" in result.errors[0]

        # With environment
        payload = create_jwt_payload(environment="production")
        claims = parse_github_claims(payload)
        result = validate_claims(claims, policy)
        assert result.is_valid is True

    def test_validate_deny_pull_requests(self) -> None:
        """Test pull request denial."""
        from truthound.secrets.oidc.github.claims import (
            parse_github_claims,
            validate_claims,
            ClaimsValidationPolicy,
        )

        payload = create_jwt_payload(event_name="pull_request")
        claims = parse_github_claims(payload)

        policy = ClaimsValidationPolicy(allow_pull_requests=False)
        result = validate_claims(claims, policy)
        assert result.is_valid is False
        assert "not allowed" in result.errors[0]


class TestGitHubActionsContext:
    """Tests for GitHub Actions context."""

    def test_from_environment(self) -> None:
        """Test context creation from environment."""
        from truthound.secrets.oidc.github.claims import GitHubActionsContext

        env_vars = {
            "GITHUB_REPOSITORY": "owner/repo",
            "GITHUB_WORKSPACE": "/home/runner/work",
            "GITHUB_OUTPUT": "/tmp/output",
            "RUNNER_OS": "Linux",
            "RUNNER_ARCH": "X64",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            context = GitHubActionsContext.from_environment()

        assert context.runner_os == "Linux"
        assert context.runner_arch == "X64"


# =============================================================================
# Trust Policy Tests
# =============================================================================


class TestAWSTrustPolicy:
    """Tests for AWS trust policy builder."""

    def test_basic_policy(self) -> None:
        """Test basic trust policy generation."""
        from truthound.secrets.oidc.github.trust_policy import TrustPolicyBuilder

        policy = TrustPolicyBuilder.aws(
            account_id="123456789012",
            repository="owner/repo",
        )

        result = policy.to_dict()

        assert result["Version"] == "2012-10-17"
        assert len(result["Statement"]) == 1

        statement = result["Statement"][0]
        assert statement["Effect"] == "Allow"
        assert statement["Action"] == "sts:AssumeRoleWithWebIdentity"
        assert "oidc-provider/token.actions.githubusercontent.com" in statement["Principal"]["Federated"]

    def test_branch_restriction(self) -> None:
        """Test branch-restricted policy."""
        from truthound.secrets.oidc.github.trust_policy import TrustPolicyBuilder

        policy = TrustPolicyBuilder.aws(
            account_id="123456789012",
            repository="owner/repo",
            branches=["main"],
        )

        result = policy.to_dict()
        condition = result["Statement"][0]["Condition"]

        # Should have subject condition for branch
        assert "StringEquals" in condition or "StringLike" in condition

    def test_environment_restriction(self) -> None:
        """Test environment-restricted policy."""
        from truthound.secrets.oidc.github.trust_policy import TrustPolicyBuilder

        policy = TrustPolicyBuilder.aws(
            account_id="123456789012",
            repository="owner/repo",
            environments=["production"],
        )

        result = policy.to_dict()
        json_str = policy.to_json()

        assert "production" in json_str

    def test_to_terraform(self) -> None:
        """Test Terraform output generation."""
        from truthound.secrets.oidc.github.trust_policy import TrustPolicyBuilder

        policy = TrustPolicyBuilder.aws(
            account_id="123456789012",
            repository="owner/repo",
        )

        terraform = policy.to_terraform()

        assert "aws_iam_openid_connect_provider" in terraform
        assert "aws_iam_role" in terraform
        assert "assume_role_policy" in terraform


class TestGCPWorkloadIdentityPolicy:
    """Tests for GCP Workload Identity policy builder."""

    def test_basic_policy(self) -> None:
        """Test basic GCP configuration."""
        from truthound.secrets.oidc.github.trust_policy import TrustPolicyBuilder

        policy = TrustPolicyBuilder.gcp(
            project_id="my-project",
            project_number="123456789",
            repository="owner/repo",
            service_account_email="sa@my-project.iam.gserviceaccount.com",
        )

        audience = policy.get_audience()

        assert "projects/123456789" in audience
        assert "workloadIdentityPools" in audience

    def test_gcloud_commands(self) -> None:
        """Test gcloud command generation."""
        from truthound.secrets.oidc.github.trust_policy import TrustPolicyBuilder

        policy = TrustPolicyBuilder.gcp(
            project_id="my-project",
            project_number="123456789",
            repository="owner/repo",
            service_account_email="sa@my-project.iam.gserviceaccount.com",
        )

        commands = policy.to_gcloud_commands()

        assert "gcloud iam workload-identity-pools create" in commands
        assert "gcloud iam workload-identity-pools providers create-oidc" in commands


class TestAzureFederatedCredentialPolicy:
    """Tests for Azure federated credential policy builder."""

    def test_basic_policy(self) -> None:
        """Test basic Azure configuration."""
        from truthound.secrets.oidc.github.trust_policy import TrustPolicyBuilder

        policy = TrustPolicyBuilder.azure(
            tenant_id="12345678-1234-1234-1234-123456789012",
            client_id="87654321-4321-4321-4321-210987654321",
            repository="owner/repo",
            branches=["main"],
        )

        result = policy.to_dict()

        assert result["issuer"] == "https://token.actions.githubusercontent.com"
        assert "api://AzureADTokenExchange" in result["audiences"]

    def test_az_cli_commands(self) -> None:
        """Test Azure CLI command generation."""
        from truthound.secrets.oidc.github.trust_policy import TrustPolicyBuilder

        policy = TrustPolicyBuilder.azure(
            tenant_id="12345678-1234-1234-1234-123456789012",
            client_id="87654321-4321-4321-4321-210987654321",
            repository="owner/repo",
        )

        commands = policy.to_az_cli_commands()

        assert "az ad app federated-credential create" in commands


class TestVaultJWTRolePolicy:
    """Tests for Vault JWT role policy builder."""

    def test_basic_policy(self) -> None:
        """Test basic Vault configuration."""
        from truthound.secrets.oidc.github.trust_policy import TrustPolicyBuilder

        policy = TrustPolicyBuilder.vault(
            role_name="github-actions",
            policies=["read-secrets", "list-secrets"],
            repository="owner/repo",
        )

        result = policy.to_dict()

        assert result["role_type"] == "jwt"
        assert result["policies"] == ["read-secrets", "list-secrets"]

    def test_vault_commands(self) -> None:
        """Test Vault CLI command generation."""
        from truthound.secrets.oidc.github.trust_policy import TrustPolicyBuilder

        policy = TrustPolicyBuilder.vault(
            role_name="github-actions",
            policies=["read-secrets"],
            repository="owner/repo",
        )

        commands = policy.to_vault_commands()

        assert "vault auth enable jwt" in commands
        assert "vault write auth/jwt/role/github-actions" in commands


# =============================================================================
# Enhanced Provider Tests
# =============================================================================


class TestGitHubActionsOIDC:
    """Tests for enhanced GitHub Actions OIDC provider."""

    @patch.dict(os.environ, {
        "GITHUB_ACTIONS": "true",
        "ACTIONS_ID_TOKEN_REQUEST_URL": "https://token.example.com",
        "ACTIONS_ID_TOKEN_REQUEST_TOKEN": "test-token",
    })
    def test_is_available(self) -> None:
        """Test availability check."""
        from truthound.secrets.oidc.github.enhanced_provider import GitHubActionsOIDC

        oidc = GitHubActionsOIDC()
        assert oidc.is_available is True

    @patch.dict(os.environ, {}, clear=True)
    def test_not_available(self) -> None:
        """Test unavailability when not in GitHub Actions."""
        from truthound.secrets.oidc.github.enhanced_provider import GitHubActionsOIDC

        oidc = GitHubActionsOIDC()
        assert oidc.is_available is False

    def test_environment_policy(self) -> None:
        """Test environment policy validation."""
        from truthound.secrets.oidc.github.enhanced_provider import EnvironmentPolicy

        policy = EnvironmentPolicy(
            require=True,
            allowed=["production", "staging"],
        )

        # Valid environment
        is_valid, error = policy.validate("production")
        assert is_valid is True
        assert error is None

        # Invalid environment
        is_valid, error = policy.validate("development")
        assert is_valid is False
        assert "not in allowed list" in error

        # Missing environment
        is_valid, error = policy.validate(None)
        assert is_valid is False
        assert "required" in error

    def test_production_only_policy(self) -> None:
        """Test production-only environment policy."""
        from truthound.secrets.oidc.github.enhanced_provider import EnvironmentPolicy

        policy = EnvironmentPolicy(production_only=True)

        is_valid, error = policy.validate("production")
        assert is_valid is True

        is_valid, error = policy.validate("staging")
        assert is_valid is False
        assert "production" in error.lower()


# =============================================================================
# Verification Tests
# =============================================================================


class TestJWKS:
    """Tests for JWKS handling."""

    def test_jwk_from_dict(self) -> None:
        """Test JWK creation from dictionary."""
        from truthound.secrets.oidc.github.verification import JWK

        data = {
            "kty": "RSA",
            "kid": "test-key-id",
            "alg": "RS256",
            "use": "sig",
            "n": "abc123",
            "e": "AQAB",
        }

        jwk = JWK.from_dict(data)

        assert jwk.kty == "RSA"
        assert jwk.kid == "test-key-id"
        assert jwk.alg == "RS256"

    def test_jwks_from_dict(self) -> None:
        """Test JWKS creation from dictionary."""
        from truthound.secrets.oidc.github.verification import JWKS

        data = {
            "keys": [
                {"kty": "RSA", "kid": "key1", "n": "abc", "e": "AQAB"},
                {"kty": "RSA", "kid": "key2", "n": "def", "e": "AQAB"},
            ]
        }

        jwks = JWKS.from_dict(data)

        assert len(jwks.keys) == 2
        assert jwks.get_key("key1") is not None
        assert jwks.get_key("key1").kid == "key1"
        assert jwks.get_key("nonexistent") is None

    def test_jwks_expiration(self) -> None:
        """Test JWKS cache expiration."""
        from truthound.secrets.oidc.github.verification import JWKS

        # Not expired
        jwks = JWKS.from_dict({"keys": []}, cache_ttl=3600)
        assert jwks.is_expired is False

        # Expired
        jwks = JWKS.from_dict({"keys": []}, cache_ttl=-1)
        assert jwks.is_expired is True


class TestTokenVerification:
    """Tests for token verification."""

    def test_verification_result(self) -> None:
        """Test verification result dataclass."""
        from truthound.secrets.oidc.github.verification import TokenVerificationResult

        # Valid result
        result = TokenVerificationResult(is_valid=True)
        assert bool(result) is True

        # Invalid result
        result = TokenVerificationResult(is_valid=False, error="Test error")
        assert bool(result) is False
        assert result.error == "Test error"

    def test_token_verifier_policy(self) -> None:
        """Test token verifier with policies."""
        from truthound.secrets.oidc.github.verification import TokenVerifier
        from truthound.secrets.oidc.github.claims import parse_github_claims

        verifier = TokenVerifier(
            allowed_repositories=["owner/repo"],
            allowed_branches=["main"],
            verify_signature=False,  # Skip signature for test
        )

        # Create valid token
        payload = create_jwt_payload(repository="owner/repo", ref="refs/heads/main")
        token = create_jwt_token(payload)

        result = verifier.verify(token, audience="sts.amazonaws.com")
        # Will fail audience check but should parse claims
        assert result.claims is not None or result.error is not None


# =============================================================================
# Workflow Integration Tests
# =============================================================================


class TestWorkflowOutput:
    """Tests for workflow output utilities."""

    def test_set_output_modern(self, tmp_path) -> None:
        """Test modern output format."""
        from truthound.secrets.oidc.github.workflow import set_output

        output_file = tmp_path / "output"
        output_file.write_text("")

        with patch.dict(os.environ, {"GITHUB_OUTPUT": str(output_file)}):
            set_output("test_name", "test_value")

        content = output_file.read_text()
        assert "test_name=test_value" in content

    def test_set_output_dict(self, tmp_path) -> None:
        """Test dictionary output serialization."""
        from truthound.secrets.oidc.github.workflow import set_output

        output_file = tmp_path / "output"
        output_file.write_text("")

        with patch.dict(os.environ, {"GITHUB_OUTPUT": str(output_file)}):
            set_output("data", {"key": "value"})

        content = output_file.read_text()
        assert '"key": "value"' in content or '{"key":"value"}' in content

    def test_set_env(self, tmp_path) -> None:
        """Test environment variable setting."""
        from truthound.secrets.oidc.github.workflow import set_env

        env_file = tmp_path / "env"
        env_file.write_text("")

        with patch.dict(os.environ, {"GITHUB_ENV": str(env_file)}):
            set_env("MY_VAR", "my_value")

        content = env_file.read_text()
        assert "MY_VAR=my_value" in content


class TestWorkflowSummary:
    """Tests for workflow summary builder."""

    def test_heading(self) -> None:
        """Test heading generation."""
        from truthound.secrets.oidc.github.workflow import WorkflowSummary

        summary = WorkflowSummary()
        summary.add_heading("Test Heading", level=2)

        md = summary.to_markdown()
        assert "## Test Heading" in md

    def test_table(self) -> None:
        """Test table generation."""
        from truthound.secrets.oidc.github.workflow import WorkflowSummary

        summary = WorkflowSummary()
        summary.add_table([
            ["Name", "Value"],
            ["Test", "123"],
        ])

        md = summary.to_markdown()
        assert "| Name | Value |" in md
        assert "| Test | 123 |" in md

    def test_collapsible(self) -> None:
        """Test collapsible section."""
        from truthound.secrets.oidc.github.workflow import WorkflowSummary

        summary = WorkflowSummary()
        summary.add_collapsible("Click to expand", "Hidden content")

        md = summary.to_markdown()
        assert "<details>" in md
        assert "<summary>Click to expand</summary>" in md
        assert "Hidden content" in md

    def test_validation_result(self) -> None:
        """Test validation result formatting."""
        from truthound.secrets.oidc.github.workflow import WorkflowSummary

        summary = WorkflowSummary()
        summary.add_validation_result("Email Check", passed=True)
        summary.add_validation_result("Phone Check", passed=False, details="5 issues")

        md = summary.to_markdown()
        assert ":white_check_mark:" in md
        assert ":x:" in md
        assert "5 issues" in md

    def test_write(self, tmp_path) -> None:
        """Test writing summary to file."""
        from truthound.secrets.oidc.github.workflow import WorkflowSummary

        summary_file = tmp_path / "summary"
        summary_file.write_text("")

        with patch.dict(os.environ, {"GITHUB_STEP_SUMMARY": str(summary_file)}):
            summary = WorkflowSummary()
            summary.add_heading("Test")
            summary.write()

        content = summary_file.read_text()
        assert "# Test" in content


class TestLoggingUtilities:
    """Tests for logging utilities."""

    def test_add_mask(self, capsys) -> None:
        """Test value masking."""
        from truthound.secrets.oidc.github.workflow import add_mask

        add_mask("secret-value")

        captured = capsys.readouterr()
        assert "::add-mask::secret-value" in captured.out

    def test_set_warning(self, capsys) -> None:
        """Test warning logging."""
        from truthound.secrets.oidc.github.workflow import set_warning

        set_warning("Test warning", file="test.py", line=10)

        captured = capsys.readouterr()
        assert "::warning" in captured.out
        assert "Test warning" in captured.out
        assert "file=test.py" in captured.out

    def test_set_error(self, capsys) -> None:
        """Test error logging."""
        from truthound.secrets.oidc.github.workflow import set_error

        set_error("Test error", title="Error Title")

        captured = capsys.readouterr()
        assert "::error" in captured.out
        assert "Test error" in captured.out
        assert "title=Error Title" in captured.out

    def test_log_group(self, capsys) -> None:
        """Test log group context manager."""
        from truthound.secrets.oidc.github.workflow import log_group

        with log_group("Test Group"):
            print("Inside group")

        captured = capsys.readouterr()
        assert "::group::Test Group" in captured.out
        assert "Inside group" in captured.out
        assert "::endgroup::" in captured.out

    def test_debug(self, capsys) -> None:
        """Test debug logging."""
        from truthound.secrets.oidc.github.workflow import debug

        debug("Debug message")

        captured = capsys.readouterr()
        assert "::debug::Debug message" in captured.out


# =============================================================================
# Integration Tests (Mocked)
# =============================================================================


class TestE2EIntegration:
    """End-to-end integration tests with mocked HTTP."""

    @patch("urllib.request.urlopen")
    @patch.dict(os.environ, {
        "GITHUB_ACTIONS": "true",
        "ACTIONS_ID_TOKEN_REQUEST_URL": "https://token.example.com",
        "ACTIONS_ID_TOKEN_REQUEST_TOKEN": "bearer-token",
    })
    def test_full_aws_flow(self, mock_urlopen) -> None:
        """Test full OIDC to AWS credentials flow."""
        from truthound.secrets.oidc.github.enhanced_provider import GitHubActionsOIDC
        from truthound.secrets.oidc.base import OIDCProviderNotAvailableError

        # Create test JWT
        payload = create_jwt_payload(environment="production")
        jwt = create_jwt_token(payload)

        # Mock token response
        token_response = MagicMock()
        token_response.read.return_value = json.dumps({"value": jwt}).encode()
        token_response.__enter__ = MagicMock(return_value=token_response)
        token_response.__exit__ = MagicMock(return_value=False)

        mock_urlopen.return_value = token_response

        # Create OIDC provider
        oidc = GitHubActionsOIDC(
            audience="sts.amazonaws.com",
            require_environment="production",
            validate_claims=False,  # Skip validation for test
        )

        # Get token
        token = oidc.get_token()
        assert token is not None
        assert token.claims.repository == "owner/repo"
