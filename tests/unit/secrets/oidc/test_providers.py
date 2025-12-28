"""Tests for OIDC CI provider implementations.

This module tests OIDC token providers for various CI/CD platforms:
- GitHubActionsOIDCProvider
- GitLabCIOIDCProvider
- CircleCIOIDCProvider
- BitbucketPipelinesOIDCProvider
- GenericOIDCProvider
- Detection functions
"""

from __future__ import annotations

import base64
import json
import os
from datetime import datetime, timedelta
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread
from unittest.mock import patch, MagicMock

import pytest

from truthound.secrets.oidc.providers import (
    GitHubActionsOIDCProvider,
    GitHubActionsConfig,
    GitLabCIOIDCProvider,
    GitLabCIConfig,
    CircleCIOIDCProvider,
    CircleCIConfig,
    BitbucketPipelinesOIDCProvider,
    BitbucketPipelinesConfig,
    GenericOIDCProvider,
    GenericOIDCConfig,
    detect_ci_oidc_provider,
    is_oidc_available,
    get_ci_provider_type,
)
from truthound.secrets.oidc.base import (
    OIDCProviderNotAvailableError,
    OIDCTokenError,
    CIProvider,
)


# =============================================================================
# Test Utilities
# =============================================================================


def create_jwt_token(
    claims: dict,
    header: dict | None = None,
) -> str:
    """Create a test JWT token (not cryptographically valid)."""
    if header is None:
        header = {"alg": "RS256", "typ": "JWT"}

    def encode_part(data: dict) -> str:
        json_bytes = json.dumps(data).encode()
        b64 = base64.urlsafe_b64encode(json_bytes).decode()
        return b64.rstrip("=")

    header_b64 = encode_part(header)
    payload_b64 = encode_part(claims)
    signature_b64 = encode_part({"sig": "dummy"})

    return f"{header_b64}.{payload_b64}.{signature_b64}"


def create_github_claims(exp_offset: int = 3600) -> dict:
    """Create GitHub Actions token claims."""
    now = datetime.now()
    return {
        "iss": "https://token.actions.githubusercontent.com",
        "sub": "repo:owner/repo:ref:refs/heads/main",
        "aud": "sts.amazonaws.com",
        "exp": int((now + timedelta(seconds=exp_offset)).timestamp()),
        "iat": int(now.timestamp()),
        "repository": "owner/repo",
        "ref": "refs/heads/main",
        "sha": "abc123",
        "actor": "user",
        "workflow": "CI",
        "run_id": "12345",
    }


# =============================================================================
# GitHub Actions OIDC Provider Tests
# =============================================================================


class TestGitHubActionsOIDCProvider:
    """Tests for GitHubActionsOIDCProvider."""

    def test_name(self):
        """Test provider name."""
        provider = GitHubActionsOIDCProvider()
        assert provider.name == "github_actions"

    def test_is_available_true(self):
        """Test availability detection when in GitHub Actions."""
        with patch.dict(
            os.environ,
            {
                "GITHUB_ACTIONS": "true",
                "ACTIONS_ID_TOKEN_REQUEST_URL": "https://token.example.com",
                "ACTIONS_ID_TOKEN_REQUEST_TOKEN": "bearer-token",
            },
            clear=True,
        ):
            provider = GitHubActionsOIDCProvider()
            assert provider.is_available() is True

    def test_is_available_false_not_github(self):
        """Test availability when not in GitHub Actions."""
        with patch.dict(os.environ, {}, clear=True):
            provider = GitHubActionsOIDCProvider()
            assert provider.is_available() is False

    def test_is_available_false_missing_token_url(self):
        """Test availability when token URL is missing."""
        with patch.dict(
            os.environ,
            {
                "GITHUB_ACTIONS": "true",
                "ACTIONS_ID_TOKEN_REQUEST_TOKEN": "bearer-token",
            },
            clear=True,
        ):
            provider = GitHubActionsOIDCProvider()
            assert provider.is_available() is False

    def test_is_available_false_missing_token(self):
        """Test availability when bearer token is missing."""
        with patch.dict(
            os.environ,
            {
                "GITHUB_ACTIONS": "true",
                "ACTIONS_ID_TOKEN_REQUEST_URL": "https://token.example.com",
            },
            clear=True,
        ):
            provider = GitHubActionsOIDCProvider()
            assert provider.is_available() is False

    def test_config_from_init(self):
        """Test config from init parameters."""
        config = GitHubActionsConfig(
            audience="custom-audience",
            cache_ttl_seconds=600,
            enable_cache=False,
        )
        provider = GitHubActionsOIDCProvider(config=config)

        assert provider._config.audience == "custom-audience"
        assert provider._config.cache_ttl_seconds == 600

    def test_config_object(self):
        """Test config from config object."""
        config = GitHubActionsConfig(
            audience="config-audience",
            request_timeout=60.0,
        )
        provider = GitHubActionsOIDCProvider(config=config)

        assert provider._config.audience == "config-audience"
        assert provider._config.request_timeout == 60.0

    def test_get_token_not_available(self):
        """Test get_token when not available."""
        with patch.dict(os.environ, {}, clear=True):
            provider = GitHubActionsOIDCProvider()

            with pytest.raises(OIDCProviderNotAvailableError) as exc_info:
                provider.get_token()

            assert exc_info.value.provider == "github_actions"

    def test_fetch_token_success(self):
        """Test successful token fetch with mocked HTTP."""
        claims = create_github_claims()
        jwt = create_jwt_token(claims)

        with patch.dict(
            os.environ,
            {
                "GITHUB_ACTIONS": "true",
                "ACTIONS_ID_TOKEN_REQUEST_URL": "https://token.example.com",
                "ACTIONS_ID_TOKEN_REQUEST_TOKEN": "bearer-token",
            },
            clear=True,
        ):
            with patch("urllib.request.urlopen") as mock_urlopen:
                mock_response = MagicMock()
                mock_response.read.return_value = json.dumps({"value": jwt}).encode()
                mock_response.__enter__ = lambda s: s
                mock_response.__exit__ = MagicMock(return_value=False)
                mock_urlopen.return_value = mock_response

                provider = GitHubActionsOIDCProvider()
                token = provider.get_token("sts.amazonaws.com")

                assert token.provider == "github_actions"
                assert token.claims.repository == "owner/repo"

    def test_get_environment_info(self):
        """Test environment info extraction."""
        with patch.dict(
            os.environ,
            {
                "GITHUB_REPOSITORY": "owner/repo",
                "GITHUB_REF": "refs/heads/main",
                "GITHUB_SHA": "abc123",
                "GITHUB_ACTOR": "developer",
                "GITHUB_WORKFLOW": "CI",
                "GITHUB_RUN_ID": "12345",
                "GITHUB_RUN_NUMBER": "1",
                "GITHUB_EVENT_NAME": "push",
                "GITHUB_JOB": "build",
            },
            clear=True,
        ):
            provider = GitHubActionsOIDCProvider()
            info = provider.get_environment_info()

            assert info["repository"] == "owner/repo"
            assert info["ref"] == "refs/heads/main"
            assert info["sha"] == "abc123"
            assert info["actor"] == "developer"


# =============================================================================
# GitLab CI OIDC Provider Tests
# =============================================================================


class TestGitLabCIOIDCProvider:
    """Tests for GitLabCIOIDCProvider."""

    def test_name(self):
        """Test provider name."""
        provider = GitLabCIOIDCProvider()
        assert provider.name == "gitlab_ci"

    def test_is_available_with_jwt_v2(self):
        """Test availability with CI_JOB_JWT_V2."""
        claims = create_github_claims()
        jwt = create_jwt_token(claims)

        with patch.dict(
            os.environ,
            {
                "GITLAB_CI": "true",
                "CI_JOB_JWT_V2": jwt,
            },
            clear=True,
        ):
            provider = GitLabCIOIDCProvider()
            assert provider.is_available() is True

    def test_is_available_with_oidc_token(self):
        """Test availability with GITLAB_OIDC_TOKEN."""
        claims = create_github_claims()
        jwt = create_jwt_token(claims)

        with patch.dict(
            os.environ,
            {
                "GITLAB_CI": "true",
                "GITLAB_OIDC_TOKEN": jwt,
            },
            clear=True,
        ):
            provider = GitLabCIOIDCProvider()
            assert provider.is_available() is True

    def test_is_available_not_gitlab(self):
        """Test availability when not in GitLab CI."""
        with patch.dict(os.environ, {}, clear=True):
            provider = GitLabCIOIDCProvider()
            assert provider.is_available() is False

    def test_fetch_token_from_jwt_v2(self):
        """Test token fetch from CI_JOB_JWT_V2."""
        claims = create_github_claims()
        jwt = create_jwt_token(claims)

        with patch.dict(
            os.environ,
            {
                "GITLAB_CI": "true",
                "CI_JOB_JWT_V2": jwt,
            },
            clear=True,
        ):
            provider = GitLabCIOIDCProvider()
            token = provider.get_token()

            assert token.provider == "gitlab_ci"
            assert token.get_token() == jwt

    def test_fetch_token_from_custom_variable(self):
        """Test token fetch from custom environment variable."""
        claims = create_github_claims()
        jwt = create_jwt_token(claims)

        with patch.dict(
            os.environ,
            {
                "GITLAB_CI": "true",
                "MY_CUSTOM_TOKEN": jwt,
            },
            clear=True,
        ):
            provider = GitLabCIOIDCProvider(token_variable="MY_CUSTOM_TOKEN")
            token = provider.get_token()

            assert token.get_token() == jwt

    def test_get_environment_info(self):
        """Test environment info extraction."""
        with patch.dict(
            os.environ,
            {
                "CI_PROJECT_PATH": "group/project",
                "CI_PROJECT_ID": "123",
                "CI_COMMIT_REF_NAME": "main",
                "CI_COMMIT_SHA": "abc123",
                "CI_PIPELINE_ID": "456",
                "CI_PIPELINE_SOURCE": "push",
                "CI_JOB_ID": "789",
                "CI_JOB_NAME": "build",
                "GITLAB_USER_LOGIN": "user",
                "GITLAB_USER_ID": "1",
                "CI_ENVIRONMENT_NAME": "production",
            },
            clear=True,
        ):
            provider = GitLabCIOIDCProvider()
            info = provider.get_environment_info()

            assert info["project_path"] == "group/project"
            assert info["pipeline_id"] == "456"
            assert info["job_name"] == "build"


# =============================================================================
# CircleCI OIDC Provider Tests
# =============================================================================


class TestCircleCIOIDCProvider:
    """Tests for CircleCIOIDCProvider."""

    def test_name(self):
        """Test provider name."""
        provider = CircleCIOIDCProvider()
        assert provider.name == "circleci"

    def test_is_available_true(self):
        """Test availability detection in CircleCI."""
        claims = create_github_claims()
        jwt = create_jwt_token(claims)

        with patch.dict(
            os.environ,
            {
                "CIRCLECI": "true",
                "CIRCLE_OIDC_TOKEN": jwt,
            },
            clear=True,
        ):
            provider = CircleCIOIDCProvider()
            assert provider.is_available() is True

    def test_is_available_with_v2_token(self):
        """Test availability with v2 token."""
        claims = create_github_claims()
        jwt = create_jwt_token(claims)

        with patch.dict(
            os.environ,
            {
                "CIRCLECI": "true",
                "CIRCLE_OIDC_TOKEN_V2": jwt,
            },
            clear=True,
        ):
            provider = CircleCIOIDCProvider()
            assert provider.is_available() is True

    def test_is_available_false(self):
        """Test availability when not in CircleCI."""
        with patch.dict(os.environ, {}, clear=True):
            provider = CircleCIOIDCProvider()
            assert provider.is_available() is False

    def test_fetch_token(self):
        """Test token fetch."""
        claims = create_github_claims()
        jwt = create_jwt_token(claims)

        with patch.dict(
            os.environ,
            {
                "CIRCLECI": "true",
                "CIRCLE_OIDC_TOKEN": jwt,
            },
            clear=True,
        ):
            provider = CircleCIOIDCProvider()
            token = provider.get_token()

            assert token.provider == "circleci"
            assert token.get_token() == jwt

    def test_get_environment_info(self):
        """Test environment info extraction."""
        with patch.dict(
            os.environ,
            {
                "CIRCLE_PROJECT_REPONAME": "repo",
                "CIRCLE_PROJECT_USERNAME": "owner",
                "CIRCLE_BRANCH": "main",
                "CIRCLE_SHA1": "abc123",
                "CIRCLE_BUILD_NUM": "123",
                "CIRCLE_JOB": "build",
                "CIRCLE_WORKFLOW_ID": "wf-123",
                "CIRCLE_USERNAME": "user",
            },
            clear=True,
        ):
            provider = CircleCIOIDCProvider()
            info = provider.get_environment_info()

            assert info["project_reponame"] == "repo"
            assert info["branch"] == "main"


# =============================================================================
# Bitbucket Pipelines OIDC Provider Tests
# =============================================================================


class TestBitbucketPipelinesOIDCProvider:
    """Tests for BitbucketPipelinesOIDCProvider."""

    def test_name(self):
        """Test provider name."""
        provider = BitbucketPipelinesOIDCProvider()
        assert provider.name == "bitbucket_pipelines"

    def test_is_available_true(self):
        """Test availability detection in Bitbucket Pipelines."""
        claims = create_github_claims()
        jwt = create_jwt_token(claims)

        with patch.dict(
            os.environ,
            {
                "BITBUCKET_BUILD_NUMBER": "123",
                "BITBUCKET_STEP_OIDC_TOKEN": jwt,
            },
            clear=True,
        ):
            provider = BitbucketPipelinesOIDCProvider()
            assert provider.is_available() is True

    def test_is_available_false_not_bitbucket(self):
        """Test availability when not in Bitbucket."""
        with patch.dict(os.environ, {}, clear=True):
            provider = BitbucketPipelinesOIDCProvider()
            assert provider.is_available() is False

    def test_is_available_false_no_token(self):
        """Test availability when token is missing."""
        with patch.dict(
            os.environ,
            {"BITBUCKET_BUILD_NUMBER": "123"},
            clear=True,
        ):
            provider = BitbucketPipelinesOIDCProvider()
            assert provider.is_available() is False

    def test_fetch_token(self):
        """Test token fetch."""
        claims = create_github_claims()
        jwt = create_jwt_token(claims)

        with patch.dict(
            os.environ,
            {
                "BITBUCKET_BUILD_NUMBER": "123",
                "BITBUCKET_STEP_OIDC_TOKEN": jwt,
            },
            clear=True,
        ):
            provider = BitbucketPipelinesOIDCProvider()
            token = provider.get_token()

            assert token.provider == "bitbucket_pipelines"
            assert token.get_token() == jwt

    def test_get_environment_info(self):
        """Test environment info extraction."""
        with patch.dict(
            os.environ,
            {
                "BITBUCKET_REPO_SLUG": "repo",
                "BITBUCKET_REPO_UUID": "{uuid}",
                "BITBUCKET_WORKSPACE": "workspace",
                "BITBUCKET_BRANCH": "main",
                "BITBUCKET_COMMIT": "abc123",
                "BITBUCKET_BUILD_NUMBER": "123",
                "BITBUCKET_PIPELINE_UUID": "{pipeline-uuid}",
                "BITBUCKET_STEP_UUID": "{step-uuid}",
            },
            clear=True,
        ):
            provider = BitbucketPipelinesOIDCProvider()
            info = provider.get_environment_info()

            assert info["repo_slug"] == "repo"
            assert info["workspace"] == "workspace"
            assert info["branch"] == "main"


# =============================================================================
# Generic OIDC Provider Tests
# =============================================================================


class TestGenericOIDCProvider:
    """Tests for GenericOIDCProvider."""

    def test_name_from_config(self):
        """Test provider name from config."""
        config = GenericOIDCConfig(name="custom-ci")
        provider = GenericOIDCProvider(config=config)
        assert provider.name == "custom-ci"

    def test_is_available_true(self):
        """Test availability when token URL env is set."""
        config = GenericOIDCConfig(
            name="custom",
            token_url_env="CUSTOM_TOKEN_URL",
        )
        with patch.dict(
            os.environ,
            {"CUSTOM_TOKEN_URL": "https://token.example.com"},
            clear=True,
        ):
            provider = GenericOIDCProvider(config=config)
            assert provider.is_available() is True

    def test_is_available_false(self):
        """Test availability when token URL env is missing."""
        config = GenericOIDCConfig(
            name="custom",
            token_url_env="CUSTOM_TOKEN_URL",
        )
        with patch.dict(os.environ, {}, clear=True):
            provider = GenericOIDCProvider(config=config)
            assert provider.is_available() is False

    def test_fetch_token_success(self):
        """Test successful token fetch."""
        claims = create_github_claims()
        jwt = create_jwt_token(claims)

        config = GenericOIDCConfig(
            name="custom",
            token_url_env="CUSTOM_TOKEN_URL",
            token_header_value_env="CUSTOM_AUTH",
            token_response_field="id_token",
        )

        with patch.dict(
            os.environ,
            {
                "CUSTOM_TOKEN_URL": "https://token.example.com",
                "CUSTOM_AUTH": "Bearer secret",
            },
            clear=True,
        ):
            with patch("urllib.request.urlopen") as mock_urlopen:
                mock_response = MagicMock()
                mock_response.read.return_value = json.dumps({"id_token": jwt}).encode()
                mock_response.__enter__ = lambda s: s
                mock_response.__exit__ = MagicMock(return_value=False)
                mock_urlopen.return_value = mock_response

                provider = GenericOIDCProvider(config=config)
                token = provider.get_token()

                assert token.provider == "custom"
                assert token.get_token() == jwt


# =============================================================================
# Detection Functions Tests
# =============================================================================


class TestDetectionFunctions:
    """Tests for CI provider detection functions."""

    def test_detect_ci_oidc_provider_github(self):
        """Test detection of GitHub Actions."""
        claims = create_github_claims()
        jwt = create_jwt_token(claims)

        with patch.dict(
            os.environ,
            {
                "GITHUB_ACTIONS": "true",
                "ACTIONS_ID_TOKEN_REQUEST_URL": "https://token.example.com",
                "ACTIONS_ID_TOKEN_REQUEST_TOKEN": jwt,
            },
            clear=True,
        ):
            provider = detect_ci_oidc_provider()
            assert provider is not None
            assert provider.name == "github_actions"

    def test_detect_ci_oidc_provider_gitlab(self):
        """Test detection of GitLab CI."""
        claims = create_github_claims()
        jwt = create_jwt_token(claims)

        with patch.dict(
            os.environ,
            {
                "GITLAB_CI": "true",
                "CI_JOB_JWT_V2": jwt,
            },
            clear=True,
        ):
            provider = detect_ci_oidc_provider()
            assert provider is not None
            assert provider.name == "gitlab_ci"

    def test_detect_ci_oidc_provider_circleci(self):
        """Test detection of CircleCI."""
        claims = create_github_claims()
        jwt = create_jwt_token(claims)

        with patch.dict(
            os.environ,
            {
                "CIRCLECI": "true",
                "CIRCLE_OIDC_TOKEN": jwt,
            },
            clear=True,
        ):
            provider = detect_ci_oidc_provider()
            assert provider is not None
            assert provider.name == "circleci"

    def test_detect_ci_oidc_provider_none(self):
        """Test detection when not in any CI."""
        with patch.dict(os.environ, {}, clear=True):
            provider = detect_ci_oidc_provider()
            assert provider is None

    def test_is_oidc_available_true(self):
        """Test is_oidc_available returns True."""
        claims = create_github_claims()
        jwt = create_jwt_token(claims)

        with patch.dict(
            os.environ,
            {
                "GITHUB_ACTIONS": "true",
                "ACTIONS_ID_TOKEN_REQUEST_URL": "https://token.example.com",
                "ACTIONS_ID_TOKEN_REQUEST_TOKEN": jwt,
            },
            clear=True,
        ):
            assert is_oidc_available() is True

    def test_is_oidc_available_false(self):
        """Test is_oidc_available returns False."""
        with patch.dict(os.environ, {}, clear=True):
            assert is_oidc_available() is False

    def test_get_ci_provider_type_github(self):
        """Test CI provider type detection for GitHub."""
        with patch.dict(os.environ, {"GITHUB_ACTIONS": "true"}, clear=True):
            assert get_ci_provider_type() == CIProvider.GITHUB_ACTIONS

    def test_get_ci_provider_type_gitlab(self):
        """Test CI provider type detection for GitLab."""
        with patch.dict(os.environ, {"GITLAB_CI": "true"}, clear=True):
            assert get_ci_provider_type() == CIProvider.GITLAB_CI

    def test_get_ci_provider_type_circleci(self):
        """Test CI provider type detection for CircleCI."""
        with patch.dict(os.environ, {"CIRCLECI": "true"}, clear=True):
            assert get_ci_provider_type() == CIProvider.CIRCLECI

    def test_get_ci_provider_type_bitbucket(self):
        """Test CI provider type detection for Bitbucket."""
        with patch.dict(os.environ, {"BITBUCKET_BUILD_NUMBER": "123"}, clear=True):
            assert get_ci_provider_type() == CIProvider.BITBUCKET

    def test_get_ci_provider_type_jenkins(self):
        """Test CI provider type detection for Jenkins."""
        with patch.dict(os.environ, {"JENKINS_URL": "http://jenkins"}, clear=True):
            assert get_ci_provider_type() == CIProvider.JENKINS

    def test_get_ci_provider_type_unknown(self):
        """Test CI provider type detection for unknown."""
        with patch.dict(os.environ, {}, clear=True):
            assert get_ci_provider_type() == CIProvider.UNKNOWN


# =============================================================================
# Priority Tests
# =============================================================================


class TestProviderPriority:
    """Tests for provider detection priority."""

    def test_github_preferred_over_gitlab(self):
        """Test that GitHub Actions is detected before GitLab when both present."""
        claims = create_github_claims()
        jwt = create_jwt_token(claims)

        with patch.dict(
            os.environ,
            {
                "GITHUB_ACTIONS": "true",
                "ACTIONS_ID_TOKEN_REQUEST_URL": "https://token.example.com",
                "ACTIONS_ID_TOKEN_REQUEST_TOKEN": jwt,
                "GITLAB_CI": "true",
                "CI_JOB_JWT_V2": jwt,
            },
            clear=True,
        ):
            provider = detect_ci_oidc_provider()
            # GitHub Actions should be detected first
            assert provider is not None
            assert provider.name == "github_actions"
