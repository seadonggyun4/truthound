"""Trust Policy Builders for GitHub Actions OIDC.

This module provides builders for creating cloud provider trust policies
that allow GitHub Actions to assume roles or access resources using OIDC.

Supported Providers:
    - AWS IAM (AssumeRoleWithWebIdentity trust policy)
    - GCP Workload Identity (pool and provider configuration)
    - Azure AD (federated credential configuration)
    - HashiCorp Vault (JWT auth role configuration)

Example:
    >>> # Generate AWS trust policy
    >>> policy = TrustPolicyBuilder.aws(
    ...     account_id="123456789012",
    ...     repository="owner/repo",
    ...     branches=["main"],
    ...     environments=["production"],
    ... )
    >>> print(policy.to_json())
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


# GitHub Actions OIDC issuer
GITHUB_OIDC_ISSUER = "token.actions.githubusercontent.com"
GITHUB_OIDC_ISSUER_URL = f"https://{GITHUB_OIDC_ISSUER}"


# =============================================================================
# AWS Trust Policy
# =============================================================================


@dataclass
class AWSTrustPolicy:
    """AWS IAM trust policy for GitHub Actions OIDC.

    This generates a trust policy for an IAM role that allows
    GitHub Actions to assume the role using OIDC.

    Attributes:
        account_id: AWS account ID.
        repository: Repository name (owner/repo) or pattern.
        branches: Allowed branches (optional).
        tags: Allowed tags (optional).
        environments: Allowed environments (optional).
        actors: Allowed actors (optional).
        audience: Token audience (default: sts.amazonaws.com).
        conditions: Additional condition blocks.
    """

    account_id: str
    repository: str
    branches: list[str] | None = None
    tags: list[str] | None = None
    environments: list[str] | None = None
    actors: list[str] | None = None
    audience: str = "sts.amazonaws.com"
    conditions: dict[str, dict[str, Any]] = field(default_factory=dict)

    def _build_subject_condition(self) -> dict[str, Any]:
        """Build subject claim condition."""
        conditions: dict[str, Any] = {}

        # Build subject patterns
        subjects: list[str] = []

        if self.branches:
            for branch in self.branches:
                subjects.append(f"repo:{self.repository}:ref:refs/heads/{branch}")
        if self.tags:
            for tag in self.tags:
                subjects.append(f"repo:{self.repository}:ref:refs/tags/{tag}")
        if self.environments:
            for env in self.environments:
                subjects.append(f"repo:{self.repository}:environment:{env}")

        # If no specific constraints, allow all
        if not subjects:
            subjects.append(f"repo:{self.repository}:*")

        # Use StringEquals for single value, StringLike for patterns
        if len(subjects) == 1 and "*" not in subjects[0]:
            conditions["StringEquals"] = {
                f"{GITHUB_OIDC_ISSUER}:sub": subjects[0]
            }
        else:
            conditions["StringLike"] = {
                f"{GITHUB_OIDC_ISSUER}:sub": subjects
            }

        return conditions

    def _build_audience_condition(self) -> dict[str, Any]:
        """Build audience claim condition."""
        return {
            "StringEquals": {
                f"{GITHUB_OIDC_ISSUER}:aud": self.audience
            }
        }

    def _build_actor_condition(self) -> dict[str, Any] | None:
        """Build actor claim condition."""
        if not self.actors:
            return None

        if len(self.actors) == 1:
            return {
                "StringEquals": {
                    f"{GITHUB_OIDC_ISSUER}:actor": self.actors[0]
                }
            }
        return {
            "ForAnyValue:StringEquals": {
                f"{GITHUB_OIDC_ISSUER}:actor": self.actors
            }
        }

    def _merge_conditions(self, *condition_dicts: dict[str, Any] | None) -> dict[str, Any]:
        """Merge multiple condition dictionaries."""
        merged: dict[str, Any] = {}

        for cond in condition_dicts:
            if cond is None:
                continue
            for operator, values in cond.items():
                if operator not in merged:
                    merged[operator] = {}
                merged[operator].update(values)

        return merged

    def to_dict(self) -> dict[str, Any]:
        """Generate trust policy as dictionary."""
        # Build conditions
        conditions = self._merge_conditions(
            self._build_audience_condition(),
            self._build_subject_condition(),
            self._build_actor_condition(),
            self.conditions if self.conditions else None,
        )

        return {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {
                        "Federated": (
                            f"arn:aws:iam::{self.account_id}:oidc-provider/"
                            f"{GITHUB_OIDC_ISSUER}"
                        )
                    },
                    "Action": "sts:AssumeRoleWithWebIdentity",
                    "Condition": conditions,
                }
            ],
        }

    def to_json(self, indent: int = 2) -> str:
        """Generate trust policy as JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def to_terraform(self, resource_name: str = "github_actions") -> str:
        """Generate Terraform configuration."""
        policy_json = self.to_json(indent=2)
        return f'''
resource "aws_iam_openid_connect_provider" "{resource_name}" {{
  url             = "{GITHUB_OIDC_ISSUER_URL}"
  client_id_list  = ["{self.audience}"]
  thumbprint_list = ["6938fd4d98bab03faadb97b34396831e3780aea1"]
}}

resource "aws_iam_role" "{resource_name}_role" {{
  name = "github-actions-{resource_name}"

  assume_role_policy = jsonencode({policy_json})

  tags = {{
    Purpose = "GitHub Actions OIDC"
    Repository = "{self.repository}"
  }}
}}
'''


# =============================================================================
# GCP Workload Identity Policy
# =============================================================================


@dataclass
class GCPWorkloadIdentityPolicy:
    """GCP Workload Identity configuration for GitHub Actions OIDC.

    Generates configuration for:
    - Workload Identity Pool
    - Workload Identity Provider
    - Service Account binding

    Attributes:
        project_id: GCP project ID.
        project_number: GCP project number.
        pool_id: Workload Identity Pool ID.
        provider_id: Workload Identity Provider ID.
        service_account_email: Service account to impersonate.
        repository: Repository name (owner/repo).
        branches: Allowed branches.
        environments: Allowed environments.
    """

    project_id: str
    project_number: str
    pool_id: str = "github-actions-pool"
    provider_id: str = "github-actions-provider"
    service_account_email: str = ""
    repository: str = ""
    branches: list[str] | None = None
    environments: list[str] | None = None

    def get_audience(self) -> str:
        """Get the Workload Identity audience URL."""
        return (
            f"//iam.googleapis.com/projects/{self.project_number}/"
            f"locations/global/workloadIdentityPools/{self.pool_id}/"
            f"providers/{self.provider_id}"
        )

    def _build_attribute_mapping(self) -> dict[str, str]:
        """Build attribute mapping for provider."""
        return {
            "google.subject": "assertion.sub",
            "attribute.repository": "assertion.repository",
            "attribute.repository_owner": "assertion.repository_owner",
            "attribute.actor": "assertion.actor",
            "attribute.ref": "assertion.ref",
            "attribute.environment": "assertion.environment",
            "attribute.workflow": "assertion.workflow",
        }

    def _build_attribute_condition(self) -> str:
        """Build CEL condition for provider."""
        conditions: list[str] = []

        if self.repository:
            conditions.append(f'attribute.repository == "{self.repository}"')

        if self.branches:
            branch_conditions = " || ".join(
                f'attribute.ref == "refs/heads/{branch}"'
                for branch in self.branches
            )
            conditions.append(f"({branch_conditions})")

        if self.environments:
            env_conditions = " || ".join(
                f'attribute.environment == "{env}"'
                for env in self.environments
            )
            conditions.append(f"({env_conditions})")

        return " && ".join(conditions) if conditions else ""

    def to_gcloud_commands(self) -> str:
        """Generate gcloud CLI commands."""
        commands = []

        # Create workload identity pool
        commands.append(f'''
# Create Workload Identity Pool
gcloud iam workload-identity-pools create {self.pool_id} \\
  --project="{self.project_id}" \\
  --location="global" \\
  --display-name="GitHub Actions Pool"
''')

        # Create provider
        attribute_condition = self._build_attribute_condition()
        condition_flag = f'--attribute-condition=\'{attribute_condition}\' \\' if attribute_condition else ""

        commands.append(f'''
# Create Workload Identity Provider
gcloud iam workload-identity-pools providers create-oidc {self.provider_id} \\
  --project="{self.project_id}" \\
  --location="global" \\
  --workload-identity-pool="{self.pool_id}" \\
  --display-name="GitHub Actions Provider" \\
  --issuer-uri="{GITHUB_OIDC_ISSUER_URL}" \\
  --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository,attribute.actor=assertion.actor" \\
  {condition_flag}
''')

        # Grant service account impersonation
        if self.service_account_email:
            principal = (
                f"principalSet://iam.googleapis.com/projects/{self.project_number}/"
                f"locations/global/workloadIdentityPools/{self.pool_id}/"
                f"attribute.repository/{self.repository}"
            )
            commands.append(f'''
# Grant Service Account Impersonation
gcloud iam service-accounts add-iam-policy-binding {self.service_account_email} \\
  --project="{self.project_id}" \\
  --role="roles/iam.workloadIdentityUser" \\
  --member="{principal}"
''')

        return "\n".join(commands)

    def to_terraform(self) -> str:
        """Generate Terraform configuration."""
        attribute_condition = self._build_attribute_condition()

        return f'''
resource "google_iam_workload_identity_pool" "github_actions" {{
  project                   = "{self.project_id}"
  workload_identity_pool_id = "{self.pool_id}"
  display_name              = "GitHub Actions Pool"
  description               = "Workload Identity Pool for GitHub Actions"
}}

resource "google_iam_workload_identity_pool_provider" "github_actions" {{
  project                            = "{self.project_id}"
  workload_identity_pool_id          = google_iam_workload_identity_pool.github_actions.workload_identity_pool_id
  workload_identity_pool_provider_id = "{self.provider_id}"
  display_name                       = "GitHub Actions Provider"

  oidc {{
    issuer_uri = "{GITHUB_OIDC_ISSUER_URL}"
  }}

  attribute_mapping = {{
    "google.subject"           = "assertion.sub"
    "attribute.repository"     = "assertion.repository"
    "attribute.actor"          = "assertion.actor"
    "attribute.ref"            = "assertion.ref"
    "attribute.environment"    = "assertion.environment"
  }}

  attribute_condition = "{attribute_condition}"
}}

resource "google_service_account_iam_binding" "github_actions" {{
  service_account_id = "{self.service_account_email}"
  role               = "roles/iam.workloadIdentityUser"

  members = [
    "principalSet://iam.googleapis.com/${{google_iam_workload_identity_pool.github_actions.name}}/attribute.repository/{self.repository}"
  ]
}}
'''


# =============================================================================
# Azure Federated Credential Policy
# =============================================================================


@dataclass
class AzureFederatedCredentialPolicy:
    """Azure AD federated credential configuration for GitHub Actions OIDC.

    Attributes:
        tenant_id: Azure AD tenant ID.
        client_id: Azure AD application client ID.
        name: Credential name.
        repository: Repository name (owner/repo).
        branches: Allowed branches.
        environments: Allowed environments.
        tags: Allowed tags.
    """

    tenant_id: str
    client_id: str
    name: str = "github-actions"
    repository: str = ""
    branches: list[str] | None = None
    environments: list[str] | None = None
    tags: list[str] | None = None

    def _build_subject(self) -> str:
        """Build subject claim value."""
        if self.environments and len(self.environments) == 1:
            return f"repo:{self.repository}:environment:{self.environments[0]}"
        if self.branches and len(self.branches) == 1:
            return f"repo:{self.repository}:ref:refs/heads/{self.branches[0]}"
        if self.tags and len(self.tags) == 1:
            return f"repo:{self.repository}:ref:refs/tags/{self.tags[0]}"
        return f"repo:{self.repository}:ref:refs/heads/main"

    def to_dict(self) -> dict[str, Any]:
        """Generate federated credential as dictionary."""
        return {
            "name": self.name,
            "issuer": GITHUB_OIDC_ISSUER_URL,
            "subject": self._build_subject(),
            "description": f"GitHub Actions OIDC for {self.repository}",
            "audiences": ["api://AzureADTokenExchange"],
        }

    def to_json(self, indent: int = 2) -> str:
        """Generate federated credential as JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def to_az_cli_commands(self) -> str:
        """Generate Azure CLI commands."""
        subject = self._build_subject()

        return f'''
# Create federated credential
az ad app federated-credential create \\
  --id {self.client_id} \\
  --parameters '{{
    "name": "{self.name}",
    "issuer": "{GITHUB_OIDC_ISSUER_URL}",
    "subject": "{subject}",
    "description": "GitHub Actions OIDC for {self.repository}",
    "audiences": ["api://AzureADTokenExchange"]
  }}'
'''

    def to_terraform(self) -> str:
        """Generate Terraform configuration."""
        subject = self._build_subject()

        return f'''
resource "azuread_application_federated_identity_credential" "github_actions" {{
  application_object_id = azuread_application.github_actions.object_id
  display_name          = "{self.name}"
  description           = "GitHub Actions OIDC for {self.repository}"
  audiences             = ["api://AzureADTokenExchange"]
  issuer                = "{GITHUB_OIDC_ISSUER_URL}"
  subject               = "{subject}"
}}
'''


# =============================================================================
# Vault JWT Role Policy
# =============================================================================


@dataclass
class VaultJWTRolePolicy:
    """HashiCorp Vault JWT auth role configuration.

    Attributes:
        role_name: Vault role name.
        policies: List of Vault policies to attach.
        repository: Repository name (owner/repo).
        branches: Allowed branches.
        environments: Allowed environments.
        ttl: Token TTL.
        max_ttl: Maximum token TTL.
        bound_audiences: Allowed audiences.
    """

    role_name: str
    policies: list[str]
    repository: str = ""
    branches: list[str] | None = None
    environments: list[str] | None = None
    ttl: str = "1h"
    max_ttl: str = "4h"
    bound_audiences: list[str] | None = None

    def _build_bound_claims(self) -> dict[str, Any]:
        """Build bound claims for role."""
        claims: dict[str, Any] = {}

        if self.repository:
            claims["repository"] = self.repository

        if self.branches:
            claims["ref"] = [f"refs/heads/{b}" for b in self.branches]

        if self.environments:
            claims["environment"] = self.environments

        return claims

    def to_dict(self) -> dict[str, Any]:
        """Generate role configuration as dictionary."""
        bound_audiences = self.bound_audiences or ["https://vault.example.com"]

        return {
            "role_type": "jwt",
            "user_claim": "repository",
            "bound_audiences": bound_audiences,
            "bound_issuer": GITHUB_OIDC_ISSUER_URL,
            "bound_claims": self._build_bound_claims(),
            "bound_claims_type": "glob",
            "policies": self.policies,
            "ttl": self.ttl,
            "max_ttl": self.max_ttl,
        }

    def to_json(self, indent: int = 2) -> str:
        """Generate role configuration as JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def to_vault_commands(self) -> str:
        """Generate Vault CLI commands."""
        bound_claims = json.dumps(self._build_bound_claims())
        bound_audiences = self.bound_audiences or ["https://vault.example.com"]
        audiences_str = ",".join(bound_audiences)

        return f'''
# Enable JWT auth backend (if not already enabled)
vault auth enable jwt

# Configure JWT auth backend with GitHub Actions OIDC
vault write auth/jwt/config \\
  oidc_discovery_url="{GITHUB_OIDC_ISSUER_URL}" \\
  bound_issuer="{GITHUB_OIDC_ISSUER_URL}"

# Create role for GitHub Actions
vault write auth/jwt/role/{self.role_name} \\
  role_type="jwt" \\
  user_claim="repository" \\
  bound_audiences="{audiences_str}" \\
  bound_claims_type="glob" \\
  bound_claims='{bound_claims}' \\
  policies="{','.join(self.policies)}" \\
  ttl="{self.ttl}" \\
  max_ttl="{self.max_ttl}"
'''

    def to_terraform(self) -> str:
        """Generate Terraform configuration."""
        bound_claims = self._build_bound_claims()
        bound_audiences = self.bound_audiences or ["https://vault.example.com"]

        return f'''
resource "vault_jwt_auth_backend" "github_actions" {{
  path               = "jwt"
  oidc_discovery_url = "{GITHUB_OIDC_ISSUER_URL}"
  bound_issuer       = "{GITHUB_OIDC_ISSUER_URL}"
}}

resource "vault_jwt_auth_backend_role" "{self.role_name}" {{
  backend        = vault_jwt_auth_backend.github_actions.path
  role_name      = "{self.role_name}"
  role_type      = "jwt"
  user_claim     = "repository"
  bound_audiences = {json.dumps(bound_audiences)}
  bound_claims   = {json.dumps(bound_claims)}
  token_policies = {json.dumps(self.policies)}
  token_ttl      = {self._parse_duration(self.ttl)}
  token_max_ttl  = {self._parse_duration(self.max_ttl)}
}}
'''

    def _parse_duration(self, duration: str) -> int:
        """Parse duration string to seconds."""
        import re
        match = re.match(r"(\d+)([hms])", duration)
        if not match:
            return 3600

        value, unit = int(match.group(1)), match.group(2)
        multipliers = {"h": 3600, "m": 60, "s": 1}
        return value * multipliers.get(unit, 1)


# =============================================================================
# Trust Policy Builder (Facade)
# =============================================================================


class TrustPolicyBuilder:
    """Facade for building trust policies for various cloud providers.

    Example:
        >>> # AWS trust policy
        >>> policy = TrustPolicyBuilder.aws(
        ...     account_id="123456789012",
        ...     repository="owner/repo",
        ...     branches=["main"],
        ... )
        >>> print(policy.to_json())
        >>>
        >>> # GCP Workload Identity
        >>> policy = TrustPolicyBuilder.gcp(
        ...     project_id="my-project",
        ...     project_number="123456789",
        ...     repository="owner/repo",
        ... )
        >>> print(policy.to_gcloud_commands())
    """

    @staticmethod
    def aws(
        account_id: str,
        repository: str,
        *,
        branches: list[str] | None = None,
        tags: list[str] | None = None,
        environments: list[str] | None = None,
        actors: list[str] | None = None,
        audience: str = "sts.amazonaws.com",
    ) -> AWSTrustPolicy:
        """Create AWS IAM trust policy.

        Args:
            account_id: AWS account ID.
            repository: Repository name (owner/repo).
            branches: Allowed branches.
            tags: Allowed tags.
            environments: Allowed environments.
            actors: Allowed actors.
            audience: Token audience.

        Returns:
            AWSTrustPolicy instance.
        """
        return AWSTrustPolicy(
            account_id=account_id,
            repository=repository,
            branches=branches,
            tags=tags,
            environments=environments,
            actors=actors,
            audience=audience,
        )

    @staticmethod
    def gcp(
        project_id: str,
        project_number: str,
        *,
        pool_id: str = "github-actions-pool",
        provider_id: str = "github-actions-provider",
        service_account_email: str = "",
        repository: str = "",
        branches: list[str] | None = None,
        environments: list[str] | None = None,
    ) -> GCPWorkloadIdentityPolicy:
        """Create GCP Workload Identity configuration.

        Args:
            project_id: GCP project ID.
            project_number: GCP project number.
            pool_id: Workload Identity Pool ID.
            provider_id: Workload Identity Provider ID.
            service_account_email: Service account to impersonate.
            repository: Repository name (owner/repo).
            branches: Allowed branches.
            environments: Allowed environments.

        Returns:
            GCPWorkloadIdentityPolicy instance.
        """
        return GCPWorkloadIdentityPolicy(
            project_id=project_id,
            project_number=project_number,
            pool_id=pool_id,
            provider_id=provider_id,
            service_account_email=service_account_email,
            repository=repository,
            branches=branches,
            environments=environments,
        )

    @staticmethod
    def azure(
        tenant_id: str,
        client_id: str,
        *,
        name: str = "github-actions",
        repository: str = "",
        branches: list[str] | None = None,
        environments: list[str] | None = None,
        tags: list[str] | None = None,
    ) -> AzureFederatedCredentialPolicy:
        """Create Azure AD federated credential configuration.

        Args:
            tenant_id: Azure AD tenant ID.
            client_id: Azure AD application client ID.
            name: Credential name.
            repository: Repository name (owner/repo).
            branches: Allowed branches.
            environments: Allowed environments.
            tags: Allowed tags.

        Returns:
            AzureFederatedCredentialPolicy instance.
        """
        return AzureFederatedCredentialPolicy(
            tenant_id=tenant_id,
            client_id=client_id,
            name=name,
            repository=repository,
            branches=branches,
            environments=environments,
            tags=tags,
        )

    @staticmethod
    def vault(
        role_name: str,
        policies: list[str],
        *,
        repository: str = "",
        branches: list[str] | None = None,
        environments: list[str] | None = None,
        ttl: str = "1h",
        max_ttl: str = "4h",
        bound_audiences: list[str] | None = None,
    ) -> VaultJWTRolePolicy:
        """Create HashiCorp Vault JWT auth role configuration.

        Args:
            role_name: Vault role name.
            policies: List of Vault policies.
            repository: Repository name (owner/repo).
            branches: Allowed branches.
            environments: Allowed environments.
            ttl: Token TTL.
            max_ttl: Maximum token TTL.
            bound_audiences: Allowed audiences.

        Returns:
            VaultJWTRolePolicy instance.
        """
        return VaultJWTRolePolicy(
            role_name=role_name,
            policies=policies,
            repository=repository,
            branches=branches,
            environments=environments,
            ttl=ttl,
            max_ttl=max_ttl,
            bound_audiences=bound_audiences,
        )
