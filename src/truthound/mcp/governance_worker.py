"""Native administrative calls with safe, explicit projections and alias selection."""

from truthound.mcp.governance import RateLimitRuntime, SecretProviderSlot, SecretRuntime
from truthound.mcp.policy import MCPError


def selected(mapping, alias):
    if alias not in mapping:
        raise MCPError("invalid_arguments")
    return mapping[alias]


def limit_result(value):
    return {
        "allowed": value.allowed,
        "remaining": value.remaining,
        "limit": value.limit,
        "reset_at": value.reset_at,
        "retry_after": value.retry_after,
        "utilization": value.utilization,
        "headers": value.to_headers(),
    }


def config_view(config):
    return {
        key: getattr(config, key)
        for key in (
            "requests_per_second",
            "requests_per_minute",
            "requests_per_hour",
            "requests_per_day",
            "burst_size",
            "window_size_seconds",
            "sub_window_count",
            "max_retry_wait",
            "warm_up_period",
            "warm_up_tokens",
        )
    } | {key: getattr(config, key).value for key in ("strategy", "scope", "action")}


class GovernanceOwner:
    def __init__(self, runtime, request):
        from truthound.ratelimit import RateLimiter
        from truthound.ratelimit.core import QuotaConfig
        from truthound.ratelimit.policy import PolicyRegistry, QuotaManager, RateLimitPolicy
        from truthound.secrets.base import BaseSecretProvider, SecretReference
        from truthound.secrets.manager import SecretManager

        if request["native_group"] == "rate_limit_services":
            if not isinstance(runtime, RateLimitRuntime) or not isinstance(
                runtime.limiter, RateLimiter
            ):
                raise MCPError("invalid_binding")
            if any(not isinstance(key, str) for key in runtime.subjects.values()):
                raise MCPError("invalid_binding")
            if runtime.policy_registry is not None and (
                not isinstance(runtime.policy_registry, PolicyRegistry)
                or runtime.policy_registry is not runtime.limiter._policy_registry
            ):
                raise MCPError("invalid_binding")
            if runtime.quota_manager is not None and (
                not isinstance(runtime.quota_manager, QuotaManager)
                or runtime.quota_manager is not runtime.limiter._quota_manager
            ):
                raise MCPError("invalid_binding")
            if any(
                not isinstance(value, RateLimitPolicy) for value in runtime.policies.values()
            ) or any(not isinstance(value, QuotaConfig) for value in runtime.quotas.values()):
                raise MCPError("invalid_binding")
        elif request["native_group"] == "secret_services":
            if not isinstance(runtime, SecretRuntime) or not isinstance(
                runtime.manager, SecretManager
            ):
                raise MCPError("invalid_binding")
            if any(
                not isinstance(value, SecretReference) for value in runtime.references.values()
            ) or any(
                not isinstance(slot, SecretProviderSlot)
                or not isinstance(slot.provider, BaseSecretProvider)
                for slot in runtime.providers.values()
            ):
                raise MCPError("invalid_binding")
        else:
            raise MCPError("invalid_binding")
        self.runtime = runtime

    async def execute(self, args, budget):
        if isinstance(self.runtime, RateLimitRuntime):
            return self.rate_limit(args)
        return self.secret(args)

    def rate_limit(self, args):
        runtime, command = self.runtime, args["command"]
        limiter = runtime.limiter
        if command == "catalog":
            offset, limit = args.get("offset", 0), args.get("limit", 100)
            return {
                key: sorted(getattr(runtime, key))[offset : offset + limit]
                for key in ("subjects", "contexts", "policies", "quotas")
            }
        if command == "config":
            return {"config": config_view(limiter.config)}
        context = selected(runtime.contexts, args["context_id"]) if args.get("context_id") else None
        if command.startswith("policy_"):
            registry = runtime.policy_registry
            if registry is None:
                raise MCPError("invalid_binding")
            if command == "policy_resolve":
                match = registry.resolve(context)
                aliases = [
                    alias
                    for alias, policy in runtime.policies.items()
                    if policy.name == match.policy_name
                ]
                return {
                    "matched": match.matched,
                    "policy_ids": aliases,
                    "config": config_view(match.config) if match.config is not None else None,
                }
            policy = selected(runtime.policies, args["policy_id"])
            if command == "policy_register":
                registry.register(policy)
                return {"registered": True}
            return {"removed": registry.unregister(policy.name)}
        if command == "quota_register":
            if runtime.quota_manager is None:
                raise MCPError("invalid_binding")
            runtime.quota_manager.register_quota(selected(runtime.quotas, args["quota_id"]))
            return {"registered": True}
        key = selected(runtime.subjects, args["subject_id"])
        if command == "acquire":
            return limit_result(
                limiter.acquire(
                    key,
                    tokens=args.get("tokens", 1),
                    wait=args.get("wait", False),
                    timeout=args.get("timeout_seconds"),
                    context=context,
                )
            )
        if command == "peek":
            return limit_result(limiter.peek(key, context=context))
        if command == "reset":
            limiter.reset(key)
            return {"reset": True}
        if runtime.quota_manager is None:
            raise MCPError("invalid_binding")
        quota = selected(runtime.quotas, args["quota_id"])
        if runtime.quota_manager.get_quota(quota.name) is None:
            raise MCPError("invalid_arguments")
        if command == "quota_check":
            return limit_result(limiter.check_quota(key, quota.name, args.get("tokens", 1)))
        if command == "quota_consume":
            limiter.consume_quota(key, quota.name, args.get("tokens", 1))
        elif command == "quota_reset":
            runtime.quota_manager.reset(key, quota.name)
        elif command != "quota_usage":
            raise MCPError("invalid_arguments")
        usage = runtime.quota_manager.get_usage(key, quota.name)
        return {
            "usage": None
            if usage is None
            else {
                key: getattr(usage, key)
                for key in (
                    "used",
                    "soft_limit",
                    "hard_limit",
                    "period_start",
                    "period_end",
                    "remaining",
                    "utilization",
                    "is_soft_exceeded",
                    "is_hard_exceeded",
                )
            }
        }

    def secret(self, args):
        from truthound.secrets.base import SecretNotFoundError, SecretValue

        runtime, command = self.runtime, args["command"]
        manager = runtime.manager
        if command == "catalog":
            registered = dict(manager.list_providers())
            offset, limit = args.get("offset", 0), args.get("limit", 100)
            return {
                "references": sorted(runtime.references)[offset : offset + limit],
                "providers": {
                    alias: {
                        "registered": manager.get_provider(slot.name) is slot.provider,
                        "priority": registered.get(slot.name),
                    }
                    for alias in sorted(runtime.providers)[offset : offset + limit]
                    for slot in (runtime.providers[alias],)
                },
            }
        if command == "resolve":
            reference = selected(runtime.references, args["reference_id"])
            try:
                value = manager.resolve_reference(reference)
            except SecretNotFoundError:
                value = None
            if value is None:
                return {"resolved": False}
            if not isinstance(value, SecretValue):
                raise MCPError("invalid_binding")
            # No value, hash, length, physical key, provider URI or metadata.
            return {
                "resolved": True,
                "expired": value.is_expired,
                "created_at": value.created_at,
                "expires_at": value.expires_at,
            }
        slot = selected(runtime.providers, args["provider_id"]) if args.get("provider_id") else None
        if command == "cache_clear":
            manager.clear_cache(slot.name if slot else None)
            return {"cleared": True}
        if command == "provider_add":
            manager.add_provider(slot.provider, name=slot.name, priority=slot.priority)
            return {"registered": True}
        if command == "provider_remove":
            return {"removed": manager.remove_provider(slot.name)}
        raise MCPError("invalid_arguments")

    async def close(self):
        if isinstance(self.runtime, SecretRuntime):
            self.runtime.manager.clear_cache()


if __name__ == "__main__":
    from truthound.mcp.broker_worker import main

    main(GovernanceOwner)
