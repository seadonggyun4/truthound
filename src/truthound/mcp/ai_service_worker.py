"""Native AI lifecycle owner; the wire cannot select providers, models or paths."""

import asyncio
import hashlib
import hmac
import json
from pathlib import Path

from truthound.mcp.ai_services import AIRuntime
from truthound.mcp.policy import MCPError


def proposal_digest(proposal):
    return hashlib.sha256(
        json.dumps(
            proposal.model_dump(mode="json"),
            sort_keys=True,
            allow_nan=False,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def public_value(value):
    """Keep native result fields while excluding operator workspace locations."""
    if isinstance(value, dict):
        return {key: public_value(item) for key, item in value.items() if key != "workspace_root"}
    if isinstance(value, list):
        return [public_value(item) for item in value]
    return value


class AIServiceOwner:
    def __init__(self, runtime, request):
        from truthound.context import TruthoundContext

        if not isinstance(runtime, AIRuntime) or not isinstance(runtime.context, TruthoundContext):
            raise MCPError("invalid_binding")
        if runtime.provider is None or not callable(
            getattr(runtime.provider, "generate_structured", None)
        ):
            raise MCPError("invalid_binding")
        root = runtime.context.root_dir.resolve()
        roots = [Path(path).resolve() for path in request["workspace_roots"]]
        if not any(root.is_relative_to(parent) for parent in roots):
            raise MCPError("access_denied")
        from truthound.mcp.contexts import validate_workspace_references

        validate_workspace_references(root)
        self.runtime = runtime

    def execute_native(self, args):
        from truthound.ai import (
            ActorRef,
            apply_proposal,
            approve_proposal,
            explain_run,
            list_analyses,
            list_proposal_approval_events,
            list_proposals,
            reject_proposal,
            show_analysis,
            show_proposal,
            suggest_suite,
        )

        runtime, command = self.runtime, args["command"]
        context = runtime.context
        from truthound.mcp.contexts import validate_workspace_references

        validate_workspace_references(context.root_dir)
        if command == "suggest":
            value = suggest_suite(
                args["prompt"],
                data=runtime.data,
                source=runtime.source,
                context=context,
                provider=runtime.provider,
                model=runtime.model,
                sample_size=args.get("sample_size", 1000),
                redact="summary_only",
            )
        elif command == "explain":
            value = explain_run(
                run=runtime.run if args.get("run_id") is None else None,
                run_id=args.get("run_id"),
                context=context,
                provider=runtime.provider,
                model=runtime.model,
                redact="summary_only",
                include_history=args.get("include_history", True),
            )
        elif command == "list_proposals":
            return {
                "proposals": [
                    public_value(item.model_dump(mode="json"))
                    for item in list_proposals(context=context, limit=args.get("limit", 100))
                ]
            }
        elif command == "proposal":
            value = show_proposal(args["proposal_id"], context=context)
        elif command == "history":
            return {
                "events": [
                    public_value(item.model_dump(mode="json"))
                    for item in list_proposal_approval_events(args["proposal_id"], context=context)
                ]
            }
        elif command in {"approve", "reject", "apply"}:
            current = show_proposal(args["proposal_id"], context=context)
            if not hmac.compare_digest(proposal_digest(current), args["expected_sha256"]):
                raise MCPError("revision_conflict")
            operation = {
                "approve": approve_proposal,
                "reject": reject_proposal,
                "apply": apply_proposal,
            }[command]
            value = operation(
                args["proposal_id"],
                context=context,
                actor=ActorRef(
                    actor_id="mcp-approved-operator", actor_name="MCP approved operator"
                ),
                comment=args.get("comment", ""),
            )
        elif command == "list_analyses":
            return {
                "analyses": [
                    public_value(item.model_dump(mode="json"))
                    for item in list_analyses(
                        context=context, run_id=args.get("run_id"), limit=args.get("limit", 100)
                    )
                ]
            }
        elif command == "analysis":
            value = show_analysis(args["analysis_id"], context=context)
        else:
            raise MCPError("invalid_arguments")
        output = public_value(value.model_dump(mode="json"))
        if command in {"suggest", "proposal", "approve", "reject", "apply"}:
            output["artifact_sha256"] = proposal_digest(
                value.proposal if command in {"approve", "reject", "apply"} else value
            )
        return output

    async def execute(self, args, budget):
        return await asyncio.to_thread(self.execute_native, args)

    async def close(self):
        pass  # Factory context owns provider/source cleanup.


if __name__ == "__main__":
    from truthound.mcp.broker_worker import main

    main(AIServiceOwner)
