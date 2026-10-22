"""Bounded, read-only previews using the native Core cron implementation."""

import tempfile
from datetime import datetime
from typing import Annotated

from pydantic import Field, field_validator

from truthound.mcp.contracts import WireModel
from truthound.mcp.operations import envelope


class SchedulePreview(WireModel):
    expression: Annotated[str, Field(min_length=1, max_length=256)]
    after: Annotated[str, Field(min_length=1, max_length=64)]
    count: Annotated[int, Field(ge=1, le=100)] = 5

    @field_validator("after")
    @classmethod
    def datetime_input(cls, value):
        datetime.fromisoformat(value)
        return value


class ScheduleResult(WireModel):
    valid: bool
    errors: list[str]
    matches: bool | None
    has_seconds: bool | None
    next_times: list[str]
    requested_count: int
    returned_count: int
    search_exhausted: bool


async def call(session, name, arguments):
    request = {
        "operation": name,
        "arguments": arguments.model_dump(mode="json"),
        "contract_version": 1,
        "max_response_bytes": session.policy.max_response_bytes,
        "max_items": session.policy.max_items,
    }
    with tempfile.TemporaryDirectory(prefix="truthound-schedule-preview-") as directory:
        return await session.service._execute_request(request, directory)


def execute(name, arguments):
    from truthound.scheduling import CronExpression, validate_expression

    errors = validate_expression(arguments.expression)
    times = []
    matches = seconds = None
    if not errors:
        expression = CronExpression.parse(arguments.expression)
        after = datetime.fromisoformat(arguments.after)
        matches = expression.matches(after)
        seconds = expression.has_seconds
        times = [value.isoformat() for value in expression.next_n(arguments.count, after)]
    return envelope(
        name,
        data={
            "valid": not errors,
            "errors": errors,
            "matches": matches,
            "has_seconds": seconds,
            "next_times": times,
            "requested_count": arguments.count,
            "returned_count": len(times),
            "search_exhausted": not errors and len(times) < arguments.count,
        },
    )
