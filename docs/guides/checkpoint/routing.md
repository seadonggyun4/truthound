# Rule-based Routing

Rule-based Routing is a system that executes different actions based on the conditions of validation results. It supports both Python expressions and Jinja2 template engine.

## Overview

```
CheckpointResult
       │
       ▼
┌──────────────────┐
│   ActionRouter   │
│  (RouteMode)     │
└────────┬─────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌───────┐ ┌───────┐
│Route 1│ │Route 2│ ...
│(Rule) │ │(Rule) │
└───┬───┘ └───┬───┘
    │         │
    ▼         ▼
[Actions]  [Actions]
```

## Core Classes

### RouteContext

Context data used for evaluating routing rules.

```python
@dataclass(frozen=True)
class RouteContext:
    """Routing context (immutable)."""
    checkpoint_name: str           # Checkpoint name
    run_id: str                    # Execution ID
    status: str                    # Result status
    data_asset: str                # Data asset
    run_time: datetime             # Execution time
    total_issues: int = 0          # Total issue count
    critical_issues: int = 0       # Critical issue count
    high_issues: int = 0           # High issue count
    medium_issues: int = 0         # Medium issue count
    low_issues: int = 0            # Low issue count
    info_issues: int = 0           # Info issue count
    pass_rate: float = 100.0       # Pass rate (0-100)
    tags: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    validation_duration_ms: float = 0.0  # Validation duration (ms)
    error: str | None = None       # Error message
```

### ActionRouter

The main class that manages routing.

```python
from truthound.checkpoint.routing import ActionRouter, Route
from truthound.checkpoint.routing.base import RouteMode

class RouteMode(str, Enum):
    """Routing mode."""
    FIRST_MATCH = "first_match"      # Execute only the first matching route
    ALL_MATCHES = "all_matches"      # Execute all matching routes
    PRIORITY_GROUP = "priority_group" # Execute the highest priority group

# Create Router
router = ActionRouter(mode=RouteMode.ALL_MATCHES)

# Add Route
router.add_route(route)
```

### Route

A mapping between rules and actions.

```python
from truthound.checkpoint.routing import Route
from truthound.checkpoint.routing.base import RoutePriority

class RoutePriority(int, Enum):
    """Route priority."""
    CRITICAL = 100
    HIGH = 80
    NORMAL = 50
    LOW = 20
    DEFAULT = 0

route = Route(
    name="critical_alerts",
    rule=SeverityRule(min_severity="critical"),
    actions=[PagerDutyAction(...)],
    priority=RoutePriority.CRITICAL,  # Or an integer value
)
```

---

## Built-in Rules (11 Types)

### AlwaysRule / NeverRule

Rules that always match or never match.

```python
from truthound.checkpoint.routing.rules import AlwaysRule, NeverRule

# For default route
always = AlwaysRule()  # Always True

# For disabling a route
never = NeverRule()  # Always False
```

### SeverityRule

Matches based on issue severity.

```python
from truthound.checkpoint.routing.rules import SeverityRule

# Match if critical issues exist
rule = SeverityRule(min_severity="critical")

# Match if 5 or more high-level issues exist
rule = SeverityRule(min_severity="high", min_count=5)

# Medium issues only (range specification)
rule = SeverityRule(min_severity="medium", max_severity="medium")

# Exactly 3 critical issues
rule = SeverityRule(min_severity="critical", exact_count=3)
```

**Severity order**: `critical` > `high` > `medium` > `low` > `info`

### IssueCountRule

Matches based on issue count.

```python
from truthound.checkpoint.routing.rules import IssueCountRule

# 10 or more issues
rule = IssueCountRule(min_issues=10)

# Between 5 and 20 issues
rule = IssueCountRule(min_issues=5, max_issues=20)

# 3 or more critical issues
rule = IssueCountRule(min_issues=3, count_type="critical")
```

**count_type**: `total`, `critical`, `high`, `medium`, `low`, `info`

### StatusRule

Matches based on checkpoint status.

```python
from truthound.checkpoint.routing.rules import StatusRule

# Failure or error
rule = StatusRule(statuses=["failure", "error"])

# All statuses except success
rule = StatusRule(statuses=["success"], negate=True)
```

### TagRule

Matches based on tag presence or value.

```python
from truthound.checkpoint.routing.rules import TagRule

# env=prod tag
rule = TagRule(tags={"env": "prod"})

# env=prod AND team=data
rule = TagRule(tags={"env": "prod", "team": "data"}, match_all=True)

# env=prod OR team=data
rule = TagRule(tags={"env": "prod", "team": "data"}, match_all=False)

# Check if 'critical' tag exists (value irrelevant)
rule = TagRule(tags={"critical": None})

# Match if tag is absent
rule = TagRule(tags={"env": "prod"}, negate=True)
```

### DataAssetRule

Matches based on data asset name pattern.

```python
from truthound.checkpoint.routing.rules import DataAssetRule

# Glob pattern
rule = DataAssetRule(pattern="sales_*")

# Regular expression
rule = DataAssetRule(pattern=r"^prod_.*_v\d+$", is_regex=True)

# Case insensitive
rule = DataAssetRule(pattern="USERS*", case_sensitive=False)
```

### MetadataRule

Matches based on metadata values.

```python
from truthound.checkpoint.routing.rules import MetadataRule

# Simple comparison
rule = MetadataRule(key_path="region", expected_value="us-east-1")

# Nested path
rule = MetadataRule(key_path="config.settings.mode", expected_value="production")

# Comparison operators
rule = MetadataRule(key_path="priority", expected_value=5, comparator="gt")  # > 5
rule = MetadataRule(key_path="priority", expected_value=5, comparator="gte")  # >= 5
rule = MetadataRule(key_path="priority", expected_value=5, comparator="lt")   # < 5

# Contains check
rule = MetadataRule(key_path="owners", expected_value="data-team", comparator="contains")

# Regular expression
rule = MetadataRule(key_path="name", expected_value=r"v\d+", comparator="regex")

# Existence check
rule = MetadataRule(key_path="special_flag", comparator="exists")
```

**comparator**: `eq`, `ne`, `gt`, `gte`, `lt`, `lte`, `contains`, `regex`, `exists`

### TimeWindowRule

Matches based on time period. Useful for distinguishing business hours from non-business hours.

```python
from truthound.checkpoint.routing.rules import TimeWindowRule

# Business hours (9 AM - 5 PM, weekdays)
rule = TimeWindowRule(
    start_time="09:00",
    end_time="17:00",
    days_of_week=[0, 1, 2, 3, 4],  # Mon-Fri
)

# Non-business hours (crossing midnight)
rule = TimeWindowRule(
    start_time="17:00",
    end_time="09:00",  # Until 9 AM the next day
)

# Weekends
rule = TimeWindowRule(
    days_of_week=[5, 6],  # Sat-Sun
)

# Specify timezone
rule = TimeWindowRule(
    start_time="09:00",
    end_time="17:00",
    timezone="America/New_York",
)
```

### PassRateRule

Matches based on pass rate.

```python
from truthound.checkpoint.routing.rules import PassRateRule

# Pass rate below 90%
rule = PassRateRule(max_rate=90.0)

# Pass rate between 50-80%
rule = PassRateRule(min_rate=50.0, max_rate=80.0)

# Pass rate 95% or above
rule = PassRateRule(min_rate=95.0)
```

### ErrorRule

Matches based on error occurrence or pattern.

```python
from truthound.checkpoint.routing.rules import ErrorRule

# If error exists
rule = ErrorRule()

# Timeout error
rule = ErrorRule(pattern=r"timeout|timed out")

# If no error
rule = ErrorRule(negate=True)
```

---

## Combinators

### AllOf (AND)

All rules must match.

```python
from truthound.checkpoint.routing import AllOf
from truthound.checkpoint.routing.rules import SeverityRule, TagRule

# Critical issues AND production environment
rule = AllOf([
    SeverityRule(min_severity="critical"),
    TagRule(tags={"env": "prod"}),
])
```

### AnyOf (OR)

At least one rule must match.

```python
from truthound.checkpoint.routing import AnyOf

# Critical issues OR error occurred
rule = AnyOf([
    SeverityRule(min_severity="critical"),
    ErrorRule(),
])
```

### NotRule (NOT)

Inverts the rule result.

```python
from truthound.checkpoint.routing import NotRule

# Not production environment
rule = NotRule(TagRule(tags={"env": "prod"}))
```

### Complex Combinations

```python
# (Critical OR Error) AND Production AND Business hours
complex_rule = AllOf([
    AnyOf([
        SeverityRule(min_severity="critical"),
        ErrorRule(),
    ]),
    TagRule(tags={"env": "prod"}),
    TimeWindowRule(start_time="09:00", end_time="18:00"),
])
```

---

## Routing Modes

### FIRST_MATCH

Executes only the first matching route.

```python
router = ActionRouter(mode=RouteMode.FIRST_MATCH)

# Evaluated in priority order
router.add_route(Route(
    name="critical",
    rule=SeverityRule(min_severity="critical"),
    actions=[PagerDutyAction(...)],
    priority=100,  # Evaluated first
))

router.add_route(Route(
    name="high",
    rule=SeverityRule(min_severity="high"),
    actions=[SlackNotification(...)],
    priority=80,  # Evaluated next
))

# If critical issues exist, only PagerDuty is called
```

### ALL_MATCHES

Executes all matching routes.

```python
router = ActionRouter(mode=RouteMode.ALL_MATCHES)

# Execute all matching routes
router.add_route(Route(
    name="always_store",
    rule=AlwaysRule(),
    actions=[StoreValidationResult(...)],
))

router.add_route(Route(
    name="failure_alert",
    rule=StatusRule(statuses=["failure"]),
    actions=[SlackNotification(...)],
))

# On failure: Both StoreValidationResult + SlackNotification are executed
```

### PRIORITY_GROUP

Executes all routes in the highest priority group.

```python
router = ActionRouter(mode=RouteMode.PRIORITY_GROUP)

router.add_route(Route(
    name="critical_pagerduty",
    rule=SeverityRule(min_severity="critical"),
    actions=[PagerDutyAction(...)],
    priority=100,
))

router.add_route(Route(
    name="critical_slack",
    rule=SeverityRule(min_severity="critical"),
    actions=[SlackNotification(...)],
    priority=100,  # Same priority
))

router.add_route(Route(
    name="high_slack",
    rule=SeverityRule(min_severity="high"),
    actions=[SlackNotification(...)],
    priority=80,
))

# Critical issues: Both routes with priority=100 are executed
# High issues only: Only the route with priority=80 is executed
```

---

## Complete Example

```python
from truthound.checkpoint import Checkpoint
from truthound.checkpoint.routing import (
    ActionRouter, Route, AllOf, AnyOf
)
from truthound.checkpoint.routing.base import RouteMode
from truthound.checkpoint.routing.rules import (
    SeverityRule, TagRule, TimeWindowRule, StatusRule, AlwaysRule
)
from truthound.checkpoint.actions import (
    StoreValidationResult, SlackNotification, PagerDutyAction, EmailNotification
)

# Create Router
router = ActionRouter(mode=RouteMode.ALL_MATCHES)

# 1. Always store results
router.add_route(Route(
    name="always_store",
    rule=AlwaysRule(),
    actions=[StoreValidationResult(store_path="./results")],
    priority=0,
))

# 2. Critical + Production → PagerDuty
router.add_route(Route(
    name="critical_prod",
    rule=AllOf([
        SeverityRule(min_severity="critical"),
        TagRule(tags={"env": "prod"}),
    ]),
    actions=[
        PagerDutyAction(routing_key="${PAGERDUTY_KEY}"),
        SlackNotification(
            webhook_url="${SLACK_WEBHOOK}",
            mention_on_failure=["@oncall"],
        ),
    ],
    priority=100,
))

# 3. Critical outside business hours → Email only
router.add_route(Route(
    name="critical_offhours",
    rule=AllOf([
        SeverityRule(min_severity="critical"),
        TimeWindowRule(start_time="18:00", end_time="09:00"),  # Non-business hours
    ]),
    actions=[EmailNotification(
        to_addresses=["oncall@example.com"],
    )],
    priority=90,
))

# 4. High issues → Slack
router.add_route(Route(
    name="high_alert",
    rule=SeverityRule(min_severity="high"),
    actions=[SlackNotification(
        webhook_url="${SLACK_WEBHOOK}",
        channel="#data-quality",
    )],
    priority=80,
))

# Connect Router to Checkpoint
checkpoint = Checkpoint(
    name="production_check",
    data_source="prod_data.parquet",
    validators=["null", "duplicate", "range"],
    router=router,
    tags={"env": "prod"},
)

result = checkpoint.run()
```

---

## YAML Configuration (RouteConfigParser)

```yaml
routes:
  - name: critical_prod
    priority: 100
    rule:
      type: all_of
      rules:
        - type: severity
          min_severity: critical
        - type: tag
          tags:
            env: prod
    actions:
      - type: pagerduty
        routing_key: ${PAGERDUTY_KEY}
      - type: slack
        webhook_url: ${SLACK_WEBHOOK}
        mention_on_failure:
          - "@oncall"

  - name: high_alert
    priority: 80
    rule:
      type: severity
      min_severity: high
    actions:
      - type: slack
        webhook_url: ${SLACK_WEBHOOK}
        channel: "#data-quality"

  - name: always_store
    priority: 0
    rule:
      type: always
    actions:
      - type: store_result
        store_path: ./results
```
