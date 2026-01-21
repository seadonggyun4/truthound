# Notification Actions

Actions for sending notifications. Supports Slack, Email, Teams, Discord, and Telegram.

## SlackNotification

Sends notifications via Slack Incoming Webhook.

### Configuration (SlackConfig)

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `webhook_url` | `str` | `""` | Slack Incoming Webhook URL |
| `channel` | `str \| None` | `None` | Channel override |
| `username` | `str` | `"Truthound"` | Bot display name |
| `icon_emoji` | `str` | `":mag:"` | Bot icon emoji |
| `include_details` | `bool` | `True` | Include detailed statistics |
| `mention_on_failure` | `list[str]` | `[]` | User IDs to mention on failure |
| `custom_message` | `str \| None` | `None` | Custom message template |
| `notify_on` | `str` | `"failure"` | Execution condition |

### Usage Examples

```python
from truthound.checkpoint.actions import SlackNotification

# Basic usage
action = SlackNotification(
    webhook_url="https://hooks.slack.com/services/T00/B00/XXX",
    notify_on="failure",
)

# Channel and mention configuration
action = SlackNotification(
    webhook_url="https://hooks.slack.com/services/T00/B00/XXX",
    channel="#data-quality",
    mention_on_failure=["U12345678", "@here"],  # User ID or @here/@channel
    include_details=True,
    notify_on="failure_or_error",
)

# Custom message
action = SlackNotification(
    webhook_url="...",
    custom_message="Checkpoint '{checkpoint}' {status}: {total_issues} issues found",
)
```

### Message Format

Messages are sent in Block Kit format:

- Status emoji (`:white_check_mark:`, `:x:`, `:warning:`)
- Checkpoint name and status
- Data asset, run ID, issue count, pass rate
- Issue count by severity
- Execution time

---

## EmailNotification

Sends email notifications. Supports SMTP, SendGrid, and AWS SES.

### Configuration (EmailConfig)

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `smtp_host` | `str` | `"localhost"` | SMTP server host |
| `smtp_port` | `int` | `587` | SMTP server port |
| `smtp_user` | `str \| None` | `None` | SMTP authentication user |
| `smtp_password` | `str \| None` | `None` | SMTP authentication password |
| `use_tls` | `bool` | `True` | Use TLS |
| `use_ssl` | `bool` | `False` | Use SSL |
| `from_address` | `str` | `""` | Sender address |
| `to_addresses` | `list[str]` | `[]` | Recipient addresses |
| `cc_addresses` | `list[str]` | `[]` | CC addresses |
| `subject_template` | `str` | `"[Truthound] {status} - {checkpoint}"` | Subject template |
| `include_html` | `bool` | `True` | Include HTML body |
| `provider` | `str` | `"smtp"` | Provider: `smtp`, `sendgrid`, `ses` |
| `api_key` | `str \| None` | `None` | API key (for SendGrid, etc.) |
| `notify_on` | `str` | `"failure"` | Execution condition |

### SMTP Usage

```python
from truthound.checkpoint.actions import EmailNotification

action = EmailNotification(
    smtp_host="smtp.gmail.com",
    smtp_port=587,
    smtp_user="alerts@example.com",
    smtp_password="${SMTP_PASSWORD}",  # Environment variable reference
    use_tls=True,
    from_address="alerts@example.com",
    to_addresses=["team@example.com", "lead@example.com"],
    cc_addresses=["manager@example.com"],
    notify_on="failure",
)
```

### SendGrid Usage

```python
action = EmailNotification(
    provider="sendgrid",
    api_key="${SENDGRID_API_KEY}",
    from_address="alerts@example.com",
    to_addresses=["team@example.com"],
    notify_on="failure",
)
```

Requirements: None (uses standard library)

### AWS SES Usage

```python
action = EmailNotification(
    provider="ses",
    from_address="alerts@example.com",  # Email verified in SES
    to_addresses=["team@example.com"],
    notify_on="failure",
)

# AWS credentials via environment variables or IAM role
```

Requirements: `pip install boto3`

---

## TeamsNotification

Sends Adaptive Card format notifications via Microsoft Teams Incoming Webhook.

### Configuration

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `webhook_url` | `str` | `""` | Teams Webhook URL |
| `channel` | `str \| None` | `None` | Channel name (for display) |
| `include_details` | `bool` | `True` | Include detailed information |
| `theme` | `MessageTheme` | `AUTO` | Message theme |
| `card_builder` | `AdaptiveCardBuilder \| None` | `None` | Custom card builder |
| `notify_on` | `str` | `"failure"` | Execution condition |

### Usage Examples

```python
from truthound.checkpoint.actions import TeamsNotification

# Basic usage
action = TeamsNotification(
    webhook_url="https://outlook.office.com/webhook/...",
    notify_on="failure",
)

# Theme and detail configuration
from truthound.checkpoint.actions.teams_notify import MessageTheme

action = TeamsNotification(
    webhook_url="...",
    channel="Data Quality",
    include_details=True,
    theme=MessageTheme.CRITICAL,
    notify_on="failure_or_error",
)
```

### Adaptive Card Customization

```python
from truthound.checkpoint.actions.teams_notify import AdaptiveCardBuilder

builder = AdaptiveCardBuilder()
builder.add_header("Data Quality Alert")
builder.add_fact("Dataset", "users.csv")
builder.add_fact("Issues", "150")
builder.add_action_button("View Report", "https://...")

action = TeamsNotification(
    webhook_url="...",
    card_builder=builder,
)
```

---

## DiscordNotification

Sends Embed format notifications via Discord Webhook.

### Configuration

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `webhook_url` | `str` | `""` | Discord Webhook URL |
| `username` | `str` | `"Truthound Bot"` | Bot display name |
| `avatar_url` | `str \| None` | `None` | Bot avatar URL |
| `embed_color` | `int` | Auto | Embed color (hex integer) |
| `embed_title` | `str \| None` | `None` | Embed title |
| `embed_description` | `str \| None` | `None` | Embed description |
| `embed_fields` | `list[dict]` | `[]` | Custom fields |
| `include_mentions` | `list[str]` | `[]` | Mention list (`@here`, role IDs, etc.) |
| `notify_on` | `str` | `"failure"` | Execution condition |

### Usage Examples

```python
from truthound.checkpoint.actions import DiscordNotification

# Basic usage
action = DiscordNotification(
    webhook_url="https://discord.com/api/webhooks/...",
    notify_on="failure",
)

# Custom configuration
action = DiscordNotification(
    webhook_url="...",
    username="Data Quality Bot",
    avatar_url="https://example.com/logo.png",
    embed_color=0xFF0000,  # Red
    include_mentions=["@here"],
    notify_on="failure_or_error",
)

# Custom Embed
action = DiscordNotification(
    webhook_url="...",
    embed_title="Data Quality Alert",
    embed_description="Validation failed for users.csv",
    embed_fields=[
        {"name": "Issues", "value": "150", "inline": True},
        {"name": "Severity", "value": "High", "inline": True},
    ],
)
```

---

## TelegramNotification

Sends notifications via Telegram Bot API.

### Configuration

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `bot_token` | `str` | `""` | Telegram Bot Token |
| `chat_id` | `str` | `""` | Channel/Group ID |
| `parse_mode` | `str` | `"Markdown"` | Parse mode: `Markdown`, `HTML` |
| `message_template` | `str \| None` | `None` | Custom message template |
| `disable_notification` | `bool` | `False` | Silent notification |
| `notify_on` | `str` | `"failure"` | Execution condition |

### Usage Examples

```python
from truthound.checkpoint.actions import TelegramNotification

# Basic usage
action = TelegramNotification(
    bot_token="${TELEGRAM_BOT_TOKEN}",
    chat_id="-1001234567890",  # Channel/Group ID
    notify_on="failure",
)

# Custom message
action = TelegramNotification(
    bot_token="...",
    chat_id="...",
    parse_mode="Markdown",
    message_template="""
*Data Quality Alert*

Dataset: `{checkpoint_name}`
Status: {status}
Issues: {issue_count}

[View Report]({report_url})
""",
)
```

### Photo Attachment

```python
from truthound.checkpoint.actions.telegram_notify import TelegramNotificationWithPhoto

action = TelegramNotificationWithPhoto(
    bot_token="...",
    chat_id="...",
    photo_url="https://example.com/chart.png",
    caption="Data quality trend chart",
)
```

---

## YAML Configuration Examples

```yaml
actions:
  # Slack notification
  - type: slack
    webhook_url: ${SLACK_WEBHOOK_URL}
    channel: "#data-quality"
    mention_on_failure:
      - "U12345678"
    notify_on: failure

  # Email notification
  - type: email
    smtp_host: smtp.gmail.com
    smtp_port: 587
    smtp_user: ${SMTP_USER}
    smtp_password: ${SMTP_PASSWORD}
    use_tls: true
    from_address: alerts@example.com
    to_addresses:
      - team@example.com
      - lead@example.com
    notify_on: failure

  # Teams notification
  - type: teams
    webhook_url: ${TEAMS_WEBHOOK_URL}
    include_details: true
    notify_on: failure_or_error

  # Discord notification
  - type: discord
    webhook_url: ${DISCORD_WEBHOOK_URL}
    include_mentions:
      - "@here"
    notify_on: failure

  # Telegram notification
  - type: telegram
    bot_token: ${TELEGRAM_BOT_TOKEN}
    chat_id: "-1001234567890"
    notify_on: failure
```
