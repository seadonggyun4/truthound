# Notification Actions

알림 발송을 위한 액션입니다. Slack, Email, Teams, Discord, Telegram을 지원합니다.

## SlackNotification

Slack Incoming Webhook을 통해 알림을 발송합니다.

### 설정 (SlackConfig)

| 속성 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `webhook_url` | `str` | `""` | Slack Incoming Webhook URL |
| `channel` | `str \| None` | `None` | 채널 오버라이드 |
| `username` | `str` | `"Truthound"` | 봇 표시 이름 |
| `icon_emoji` | `str` | `":mag:"` | 봇 아이콘 이모지 |
| `include_details` | `bool` | `True` | 상세 통계 포함 |
| `mention_on_failure` | `list[str]` | `[]` | 실패 시 멘션할 사용자 ID |
| `custom_message` | `str \| None` | `None` | 커스텀 메시지 템플릿 |
| `notify_on` | `str` | `"failure"` | 실행 조건 |

### 사용 예시

```python
from truthound.checkpoint.actions import SlackNotification

# 기본 사용
action = SlackNotification(
    webhook_url="https://hooks.slack.com/services/T00/B00/XXX",
    notify_on="failure",
)

# 채널 및 멘션 설정
action = SlackNotification(
    webhook_url="https://hooks.slack.com/services/T00/B00/XXX",
    channel="#data-quality",
    mention_on_failure=["U12345678", "@here"],  # 사용자 ID 또는 @here/@channel
    include_details=True,
    notify_on="failure_or_error",
)

# 커스텀 메시지
action = SlackNotification(
    webhook_url="...",
    custom_message="Checkpoint '{checkpoint}' {status}: {total_issues} issues found",
)
```

### 메시지 형식

Block Kit 형식의 메시지가 발송됩니다:

- 상태 이모지 (`:white_check_mark:`, `:x:`, `:warning:`)
- 체크포인트 이름 및 상태
- 데이터 자산, 실행 ID, 이슈 수, 통과율
- Severity별 이슈 카운트
- 실행 시간

---

## EmailNotification

이메일 알림을 발송합니다. SMTP, SendGrid, AWS SES를 지원합니다.

### 설정 (EmailConfig)

| 속성 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `smtp_host` | `str` | `"localhost"` | SMTP 서버 호스트 |
| `smtp_port` | `int` | `587` | SMTP 서버 포트 |
| `smtp_user` | `str \| None` | `None` | SMTP 인증 사용자 |
| `smtp_password` | `str \| None` | `None` | SMTP 인증 비밀번호 |
| `use_tls` | `bool` | `True` | TLS 사용 |
| `use_ssl` | `bool` | `False` | SSL 사용 |
| `from_address` | `str` | `""` | 발신자 주소 |
| `to_addresses` | `list[str]` | `[]` | 수신자 주소 목록 |
| `cc_addresses` | `list[str]` | `[]` | 참조 주소 목록 |
| `subject_template` | `str` | `"[Truthound] {status} - {checkpoint}"` | 제목 템플릿 |
| `include_html` | `bool` | `True` | HTML 본문 포함 |
| `provider` | `str` | `"smtp"` | 제공자: `smtp`, `sendgrid`, `ses` |
| `api_key` | `str \| None` | `None` | API 키 (SendGrid 등) |
| `notify_on` | `str` | `"failure"` | 실행 조건 |

### SMTP 사용

```python
from truthound.checkpoint.actions import EmailNotification

action = EmailNotification(
    smtp_host="smtp.gmail.com",
    smtp_port=587,
    smtp_user="alerts@example.com",
    smtp_password="${SMTP_PASSWORD}",  # 환경 변수 참조
    use_tls=True,
    from_address="alerts@example.com",
    to_addresses=["team@example.com", "lead@example.com"],
    cc_addresses=["manager@example.com"],
    notify_on="failure",
)
```

### SendGrid 사용

```python
action = EmailNotification(
    provider="sendgrid",
    api_key="${SENDGRID_API_KEY}",
    from_address="alerts@example.com",
    to_addresses=["team@example.com"],
    notify_on="failure",
)
```

요구 사항: 없음 (표준 라이브러리 사용)

### AWS SES 사용

```python
action = EmailNotification(
    provider="ses",
    from_address="alerts@example.com",  # SES에서 검증된 이메일
    to_addresses=["team@example.com"],
    notify_on="failure",
)

# AWS 자격 증명은 환경 변수 또는 IAM 역할 사용
```

요구 사항: `pip install boto3`

---

## TeamsNotification

Microsoft Teams Incoming Webhook을 통해 Adaptive Card 형식의 알림을 발송합니다.

### 설정

| 속성 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `webhook_url` | `str` | `""` | Teams Webhook URL |
| `channel` | `str \| None` | `None` | 채널 이름 (표시용) |
| `include_details` | `bool` | `True` | 상세 정보 포함 |
| `theme` | `MessageTheme` | `AUTO` | 메시지 테마 |
| `card_builder` | `AdaptiveCardBuilder \| None` | `None` | 커스텀 카드 빌더 |
| `notify_on` | `str` | `"failure"` | 실행 조건 |

### 사용 예시

```python
from truthound.checkpoint.actions import TeamsNotification

# 기본 사용
action = TeamsNotification(
    webhook_url="https://outlook.office.com/webhook/...",
    notify_on="failure",
)

# 테마 및 상세 설정
from truthound.checkpoint.actions.teams_notify import MessageTheme

action = TeamsNotification(
    webhook_url="...",
    channel="Data Quality",
    include_details=True,
    theme=MessageTheme.CRITICAL,
    notify_on="failure_or_error",
)
```

### Adaptive Card 커스터마이징

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

Discord Webhook을 통해 Embed 형식의 알림을 발송합니다.

### 설정

| 속성 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `webhook_url` | `str` | `""` | Discord Webhook URL |
| `username` | `str` | `"Truthound Bot"` | 봇 표시 이름 |
| `avatar_url` | `str \| None` | `None` | 봇 아바타 URL |
| `embed_color` | `int` | 자동 | Embed 색상 (hex 정수) |
| `embed_title` | `str \| None` | `None` | Embed 제목 |
| `embed_description` | `str \| None` | `None` | Embed 설명 |
| `embed_fields` | `list[dict]` | `[]` | 커스텀 필드 |
| `include_mentions` | `list[str]` | `[]` | 멘션 목록 (`@here`, 역할 ID 등) |
| `notify_on` | `str` | `"failure"` | 실행 조건 |

### 사용 예시

```python
from truthound.checkpoint.actions import DiscordNotification

# 기본 사용
action = DiscordNotification(
    webhook_url="https://discord.com/api/webhooks/...",
    notify_on="failure",
)

# 커스텀 설정
action = DiscordNotification(
    webhook_url="...",
    username="Data Quality Bot",
    avatar_url="https://example.com/logo.png",
    embed_color=0xFF0000,  # 빨강
    include_mentions=["@here"],
    notify_on="failure_or_error",
)

# 커스텀 Embed
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

Telegram Bot API를 통해 알림을 발송합니다.

### 설정

| 속성 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `bot_token` | `str` | `""` | Telegram Bot Token |
| `chat_id` | `str` | `""` | 채널/그룹 ID |
| `parse_mode` | `str` | `"Markdown"` | 파싱 모드: `Markdown`, `HTML` |
| `message_template` | `str \| None` | `None` | 커스텀 메시지 템플릿 |
| `disable_notification` | `bool` | `False` | 무음 알림 |
| `notify_on` | `str` | `"failure"` | 실행 조건 |

### 사용 예시

```python
from truthound.checkpoint.actions import TelegramNotification

# 기본 사용
action = TelegramNotification(
    bot_token="${TELEGRAM_BOT_TOKEN}",
    chat_id="-1001234567890",  # 채널/그룹 ID
    notify_on="failure",
)

# 커스텀 메시지
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

### 사진 첨부

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

## YAML 설정 예시

```yaml
actions:
  # Slack 알림
  - type: slack
    webhook_url: ${SLACK_WEBHOOK_URL}
    channel: "#data-quality"
    mention_on_failure:
      - "U12345678"
    notify_on: failure

  # 이메일 알림
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

  # Teams 알림
  - type: teams
    webhook_url: ${TEAMS_WEBHOOK_URL}
    include_details: true
    notify_on: failure_or_error

  # Discord 알림
  - type: discord
    webhook_url: ${DISCORD_WEBHOOK_URL}
    include_mentions:
      - "@here"
    notify_on: failure

  # Telegram 알림
  - type: telegram
    bot_token: ${TELEGRAM_BOT_TOKEN}
    chat_id: "-1001234567890"
    notify_on: failure
```
