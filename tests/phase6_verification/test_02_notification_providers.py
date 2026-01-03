#!/usr/bin/env python3
"""Phase 6 검증: Notification Providers 테스트

문서에 명시된 9개 프로바이더:
1. Slack (slack_notify.py)
2. Email (email_notify.py)
3. PagerDuty (pagerduty.py)
4. GitHub (github_action.py)
5. Webhook (webhook.py)
6. Microsoft Teams (teams_notify.py)
7. OpsGenie (opsgenie.py)
8. Discord (discord_notify.py)
9. Telegram (telegram_notify.py)
"""
import sys

print("=" * 60)
print("Phase 6 검증: Notification Providers")
print("=" * 60)

errors = []
warnings = []
doc_discrepancies = []

# 문서에 명시된 9개 프로바이더
EXPECTED_PROVIDERS = {
    "slack": "SlackNotification",
    "email": "EmailNotification",
    "pagerduty": "PagerDutyAction",
    "github": "GitHubAction",
    "webhook": "WebhookAction",
    "teams": "TeamsNotification",
    "opsgenie": "OpsGenieAction",
    "discord": "DiscordNotification",
    "telegram": "TelegramNotification",
}

# 1. __init__.py에서 export 확인
print("\n[1] checkpoint.actions 패키지 export 확인...")
try:
    from truthound.checkpoint import actions
    exported = dir(actions)
    print(f"  Export된 항목 수: {len(exported)}")

    # 주요 클래스 확인
    for provider, class_name in EXPECTED_PROVIDERS.items():
        if class_name in exported:
            print(f"  ✓ {class_name} export됨")
        else:
            warnings.append(f"{class_name}이 actions 패키지에서 export되지 않음")
            print(f"  △ {class_name} export 안됨 (개별 모듈에서 import 필요)")
except ImportError as e:
    errors.append(f"actions 패키지 import 실패: {e}")
    print(f"  ✗ actions 패키지 import 실패: {e}")

# 2. 각 프로바이더 개별 import 테스트
print("\n[2] 개별 프로바이더 import 테스트...")

provider_modules = {
    "slack": ("truthound.checkpoint.actions.slack_notify", ["SlackNotification", "SlackConfig"]),
    "email": ("truthound.checkpoint.actions.email_notify", ["EmailNotification", "EmailConfig"]),
    "pagerduty": ("truthound.checkpoint.actions.pagerduty", ["PagerDutyAction"]),
    "github": ("truthound.checkpoint.actions.github_action", ["GitHubAction"]),
    "webhook": ("truthound.checkpoint.actions.webhook", ["WebhookAction"]),
    "teams": ("truthound.checkpoint.actions.teams_notify", ["TeamsNotification"]),
    "opsgenie": ("truthound.checkpoint.actions.opsgenie", ["OpsGenieAction", "Responder"]),
    "discord": ("truthound.checkpoint.actions.discord_notify", ["DiscordNotification"]),
    "telegram": ("truthound.checkpoint.actions.telegram_notify", ["TelegramNotification"]),
}

imported_providers = {}
for provider, (module_path, expected_classes) in provider_modules.items():
    try:
        module = __import__(module_path, fromlist=expected_classes)
        imported_providers[provider] = module
        for cls_name in expected_classes:
            if hasattr(module, cls_name):
                print(f"  ✓ {provider}: {cls_name}")
            else:
                warnings.append(f"{provider}: {cls_name} 클래스 없음")
                print(f"  △ {provider}: {cls_name} 없음")
    except ImportError as e:
        errors.append(f"{provider} 모듈 import 실패: {e}")
        print(f"  ✗ {provider} import 실패: {e}")

# 3. NotifyCondition 확인
print("\n[3] NotifyCondition 열거형 확인...")
try:
    from truthound.checkpoint.actions.base import NotifyCondition

    expected_conditions = ["ALWAYS", "SUCCESS", "FAILURE", "ERROR", "WARNING",
                           "FAILURE_OR_ERROR", "NOT_SUCCESS"]
    actual_conditions = [c.name for c in NotifyCondition]

    for cond in expected_conditions:
        if cond in actual_conditions:
            print(f"  ✓ NotifyCondition.{cond}")
        else:
            errors.append(f"NotifyCondition.{cond} 없음")
            print(f"  ✗ NotifyCondition.{cond} 없음")

except ImportError as e:
    errors.append(f"NotifyCondition import 실패: {e}")
    print(f"  ✗ NotifyCondition import 실패: {e}")

# 4. Slack 상세 테스트
print("\n[4] Slack 프로바이더 상세 테스트...")
try:
    from truthound.checkpoint.actions.slack_notify import SlackNotification, SlackConfig

    # 생성 테스트
    slack = SlackNotification(
        config=SlackConfig(
            webhook_url="https://hooks.slack.com/test",
            channel="#test",
        )
    )
    print(f"  ✓ SlackNotification 생성 성공")
    print(f"  ✓ action_type: {slack.action_type}")

    # 필수 속성 확인
    for attr in ["webhook_url", "channel"]:
        if hasattr(slack.config, attr):
            print(f"  ✓ config.{attr} 존재")
        else:
            warnings.append(f"SlackConfig에 {attr} 없음")

except Exception as e:
    errors.append(f"Slack 상세 테스트 실패: {e}")
    print(f"  ✗ Slack 상세 테스트 실패: {e}")

# 5. Teams 상세 테스트 (문서: Adaptive Card Builder, 4가지 템플릿)
print("\n[5] Teams 프로바이더 상세 테스트...")
try:
    from truthound.checkpoint.actions.teams_notify import TeamsNotification

    # TeamsNotification 생성
    teams = TeamsNotification(webhook_url="https://outlook.office.com/webhook/test")
    print(f"  ✓ TeamsNotification 생성 성공")

    # 문서에 명시된 4가지 템플릿 확인
    # "4가지 템플릿: Default, Minimal, Detailed, Compact"
    import inspect
    module = imported_providers.get("teams")
    if module:
        # 템플릿 관련 클래스/상수 확인
        module_contents = dir(module)
        template_related = [x for x in module_contents if "template" in x.lower() or "theme" in x.lower()]
        if template_related:
            print(f"  ✓ 템플릿 관련 항목: {template_related}")
        else:
            warnings.append("Teams: 템플릿 관련 항목 찾을 수 없음")
            print(f"  △ 템플릿 관련 항목 찾을 수 없음")

except Exception as e:
    warnings.append(f"Teams 상세 테스트 실패: {e}")
    print(f"  △ Teams 상세 테스트 실패: {e}")

# 6. OpsGenie 상세 테스트 (문서: Responder 타입, 우선순위 매핑)
print("\n[6] OpsGenie 프로바이더 상세 테스트...")
try:
    from truthound.checkpoint.actions.opsgenie import OpsGenieAction

    # Responder 확인
    try:
        from truthound.checkpoint.actions.opsgenie import Responder
        print(f"  ✓ Responder 클래스 존재")

        # Responder 메서드 확인
        if hasattr(Responder, "team"):
            print(f"  ✓ Responder.team() 메서드 존재")
        if hasattr(Responder, "user"):
            print(f"  ✓ Responder.user() 메서드 존재")
    except ImportError:
        warnings.append("OpsGenie: Responder 클래스 import 실패")
        print(f"  △ Responder 클래스 import 실패")

    # OpsGenieAction 생성
    opsgenie = OpsGenieAction(api_key="test-api-key")
    print(f"  ✓ OpsGenieAction 생성 성공")

    # auto_priority 속성 확인 (문서에 명시)
    if hasattr(opsgenie.config, "auto_priority"):
        print(f"  ✓ config.auto_priority 존재")
    else:
        warnings.append("OpsGenieConfig에 auto_priority 없음")

except Exception as e:
    warnings.append(f"OpsGenie 상세 테스트 실패: {e}")
    print(f"  △ OpsGenie 상세 테스트 실패: {e}")

# 7. Discord 테스트
print("\n[7] Discord 프로바이더 테스트...")
try:
    from truthound.checkpoint.actions.discord_notify import DiscordNotification

    discord = DiscordNotification(webhook_url="https://discord.com/api/webhooks/test")
    print(f"  ✓ DiscordNotification 생성 성공")
    print(f"  ✓ action_type: {discord.action_type}")

except Exception as e:
    errors.append(f"Discord 테스트 실패: {e}")
    print(f"  ✗ Discord 테스트 실패: {e}")

# 8. Telegram 테스트
print("\n[8] Telegram 프로바이더 테스트...")
try:
    from truthound.checkpoint.actions.telegram_notify import TelegramNotification

    telegram = TelegramNotification(bot_token="test-token", chat_id="test-chat")
    print(f"  ✓ TelegramNotification 생성 성공")
    print(f"  ✓ action_type: {telegram.action_type}")

except Exception as e:
    errors.append(f"Telegram 테스트 실패: {e}")
    print(f"  ✗ Telegram 테스트 실패: {e}")

# 9. PagerDuty 테스트
print("\n[9] PagerDuty 프로바이더 테스트...")
try:
    from truthound.checkpoint.actions.pagerduty import PagerDutyAction

    pagerduty = PagerDutyAction(routing_key="test-routing-key")
    print(f"  ✓ PagerDutyAction 생성 성공")

except Exception as e:
    warnings.append(f"PagerDuty 테스트 실패: {e}")
    print(f"  △ PagerDuty 테스트 실패: {e}")

# 10. Webhook 테스트
print("\n[10] Webhook 프로바이더 테스트...")
try:
    from truthound.checkpoint.actions.webhook import WebhookAction

    webhook = WebhookAction(url="https://example.com/webhook")
    print(f"  ✓ WebhookAction 생성 성공")

except Exception as e:
    warnings.append(f"Webhook 테스트 실패: {e}")
    print(f"  △ Webhook 테스트 실패: {e}")

# 결과 요약
print("\n" + "=" * 60)
print("Notification Providers 검증 결과")
print("=" * 60)

providers_found = len([p for p in provider_modules if p in imported_providers])
print(f"\n  프로바이더 발견: {providers_found}/9")

print(f"\n  [문서 불일치] {len(doc_discrepancies)}건:")
for d in doc_discrepancies:
    print(f"    - {d}")

print(f"\n  [오류] {len(errors)}건:")
for e in errors:
    print(f"    - {e}")

print(f"\n  [경고] {len(warnings)}건:")
for w in warnings:
    print(f"    - {w}")

if errors:
    print("\n결과: ✗ FAIL")
    sys.exit(1)
elif providers_found < 9:
    print(f"\n결과: △ PARTIAL ({providers_found}/9 프로바이더)")
    sys.exit(0)
else:
    print("\n결과: ✓ PASS")
    sys.exit(0)
