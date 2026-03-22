---
title: Notifications
---

# Notifications

The notification module routes data quality and SLA events to operator-facing channels.

## Main Components

- `NotificationRegistry`
- handlers such as `SlackNotificationHandler`, `WebhookNotificationHandler`, and
  `EmailNotificationHandler`
- formatters for text, markdown, JSON, and Slack blocks
- hook support for logging and metrics

## When To Use It

Use notifications when:

- quality failures must reach humans or incident systems quickly
- different severities should route to different channels
- the host platform's native alerting needs a normalized Truthound event payload

## Recommended Pattern

- register handlers centrally
- keep severity routing explicit
- separate informational alerts from blocking incidents
- test notification formatting and transport in CI, not only in production

## Channel Strategy

Typical routing looks like:

- Slack or webhook for routine warnings
- email for broader operator visibility
- incident tooling for critical SLA or data quality failures

## Related Pages

- [Secrets](secrets.md)
- [CI/CD and Production Rollout](ci-cd-production.md)
