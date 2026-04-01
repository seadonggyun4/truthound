# Privacy and Redaction

Truthound AI defaults to strict summary-only redaction.

## Allowed Outbound Content

- schema summary
- column names and coarse types
- aggregate counts and ratios
- validator/check names
- severity and issue counts
- artifact references
- history window summary

## Forbidden Outbound Content

- raw rows
- raw sample values
- free-form row excerpts
- unexpected row payloads
- docs HTML bodies
- PII-like literal strings

## Enforcement

The redaction model is enforced in three places:

1. context-bundle construction
2. provider payload inspection
3. artifact validation

That means unsafe content can be rejected before it reaches a provider and can
also be rejected before it is stored locally.

## Why Summary-Only

Summary-only redaction trades breadth for auditability. It makes provider
contracts smaller, more stable, and easier to review, which is especially
important when dashboard review/apply workflows are layered on top later.
