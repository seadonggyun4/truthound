"""Email notification action.

This action sends email notifications when checkpoint validations complete.
Supports SMTP, SendGrid, AWS SES, and other providers.
"""

from __future__ import annotations

import smtplib
from dataclasses import dataclass, field
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import TYPE_CHECKING, Any

from truthound.checkpoint.actions.base import (
    ActionConfig,
    ActionResult,
    ActionStatus,
    BaseAction,
    NotifyCondition,
)

if TYPE_CHECKING:
    from truthound.checkpoint.checkpoint import CheckpointResult


@dataclass
class EmailConfig(ActionConfig):
    """Configuration for email notification action.

    Attributes:
        smtp_host: SMTP server hostname.
        smtp_port: SMTP server port.
        smtp_user: SMTP authentication username.
        smtp_password: SMTP authentication password.
        use_tls: Use TLS encryption.
        use_ssl: Use SSL encryption.
        from_address: Sender email address.
        to_addresses: List of recipient email addresses.
        cc_addresses: List of CC email addresses.
        subject_template: Email subject template.
        include_html: Include HTML version of email.
        provider: Email provider ("smtp", "sendgrid", "ses").
        api_key: API key for provider (SendGrid, etc.).
    """

    smtp_host: str = "localhost"
    smtp_port: int = 587
    smtp_user: str | None = None
    smtp_password: str | None = None
    use_tls: bool = True
    use_ssl: bool = False
    from_address: str = ""
    to_addresses: list[str] = field(default_factory=list)
    cc_addresses: list[str] = field(default_factory=list)
    subject_template: str = "[Truthound] {status} - {checkpoint}"
    include_html: bool = True
    provider: str = "smtp"
    api_key: str | None = None
    notify_on: NotifyCondition | str = NotifyCondition.FAILURE


class EmailNotification(BaseAction[EmailConfig]):
    """Action to send email notifications.

    Sends formatted email notifications with validation results
    and statistics via SMTP or email providers.

    Example:
        >>> action = EmailNotification(
        ...     smtp_host="smtp.gmail.com",
        ...     smtp_port=587,
        ...     from_address="alerts@example.com",
        ...     to_addresses=["team@example.com"],
        ...     notify_on="failure",
        ... )
        >>> result = action.execute(checkpoint_result)
    """

    action_type = "email_notification"

    @classmethod
    def _default_config(cls) -> EmailConfig:
        return EmailConfig()

    def _execute(self, checkpoint_result: "CheckpointResult") -> ActionResult:
        """Send email notification."""
        config = self._config

        if not config.to_addresses:
            return ActionResult(
                action_name=self.name,
                action_type=self.action_type,
                status=ActionStatus.ERROR,
                message="No recipients configured",
                error="to_addresses is required",
            )

        if not config.from_address:
            return ActionResult(
                action_name=self.name,
                action_type=self.action_type,
                status=ActionStatus.ERROR,
                message="No sender configured",
                error="from_address is required",
            )

        # Build email
        subject = self._build_subject(checkpoint_result)
        text_body = self._build_text_body(checkpoint_result)
        html_body = self._build_html_body(checkpoint_result) if config.include_html else None

        # Send based on provider
        try:
            if config.provider == "smtp":
                self._send_smtp(subject, text_body, html_body)
            elif config.provider == "sendgrid":
                self._send_sendgrid(subject, text_body, html_body)
            elif config.provider == "ses":
                self._send_ses(subject, text_body, html_body)
            else:
                self._send_smtp(subject, text_body, html_body)

            return ActionResult(
                action_name=self.name,
                action_type=self.action_type,
                status=ActionStatus.SUCCESS,
                message=f"Email sent to {len(config.to_addresses)} recipient(s)",
                details={
                    "to": config.to_addresses,
                    "subject": subject,
                    "provider": config.provider,
                },
            )

        except Exception as e:
            return ActionResult(
                action_name=self.name,
                action_type=self.action_type,
                status=ActionStatus.ERROR,
                message="Failed to send email",
                error=str(e),
            )

    def _build_subject(self, checkpoint_result: "CheckpointResult") -> str:
        """Build email subject."""
        status = checkpoint_result.status.value.upper()
        return self._config.subject_template.format(
            status=status,
            checkpoint=checkpoint_result.checkpoint_name,
            run_id=checkpoint_result.run_id,
        )

    def _build_text_body(self, checkpoint_result: "CheckpointResult") -> str:
        """Build plain text email body."""
        validation = checkpoint_result.validation_result
        stats = validation.statistics if validation else None

        lines = [
            f"Checkpoint: {checkpoint_result.checkpoint_name}",
            f"Status: {checkpoint_result.status.value.upper()}",
            f"Run ID: {checkpoint_result.run_id}",
            f"Run Time: {checkpoint_result.run_time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Data Asset: {checkpoint_result.data_asset}",
            "",
            "Statistics:",
            f"  Total Issues: {stats.total_issues if stats else 0}",
            f"  Critical: {stats.critical_issues if stats else 0}",
            f"  High: {stats.high_issues if stats else 0}",
            f"  Medium: {stats.medium_issues if stats else 0}",
            f"  Low: {stats.low_issues if stats else 0}",
            f"  Pass Rate: {stats.pass_rate * 100 if stats else 100:.1f}%",
            "",
            f"Data Info:",
            f"  Rows: {stats.total_rows if stats else 'N/A':,}",
            f"  Columns: {stats.total_columns if stats else 'N/A'}",
            "",
            "---",
            "Sent by Truthound Data Quality Toolkit",
        ]
        return "\n".join(lines)

    def _build_html_body(self, checkpoint_result: "CheckpointResult") -> str:
        """Build HTML email body."""
        status = checkpoint_result.status.value
        validation = checkpoint_result.validation_result
        stats = validation.statistics if validation else None

        status_color = {
            "success": "#28a745",
            "failure": "#dc3545",
            "error": "#dc3545",
            "warning": "#ffc107",
        }.get(status, "#6c757d")

        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; color: #333; }}
        .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
        .header {{ text-align: center; padding: 20px; background: #f8f9fa; border-radius: 8px; }}
        .status {{ display: inline-block; padding: 8px 16px; border-radius: 20px; color: white; font-weight: bold; background: {status_color}; }}
        .section {{ margin: 20px 0; padding: 15px; background: white; border: 1px solid #e9ecef; border-radius: 8px; }}
        .section h3 {{ margin-top: 0; color: #495057; border-bottom: 1px solid #e9ecef; padding-bottom: 10px; }}
        .stats {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }}
        .stat {{ text-align: center; padding: 10px; background: #f8f9fa; border-radius: 5px; }}
        .stat-value {{ font-size: 1.5em; font-weight: bold; }}
        .stat-label {{ font-size: 0.85em; color: #666; }}
        .critical {{ color: #dc3545; }}
        .high {{ color: #e67e22; }}
        .medium {{ color: #ffc107; }}
        .low {{ color: #17a2b8; }}
        .footer {{ text-align: center; margin-top: 30px; padding-top: 20px; border-top: 1px solid #e9ecef; color: #666; font-size: 0.9em; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h2>Truthound Validation Report</h2>
            <div class="status">{status.upper()}</div>
        </div>

        <div class="section">
            <h3>Run Details</h3>
            <p><strong>Checkpoint:</strong> {checkpoint_result.checkpoint_name}</p>
            <p><strong>Run ID:</strong> {checkpoint_result.run_id}</p>
            <p><strong>Time:</strong> {checkpoint_result.run_time.strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>Data Asset:</strong> {checkpoint_result.data_asset}</p>
        </div>

        <div class="section">
            <h3>Validation Statistics</h3>
            <div class="stats">
                <div class="stat">
                    <div class="stat-value">{stats.total_issues if stats else 0}</div>
                    <div class="stat-label">Total Issues</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{stats.pass_rate * 100 if stats else 100:.1f}%</div>
                    <div class="stat-label">Pass Rate</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{stats.total_rows if stats else 0:,}</div>
                    <div class="stat-label">Rows</div>
                </div>
            </div>

            <div class="stats" style="margin-top: 10px;">
                <div class="stat">
                    <div class="stat-value critical">{stats.critical_issues if stats else 0}</div>
                    <div class="stat-label">Critical</div>
                </div>
                <div class="stat">
                    <div class="stat-value high">{stats.high_issues if stats else 0}</div>
                    <div class="stat-label">High</div>
                </div>
                <div class="stat">
                    <div class="stat-value medium">{stats.medium_issues if stats else 0}</div>
                    <div class="stat-label">Medium</div>
                </div>
            </div>
        </div>

        <div class="footer">
            <p>This email was sent by Truthound Data Quality Toolkit</p>
        </div>
    </div>
</body>
</html>"""
        return html

    def _send_smtp(
        self,
        subject: str,
        text_body: str,
        html_body: str | None,
    ) -> None:
        """Send email via SMTP."""
        config = self._config

        # Create message
        if html_body:
            msg = MIMEMultipart("alternative")
            msg.attach(MIMEText(text_body, "plain"))
            msg.attach(MIMEText(html_body, "html"))
        else:
            msg = MIMEMultipart()
            msg.attach(MIMEText(text_body, "plain"))

        msg["Subject"] = subject
        msg["From"] = config.from_address
        msg["To"] = ", ".join(config.to_addresses)
        if config.cc_addresses:
            msg["Cc"] = ", ".join(config.cc_addresses)

        all_recipients = config.to_addresses + config.cc_addresses

        # Connect and send
        if config.use_ssl:
            smtp = smtplib.SMTP_SSL(config.smtp_host, config.smtp_port)
        else:
            smtp = smtplib.SMTP(config.smtp_host, config.smtp_port)
            if config.use_tls:
                smtp.starttls()

        try:
            if config.smtp_user and config.smtp_password:
                smtp.login(config.smtp_user, config.smtp_password)
            smtp.sendmail(config.from_address, all_recipients, msg.as_string())
        finally:
            smtp.quit()

    def _send_sendgrid(
        self,
        subject: str,
        text_body: str,
        html_body: str | None,
    ) -> None:
        """Send email via SendGrid API."""
        import json
        import urllib.request

        config = self._config

        if not config.api_key:
            raise ValueError("SendGrid API key is required")

        payload = {
            "personalizations": [{"to": [{"email": addr} for addr in config.to_addresses]}],
            "from": {"email": config.from_address},
            "subject": subject,
            "content": [{"type": "text/plain", "value": text_body}],
        }

        if config.cc_addresses:
            payload["personalizations"][0]["cc"] = [{"email": addr} for addr in config.cc_addresses]

        if html_body:
            payload["content"].append({"type": "text/html", "value": html_body})

        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            "https://api.sendgrid.com/v3/mail/send",
            data=data,
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json",
            },
        )

        with urllib.request.urlopen(request, timeout=config.timeout_seconds) as response:
            response.read()

    def _send_ses(
        self,
        subject: str,
        text_body: str,
        html_body: str | None,
    ) -> None:
        """Send email via AWS SES."""
        try:
            import boto3
        except ImportError:
            raise ImportError("boto3 is required for AWS SES. Install with: pip install boto3")

        config = self._config
        ses = boto3.client("ses")

        body: dict[str, Any] = {
            "Text": {"Data": text_body, "Charset": "UTF-8"},
        }
        if html_body:
            body["Html"] = {"Data": html_body, "Charset": "UTF-8"}

        destination: dict[str, list[str]] = {
            "ToAddresses": config.to_addresses,
        }
        if config.cc_addresses:
            destination["CcAddresses"] = config.cc_addresses

        ses.send_email(
            Source=config.from_address,
            Destination=destination,
            Message={
                "Subject": {"Data": subject, "Charset": "UTF-8"},
                "Body": body,
            },
        )

    def validate_config(self) -> list[str]:
        """Validate configuration."""
        errors = []

        if not self._config.from_address:
            errors.append("from_address is required")

        if not self._config.to_addresses:
            errors.append("at least one to_address is required")

        if self._config.provider == "sendgrid" and not self._config.api_key:
            errors.append("api_key is required for SendGrid provider")

        return errors
