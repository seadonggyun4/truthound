"""Example plugins demonstrating the plugin architecture.

This module contains example plugins that showcase different plugin types:
- CustomValidatorPlugin: Example validator plugin with custom validators
- SlackNotifierPlugin: Example hook plugin for Slack notifications
- XMLReporterPlugin: Example reporter plugin for XML output

These plugins can be used as templates for creating custom plugins.
"""

from truthound.plugins.examples.custom_validators import CustomValidatorPlugin
from truthound.plugins.examples.slack_notifier import SlackNotifierPlugin
from truthound.plugins.examples.xml_reporter import XMLReporterPlugin

__all__ = [
    "CustomValidatorPlugin",
    "SlackNotifierPlugin",
    "XMLReporterPlugin",
]
