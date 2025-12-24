"""Mock implementations for reporter dependencies.

These mocks simulate Jinja2 behavior for testing HTMLReporter
without requiring the actual jinja2 package.
"""

from __future__ import annotations

from typing import Any


class MockJinja2Template:
    """Mock Jinja2 Template."""

    def __init__(self, source: str):
        self._source = source

    def render(self, **context: Any) -> str:
        """Render the template with context.

        This is a simplified renderer that handles basic variable substitution.
        For testing purposes, it returns a predictable HTML structure.
        """
        # For testing, return a simplified HTML that includes key context values
        result = context.get("result")
        statistics = context.get("statistics")
        title = context.get("title", "Validation Report")

        html_parts = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            f"<title>{title}</title>",
            "<style>font-family: sans-serif;</style>",
            "</head>",
            "<body>",
            f"<h1>{title}</h1>",
        ]

        if result:
            html_parts.append(f'<p class="data-asset">{result.data_asset}</p>')
            html_parts.append(f'<p class="run-id">{result.run_id}</p>')

            status_class = "success" if result.success else "failure"
            status_text = "Passed" if result.success else "Failed"
            html_parts.append(f'<span class="status-badge {status_class}">{status_text}</span>')

        if statistics:
            html_parts.append('<div class="stats">')
            html_parts.append(f'<p>Total Rows: {statistics.total_rows:,}</p>')
            html_parts.append(f'<p>Total Issues: {statistics.total_issues}</p>')
            html_parts.append("</div>")

        issues = context.get("issues", [])
        if issues:
            html_parts.append("<table>")
            html_parts.append("<tr><th>Column</th><th>Issue Type</th><th>Severity</th></tr>")
            for issue in issues:
                html_parts.append(
                    f"<tr>"
                    f"<td>{issue.column or '-'}</td>"
                    f"<td>{issue.issue_type or issue.validator_name}</td>"
                    f"<td>{issue.severity or 'low'}</td>"
                    f"</tr>"
                )
            html_parts.append("</table>")

        generated_at = context.get("generated_at", "")
        html_parts.append(f'<footer>Generated at {generated_at}</footer>')
        html_parts.append("</body>")
        html_parts.append("</html>")

        return "\n".join(html_parts)


class MockJinja2Environment:
    """Mock Jinja2 Environment."""

    def __init__(self, template_source: str | None = None):
        self._template_source = template_source or ""
        self._templates: dict[str, MockJinja2Template] = {}

    def get_template(self, name: str) -> MockJinja2Template:
        """Get a template by name."""
        if name not in self._templates:
            self._templates[name] = MockJinja2Template(self._template_source)
        return self._templates[name]

    def add_template(self, name: str, source: str) -> None:
        """Add a template with source."""
        self._templates[name] = MockJinja2Template(source)


class MockJinja2BaseLoader:
    """Mock Jinja2 BaseLoader for custom loaders."""

    def get_source(
        self,
        environment: Any,
        template: str,
    ) -> tuple[str, str | None, Any]:
        """Get template source."""
        raise NotImplementedError


class MockJinja2FileSystemLoader(MockJinja2BaseLoader):
    """Mock FileSystemLoader."""

    def __init__(self, searchpath: str):
        self.searchpath = searchpath

    def get_source(
        self,
        environment: Any,
        template: str,
    ) -> tuple[str, str | None, Any]:
        """Get template source from file (mock)."""
        # For testing, return empty template
        return "", None, lambda: False


def create_mock_jinja2_env() -> MockJinja2Environment:
    """Create a mock Jinja2 environment."""
    return MockJinja2Environment()
