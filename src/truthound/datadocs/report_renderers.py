"""HTML renderers for Data Docs research report objects."""

from __future__ import annotations

import html
import platform
import sys
from typing import Any

from truthound.datadocs.base import ReportSpec, SectionSpec, SectionType
from truthound.datadocs.report_document import ChapterGroup, ResearchReportDocument
from truthound.datadocs.type_labels import profile_type_label


class ReportDocumentRenderer:
    """Render research report model objects to HTML."""

    def __init__(self, document: ResearchReportDocument) -> None:
        self.document = document

    def render_chapters(
        self,
        chapter_groups: list[ChapterGroup],
        rendered_sections: dict[SectionType, str],
    ) -> str:
        """Render chapter wrappers around already-rendered section HTML."""
        chapters_html = []
        for chapter in chapter_groups:
            body = "".join(rendered_sections[section.section_type] for section in chapter.sections)
            lead_html = (
                f'<p class="report-paragraph chapter-lead">{html.escape(chapter.lead)}</p>'
                if chapter.lead
                else ""
            )
            chapters_html.append(
                f'''
                <section class="report-chapter" id="chapter-{chapter.number}">
                    <header class="chapter-header">
                        <h2 class="chapter-title">{html.escape(chapter.title)}</h2>
                    </header>
                    {lead_html}
                    {body}
                </section>
                '''
            )
        return "".join(chapters_html)

    def render_toc(self, chapter_groups: list[ChapterGroup], *, for_pdf: bool) -> str:
        """Render the shared table of contents for HTML or PDF output."""
        toc_entries = self.document.toc_entries(chapter_groups)
        if for_pdf:
            toc_items = []
            for idx, entry in enumerate(toc_entries, 1):
                target_id = html.escape(entry.target_id, quote=True)
                title = html.escape(entry.title)
                toc_items.append(
                    f'''<li class="toc-row-professional">
                        <a class="toc-link-professional" href="#{target_id}">
                            <span class="toc-number">{idx}.</span>
                            <span class="toc-entry">{title}</span>
                            <span class="toc-leader" aria-hidden="true"></span>
                            <span class="toc-page" data-target="#{target_id}" aria-label="page number"></span>
                        </a>
                    </li>'''
                )
            return f'''
                <nav class="report-toc-professional" aria-labelledby="report-toc-title">
                    <h2 id="report-toc-title" class="toc-title-professional">{html.escape(self.document.label("report.toc", "Table of Contents"))}</h2>
                    <ol class="toc-list-professional">{"".join(toc_items)}</ol>
                </nav>
            '''

        toc_items = [
            f'<li class="toc-item"><a href="#{html.escape(entry.target_id, quote=True)}">{html.escape(entry.title)}</a></li>'
            for entry in toc_entries
        ]
        return f'''
            <nav class="report-toc">
                <h3 class="toc-title">{html.escape(self.document.label("report.contents", "Contents"))}</h3>
                <ul class="toc-list">{"".join(toc_items)}</ul>
            </nav>
        '''

    def render_executive_summary(self, spec: ReportSpec) -> str:
        """Render the executive summary section."""
        item_html = "".join(
            f'''
            <div class="executive-summary-item">
                <h3>{html.escape(item.label)}</h3>
                <p>{html.escape(item.body)}</p>
            </div>
            '''
            for item in self.document.summary_items(spec)
        )
        return f'''
<section class="executive-summary" id="executive-summary">
    <div class="section-header">
        <h2 class="section-title">{html.escape(self.document.label("report.executive_summary", "Executive Summary"))}</h2>
        <p class="section-subtitle">{html.escape(self.document.label("summary.subtitle", "Purpose, findings, risks, actions, and limitations"))}</p>
    </div>
    <div class="executive-summary-grid">
        {item_html}
    </div>
</section>
'''

    def render_quality_framework(self) -> str:
        """Render the report's quality framework interpretation table."""
        rows_html = "".join(
            "<tr>"
            f"<td>{html.escape(row.dimension)}</td>"
            f"<td>{html.escape(row.evidence)}</td>"
            f"<td>{html.escape(row.status)}</td>"
            f"<td>{html.escape(row.note)}</td>"
            "</tr>"
            for row in self.document.quality_rows()
        )
        return f'''
<section class="quality-framework" id="quality-framework">
    <div class="section-header">
        <h2 class="section-title">{html.escape(self.document.label("report.quality_framework", "Quality Framework Mapping"))}</h2>
        <p class="section-subtitle">{html.escape(self.document.label("quality.framework.subtitle", "Interpretation layer based on recognized data quality dimensions"))}</p>
    </div>
    <p class="report-paragraph">{html.escape(self.document.label("quality.framework.note", "The mapping below explains how available Truthound profile signals relate to common data quality dimensions. It does not change metric calculations."))}</p>
    <div class="table-container">
        <h4 class="table-title">{html.escape(self.document.captions.table(2, self.document.label("quality.framework.table", "Quality dimension mapping")))}</h4>
        <table class="data-table report-object-table">
            <thead><tr><th>{html.escape(self.document.label("quality.dimension", "Dimension"))}</th><th>{html.escape(self.document.label("quality.metric", "Evidence"))}</th><th>{html.escape(self.document.label("quality.measurement_status", "Status"))}</th><th>{html.escape(self.document.label("quality.note", "Note"))}</th></tr></thead>
            <tbody>{rows_html}</tbody>
        </table>
    </div>
    <p class="report-caption">{html.escape(self.document.label("quality.framework.caption", "Reference basis: Eurostat ESS/QAF, ISO 8000, Wang and Strong, Batini and Scannapieco, and DAMA-DMBOK quality dimensions."))}</p>
</section>
'''

    def render_appendices(self, spec: ReportSpec) -> str:
        """Render auditability appendices without embedding raw source data."""
        columns = spec.profile_data.get("columns", [])
        source = spec.metadata.data_source or self.document.label("common.unknown", "Unknown")
        definition_rows = "".join(
            f"<tr><td>{html.escape(self.document.label(metric_key, metric_key))}</td><td>{html.escape(self.document.label(formula_key, default_formula))}</td></tr>"
            for metric_key, formula_key, default_formula in self._metric_definitions()
        )
        env_rows_html = "".join(
            f"<tr><td>{html.escape(k)}</td><td>{html.escape(str(v))}</td></tr>"
            for k, v in self._environment_rows(spec, source)
        )
        profile_rows = "".join(self._render_profile_row(column) for column in columns)
        coverage_rows = "".join(
            "<tr>"
            f"<td>{html.escape(dimension)}</td>"
            f"<td>{html.escape(status)}</td>"
            f"<td>{html.escape(limitation)}</td>"
            "</tr>"
            for dimension, status, limitation in self.document.quality_coverage_rows()
        )
        methodology_rows = "".join(
            "<tr>"
            f"<td>{html.escape(row.criterion)}</td>"
            f"<td>{html.escape(row.threshold)}</td>"
            f"<td>{html.escape(row.interpretation)}</td>"
            "</tr>"
            for row in self.document.methodology_threshold_rows()
        )
        return f'''
<section class="report-appendices" id="appendices">
    <section class="report-appendix" id="appendix-a">
        <h2 class="appendix-title">{html.escape(self.document.captions.appendix("A", "appendix.metrics", "Metric Definitions and Formulae"))}</h2>
        <p class="report-paragraph appendix-lead">{html.escape(self.document.label("appendix.metrics.lead", "This appendix defines the metrics used in the body of the report. It documents interpretation boundaries and does not redefine Truthound profile calculations."))}</p>
        <div class="table-container">
            <h4 class="table-title">{html.escape(self.document.captions.table(3, self.document.label("appendix.metrics.table", "Metric definitions")))}</h4>
            <table class="data-table report-object-table"><thead><tr><th>{html.escape(self.document.label("appendix.metric", "Metric"))}</th><th>{html.escape(self.document.label("appendix.definition", "Definition"))}</th></tr></thead><tbody>{definition_rows}</tbody></table>
        </div>
    </section>
    <section class="report-appendix" id="appendix-b">
        <h2 class="appendix-title">{html.escape(self.document.captions.appendix("B", "appendix.reproducibility", "Execution Environment and Reproducibility"))}</h2>
        <p class="report-paragraph appendix-lead">{html.escape(self.document.label("appendix.reproducibility.lead", "This appendix records execution metadata needed to reproduce the report artifact without exposing raw input data."))}</p>
        <div class="auditability-block">
            <table class="data-table report-object-table"><tbody>{env_rows_html}</tbody></table>
        </div>
    </section>
    <section class="report-appendix" id="appendix-c">
        <h2 class="appendix-title">{html.escape(self.document.captions.appendix("C", "appendix.full_profile", "Full Column Profile"))}</h2>
        <p class="report-paragraph appendix-lead">{html.escape(self.document.label("appendix.full_profile.lead", "This appendix provides the complete column profile used by the body analysis while preserving source column names and technical type identifiers."))}</p>
        <div class="table-container">
            <h4 class="table-title">{html.escape(self.document.captions.table(4, self.document.label("appendix.full_profile.table", "Full column profile")))}</h4>
            <table class="data-table report-object-table"><thead><tr><th>{html.escape(self.document.label("table.column", "Column"))}</th><th>{html.escape(self.document.label("table.type", "Type"))}</th><th>{html.escape(self.document.label("table.null_percent", "Null %"))}</th><th>{html.escape(self.document.label("table.unique_percent", "Unique %"))}</th><th>{html.escape(self.document.label("table.distinct", "Distinct"))}</th></tr></thead><tbody>{profile_rows}</tbody></table>
        </div>
    </section>
    <section class="report-appendix" id="appendix-d">
        <h2 class="appendix-title">{html.escape(self.document.captions.appendix("D", "appendix.quality_coverage", "Quality Dimension Coverage and Limitations"))}</h2>
        <p class="report-paragraph appendix-lead">{html.escape(self.document.label("appendix.quality_coverage.lead", "This appendix supports audit review by separating measured profile evidence from quality dimensions that require business rules, reference data, or freshness metadata."))}</p>
        <p class="report-paragraph">{html.escape(self.document.label("appendix.quality_coverage.note", "This appendix separates profiled evidence from dimensions that require business rules, reference data, or source freshness metadata."))}</p>
        <div class="table-container">
            <h4 class="table-title">{html.escape(self.document.captions.table(5, self.document.label("appendix.quality_coverage.table", "Quality dimension coverage")))}</h4>
            <table class="data-table report-object-table quality-coverage-table"><thead><tr><th>{html.escape(self.document.label("quality.dimension", "Dimension"))}</th><th>{html.escape(self.document.label("quality.measurement_status", "Status"))}</th><th>{html.escape(self.document.label("appendix.limitation", "Limitation"))}</th></tr></thead><tbody>{coverage_rows}</tbody></table>
        </div>
    </section>
    <section class="report-appendix" id="appendix-e">
        <h2 class="appendix-title">{html.escape(self.document.captions.appendix("E", "appendix.methodology", "Diagnostic Criteria and Thresholds"))}</h2>
        <p class="report-paragraph appendix-lead">{html.escape(self.document.label("appendix.methodology.lead", "This appendix records the threshold policy used by generated alerts and recommendations so that reviewers can reproduce the report interpretation."))}</p>
        <div class="table-container">
            <h4 class="table-title">{html.escape(self.document.captions.table(6, self.document.label("appendix.methodology.table", "Diagnostic criteria and thresholds")))}</h4>
            <table class="data-table report-object-table methodology-table"><thead><tr><th>{html.escape(self.document.label("methodology.criterion", "Criterion"))}</th><th>{html.escape(self.document.label("methodology.threshold", "Threshold"))}</th><th>{html.escape(self.document.label("methodology.interpretation", "Interpretation"))}</th></tr></thead><tbody>{methodology_rows}</tbody></table>
        </div>
    </section>
</section>
'''

    def _environment_rows(self, spec: ReportSpec, source: str) -> list[tuple[str, Any]]:
        return [
            (self.document.label("appendix.truthound_version", "Truthound version"), self.document.framework_version),
            (self.document.label("appendix.python_version", "Python version"), sys.version.split()[0]),
            (self.document.label("appendix.platform", "Platform"), platform.platform()),
            (self.document.label("appendix.theme", "Theme"), self.document.theme_name),
            (self.document.label("appendix.language", "Language"), spec.config.language),
            (self.document.label("appendix.generated_at", "Generated at"), spec.metadata.created_at.strftime(spec.config.date_format)),
            (self.document.label("appendix.source_label", "Source label"), source),
            (self.document.label("appendix.input_fingerprint", "Input fingerprint"), self.document.input_fingerprint(spec)),
        ]

    @staticmethod
    def _metric_definitions() -> list[tuple[str, str, str]]:
        return [
            ("stats.rows", "appendix.formula.rows", "Total number of profiled rows."),
            ("stats.columns", "appendix.formula.columns", "Total number of profiled columns."),
            ("stats.missing", "appendix.formula.missing", "Total null cells observed in profiled columns."),
            ("stats.duplicates", "appendix.formula.duplicates", "Rows identified as duplicates by the supplied profile."),
            ("stats.duplicate_ratio", "appendix.formula.duplicate_ratio", "Duplicate rows divided by total rows."),
            ("quality.score", "appendix.formula.quality_score", "Existing Truthound profile quality score; this report does not alter its calculation."),
        ]

    @staticmethod
    def _render_profile_row(column: dict[str, Any]) -> str:
        return (
            "<tr>"
            f"<td>{html.escape(str(column.get('name', '')))}</td>"
            f"<td>{html.escape(profile_type_label(column))}</td>"
            f"<td>{column.get('null_ratio', 0):.1%}</td>"
            f"<td>{column.get('unique_ratio', 0):.1%}</td>"
            f"<td>{int(column.get('distinct_count', 0)):,}</td>"
            "</tr>"
        )
