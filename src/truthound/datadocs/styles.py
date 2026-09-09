"""CSS styles for Data Docs reports.

This module provides comprehensive CSS styling for the static HTML reports.
The styles are designed to be:
- Korean public/research A4 report-oriented
- Responsive for all screen sizes
- Print-friendly
- Dark/Light mode compatible
"""

# =============================================================================
# Base CSS - Core styles and reset
# =============================================================================

BASE_CSS = """
/* Reset and Base */
*, *::before, *::after {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}

html {
    font-size: 16px;
    scroll-behavior: smooth;
}

body {
    font-family: var(--font-family);
    font-size: var(--font-size-base);
    line-height: var(--line-height-normal);
    color: var(--color-text-primary);
    background-color: var(--color-background);
    letter-spacing: 0;
    word-break: keep-all;
    overflow-wrap: break-word;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    print-color-adjust: exact;
    -webkit-print-color-adjust: exact;
}

/* Typography */
h1, h2, h3, h4, h5, h6 {
    font-weight: var(--font-weight-semibold);
    line-height: var(--line-height-tight);
    color: var(--color-text-primary);
    margin-bottom: var(--spacing-sm);
}

h1 { font-size: var(--font-size-3xl); }
h2 { font-size: var(--font-size-2xl); }
h3 { font-size: var(--font-size-xl); }
h4 { font-size: var(--font-size-lg); }

p {
    margin-bottom: var(--spacing-md);
    color: var(--color-text-secondary);
}

a {
    color: var(--color-primary);
    text-decoration: none;
    transition: color 0.2s ease;
}

a:hover {
    color: var(--color-secondary);
    text-decoration: underline;
}

code {
    font-family: var(--font-family-mono);
    font-size: var(--font-size-sm);
    background-color: var(--color-surface);
    padding: 0.125rem 0.375rem;
    border-radius: var(--border-radius-sm);
    border: 1px solid var(--color-border);
}

/* Utilities */
.text-primary { color: var(--color-text-primary); }
.text-secondary { color: var(--color-text-secondary); }
.text-success { color: var(--color-success); }
.text-warning { color: var(--color-warning); }
.text-error { color: var(--color-error); }
.text-info { color: var(--color-info); }

.hidden { display: none !important; }
.sr-only {
    position: absolute;
    width: 1px;
    height: 1px;
    padding: 0;
    margin: -1px;
    overflow: hidden;
    clip: rect(0, 0, 0, 0);
    border: 0;
}
"""

# =============================================================================
# Layout CSS
# =============================================================================

LAYOUT_CSS = """
/* Main Layout */
.report-container {
    max-width: 1400px;
    margin: 0 auto;
    padding: var(--spacing-lg);
}

/* Header */
.report-header {
    display: flex;
    flex-direction: column;
    gap: var(--spacing-md);
    padding-bottom: var(--spacing-xl);
    border-bottom: 1px solid var(--color-border);
    margin-bottom: var(--spacing-xl);
}

.report-header-main {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    flex-wrap: wrap;
    gap: var(--spacing-md);
}

.report-title {
    font-size: var(--font-size-3xl);
    font-weight: var(--font-weight-bold);
    margin: 0;
}

.report-subtitle {
    font-size: var(--font-size-lg);
    color: var(--color-text-secondary);
    margin-top: var(--spacing-xs);
}

.report-meta {
    display: flex;
    flex-wrap: wrap;
    gap: var(--spacing-md);
    font-size: var(--font-size-sm);
    color: var(--color-text-secondary);
}

.report-meta-item {
    display: flex;
    align-items: center;
    gap: var(--spacing-xs);
}

.report-logo {
    max-height: 48px;
    width: auto;
}

/* Table of Contents */
.report-toc {
    background-color: var(--color-surface);
    border-radius: var(--border-radius-lg);
    padding: var(--spacing-lg);
    margin-bottom: var(--spacing-xl);
}

.toc-title {
    font-size: var(--font-size-lg);
    margin-bottom: var(--spacing-md);
}

.toc-list {
    list-style: none;
    display: flex;
    flex-wrap: wrap;
    gap: var(--spacing-sm);
}

.toc-item a {
    display: inline-block;
    padding: var(--spacing-xs) var(--spacing-md);
    background-color: var(--color-background);
    border-radius: var(--border-radius-md);
    font-size: var(--font-size-sm);
    color: var(--color-text-primary);
    transition: all 0.2s ease;
}

.toc-item a:hover {
    background-color: var(--color-primary);
    color: white;
    text-decoration: none;
}

/* Sections */
.report-section {
    margin-bottom: var(--spacing-2xl);
}

.section-header {
    margin-bottom: var(--spacing-lg);
    padding-bottom: var(--spacing-sm);
    border-bottom: 2px solid var(--color-primary);
}

.section-title {
    font-size: var(--font-size-2xl);
    color: var(--color-text-primary);
    margin: 0;
}

.section-subtitle {
    font-size: var(--font-size-base);
    color: var(--color-text-secondary);
    margin-top: var(--spacing-xs);
}

.section-content {
    display: flex;
    flex-direction: column;
    gap: var(--spacing-lg);
}

/* Footer */
.report-footer {
    margin-top: var(--spacing-2xl);
    padding-top: var(--spacing-lg);
    border-top: 1px solid var(--color-border);
    text-align: center;
    font-size: var(--font-size-sm);
    color: var(--color-text-secondary);
}
"""

# =============================================================================
# Korean public/research A4 report CSS
# =============================================================================

REPORT_DOCUMENT_CSS = """
/* A4 report document shell */
body {
    padding: 10mm 0;
}

.report-container,
.container {
    width: 210mm;
    max-width: 210mm;
    min-height: 297mm;
    margin: 0 auto;
    padding: 20mm 18mm 16mm;
    background-color: var(--color-surface);
    border: 1px solid var(--color-border);
    box-shadow: 0 2mm 10mm var(--color-shadow);
}

.report-header,
.hero {
    display: block;
    margin-bottom: 9mm;
    padding: 0 0 5mm;
    background: transparent;
    border: 0;
    border-bottom: 2pt solid var(--color-primary);
    border-radius: 0;
    box-shadow: none;
}

.report-header-main {
    align-items: flex-start;
    gap: 6mm;
}

.report-title,
.hero h1,
.cover-title {
    color: var(--color-primary);
    font-size: var(--font-size-3xl);
    font-weight: var(--font-weight-bold);
    line-height: 1.3;
    letter-spacing: 0;
}

.report-subtitle,
.hero .muted {
    color: var(--color-text-secondary);
    font-size: var(--font-size-lg);
    margin-top: 2mm;
}

.report-meta {
    display: flex;
    gap: 5mm;
    margin-top: 4mm;
    padding-top: 3mm;
    border-top: 0.5pt solid var(--color-border);
    font-size: var(--font-size-sm);
}

.report-toc,
.report-summary,
.summary-box,
.quality-overview {
    margin-bottom: 8mm;
    padding: 5mm 6mm;
    background-color: rgba(31, 78, 121, 0.06);
    border: 0.8pt solid var(--color-border);
    border-left: 3pt solid var(--color-primary);
    border-radius: var(--border-radius-md);
    box-shadow: none;
    break-inside: avoid;
}

.toc-title,
.report-section h2,
.section-title {
    color: var(--color-primary);
    font-size: var(--font-size-xl);
    font-weight: var(--font-weight-bold);
    line-height: 1.35;
    letter-spacing: 0;
}

.toc-title::before,
.report-section h2::before,
.section-title::before {
    content: "□ ";
}

.toc-list {
    display: block;
    list-style: none;
    margin-top: 3mm;
}

.toc-item {
    margin: 1.5mm 0;
}

.toc-item a {
    display: inline;
    padding: 0;
    background: transparent;
    border-radius: 0;
    color: var(--color-text-primary);
}

.toc-item a::before {
    content: "○ ";
    color: var(--color-secondary);
}

.report-section {
    margin-bottom: 9mm;
    padding: 0;
    background: transparent;
    border: 0;
    border-radius: 0;
    box-shadow: none;
    break-inside: auto;
    page-break-inside: auto;
}

.panel {
    margin-bottom: 9mm;
    padding: 0;
    background: transparent;
    border: 0;
    border-radius: 0;
    box-shadow: none;
    break-inside: avoid;
    page-break-inside: avoid;
}

.section-header {
    margin-bottom: 4mm;
    padding-bottom: 2mm;
    border-bottom: 1.2pt solid var(--color-primary);
}

.section-content {
    gap: 4mm;
}

.section-content p,
.report-section p {
    color: var(--color-text-primary);
    margin-bottom: 3mm;
}

.executive-summary,
.quality-framework,
.report-appendix {
    margin-bottom: 9mm;
    break-inside: avoid;
    page-break-inside: avoid;
}

.report-chapter {
    margin-bottom: 9mm;
    break-inside: auto;
    page-break-inside: auto;
}

.executive-summary {
    padding: 6mm;
    background-color: rgba(31, 78, 121, 0.045);
    border: 0.8pt solid var(--color-border);
    border-left: 3pt solid var(--color-primary);
}

.executive-summary-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 3mm;
}

.executive-summary-item {
    padding: 3mm;
    background-color: var(--color-surface);
    border: 0.5pt solid var(--color-border);
}

.executive-summary-item h3 {
    margin: 0 0 1.5mm;
    color: var(--color-primary);
    font-size: var(--font-size-sm);
    font-weight: var(--font-weight-bold);
}

.executive-summary-item p,
.report-paragraph {
    margin: 0;
    color: var(--color-text-primary);
    font-size: var(--font-size-sm);
    line-height: 1.55;
}

.chapter-lead,
.appendix-lead {
    margin: 0 0 4mm;
    padding-left: 3mm;
    border-left: 2pt solid var(--color-primary);
    color: var(--color-text-secondary);
    word-break: keep-all;
    overflow-wrap: anywhere;
}

.chapter-header {
    margin: 10mm 0 5mm;
    padding-bottom: 2.5mm;
    border-bottom: 1.5pt solid var(--color-primary);
}

.chapter-title,
.appendix-title {
    margin: 0;
    color: var(--color-primary);
    font-size: var(--font-size-2xl);
    font-weight: var(--font-weight-bold);
    line-height: 1.35;
}

.appendix-title {
    font-size: var(--font-size-xl);
    border-bottom: 1pt solid var(--color-primary);
    padding-bottom: 2mm;
    margin-bottom: 4mm;
}

.auditability-block {
    padding: 4mm;
    background-color: var(--color-surface);
    border: 0.6pt solid var(--color-border);
    break-inside: avoid;
}

.data-list {
    display: grid;
    grid-template-columns: minmax(34mm, auto) 1fr;
    gap: 0;
    border-top: 0.6pt solid var(--color-border);
    border-left: 0.6pt solid var(--color-border);
}

.data-list dt,
.data-list dd {
    margin: 0;
    padding: 2mm 3mm;
    border-right: 0.6pt solid var(--color-border);
    border-bottom: 0.6pt solid var(--color-border);
    font-size: var(--font-size-sm);
}

.data-list dt {
    background-color: rgba(31, 78, 121, 0.08);
    color: var(--color-primary);
    font-weight: var(--font-weight-semibold);
}

.table-container {
    overflow-x: auto;
    margin-bottom: 6mm;
    break-inside: avoid;
}

.table-title,
.chart-title {
    color: var(--color-primary);
    font-size: var(--font-size-lg);
    font-weight: var(--font-weight-semibold);
    margin-bottom: 2.5mm;
}

table,
.data-table {
    width: 100%;
    border-collapse: collapse;
    table-layout: auto;
    font-size: var(--font-size-sm);
    border-top: 1pt solid var(--color-primary);
    border-left: 0.5pt solid var(--color-border);
}

table th,
table td,
.data-table th,
.data-table td {
    color: var(--color-text-primary);
    padding: 2mm 2.5mm;
    border-right: 0.5pt solid var(--color-border);
    border-bottom: 0.5pt solid var(--color-border);
    text-align: left;
    vertical-align: middle;
}

table.report-object-table th,
table.report-object-table td,
.data-table.report-object-table th,
.data-table.report-object-table td {
    word-break: keep-all;
    overflow-wrap: anywhere;
}

table th,
.data-table th {
    background-color: var(--color-primary);
    color: var(--color-background);
    font-weight: var(--font-weight-semibold);
    text-align: center;
}

table td:nth-child(n+3),
.data-table td:nth-child(n+3) {
    text-align: right;
}

.quality-coverage-table {
    table-layout: fixed;
}

.quality-coverage-table th:nth-child(1),
.quality-coverage-table td:nth-child(1) {
    width: 22%;
}

.quality-coverage-table th:nth-child(2),
.quality-coverage-table td:nth-child(2) {
    width: 18%;
    text-align: center;
}

.quality-coverage-table th:nth-child(3),
.quality-coverage-table td:nth-child(3) {
    width: 60%;
    text-align: left;
    line-height: 1.55;
}

.methodology-table {
    table-layout: fixed;
}

.methodology-table th:nth-child(1),
.methodology-table td:nth-child(1) {
    width: 26%;
}

.methodology-table th:nth-child(2),
.methodology-table td:nth-child(2) {
    width: 22%;
    text-align: center;
    white-space: nowrap;
}

.methodology-table th:nth-child(3),
.methodology-table td:nth-child(3) {
    width: 52%;
    text-align: left;
    line-height: 1.55;
}

table tbody tr:nth-child(even),
.data-table tbody tr:nth-child(even) {
    background-color: rgba(31, 78, 121, 0.035);
}

.metric-card,
.quality-score-card,
.column-card,
.chart-container,
.pattern-item,
.correlation-item,
.recommendation-item,
.validator-suggestion,
.panel {
    border: 0.6pt solid var(--color-border);
    border-radius: var(--border-radius-md);
    box-shadow: none;
}

.metrics-grid,
.quality-scores-grid {
    gap: 3mm;
}

.metric-card,
.quality-score-card {
    padding: 4mm;
    background-color: var(--color-surface);
}

.metric-value {
    color: var(--color-primary);
}

.chart-container {
    padding: 4mm;
    background-color: var(--color-surface);
    break-inside: avoid;
    page-break-inside: avoid;
}

.chart-container::after,
.figure-caption,
.report-caption {
    display: block;
    margin-top: 2mm;
    color: var(--color-text-secondary);
    font-size: var(--font-size-xs);
    text-align: center;
}

.alert {
    border-width: 0.6pt 0.6pt 0.6pt 3pt;
    border-style: solid;
    border-radius: var(--border-radius-md);
    break-inside: avoid;
}

.recommendations-list {
    list-style: none;
    padding-left: 0;
}

.recommendation-item {
    position: relative;
    padding-left: 7mm;
    background-color: var(--color-surface);
}

.recommendation-item::before {
    content: "-";
    position: absolute;
    left: 3mm;
    color: var(--color-secondary);
    font-weight: var(--font-weight-bold);
}
"""

# =============================================================================
# Components CSS
# =============================================================================

COMPONENTS_CSS = """
/* Metric Cards */
.metrics-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: var(--spacing-md);
}

.metric-card {
    background-color: var(--color-surface);
    border-radius: var(--border-radius-lg);
    padding: var(--spacing-lg);
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
    box-shadow: var(--shadow-sm);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-md);
}

.metric-icon {
    width: 40px;
    height: 40px;
    margin-bottom: var(--spacing-sm);
    background-color: var(--color-primary);
    border-radius: var(--border-radius-md);
    opacity: 0.8;
}

.metric-value {
    font-size: var(--font-size-2xl);
    font-weight: var(--font-weight-bold);
    color: var(--color-text-primary);
    line-height: 1;
}

.metric-label {
    font-size: var(--font-size-sm);
    color: var(--color-text-secondary);
    margin-top: var(--spacing-xs);
}

/* Column Cards */
.columns-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
    gap: var(--spacing-lg);
}

.column-card {
    background-color: var(--color-surface);
    border-radius: var(--border-radius-lg);
    padding: var(--spacing-lg);
    box-shadow: var(--shadow-sm);
}

.column-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: var(--spacing-md);
    padding-bottom: var(--spacing-sm);
    border-bottom: 1px solid var(--color-border);
}

.column-name {
    font-size: var(--font-size-lg);
    font-weight: var(--font-weight-semibold);
    margin: 0;
}

.column-type {
    font-size: var(--font-size-xs);
    padding: 0.25rem 0.5rem;
    border-radius: var(--border-radius-sm);
    font-family: var(--font-family-mono);
}

.type-numeric { background-color: #dbeafe; color: #1e40af; }
.type-string { background-color: #fef3c7; color: #92400e; }
.type-datetime { background-color: #d1fae5; color: #065f46; }
.type-boolean { background-color: #f3e8ff; color: #6b21a8; }
.type-email { background-color: #fee2e2; color: #991b1b; }
.type-url { background-color: #cffafe; color: #0e7490; }
.type-phone { background-color: #fce7f3; color: #9d174d; }
.type-other { background-color: #f3f4f6; color: #374151; }

.column-metrics {
    display: flex;
    gap: var(--spacing-md);
    margin-bottom: var(--spacing-md);
}

.metric-mini {
    flex: 1;
    text-align: center;
    padding: var(--spacing-sm);
    background-color: var(--color-background);
    border-radius: var(--border-radius-sm);
}

.metric-mini .metric-label {
    display: block;
    font-size: var(--font-size-xs);
    color: var(--color-text-secondary);
}

.metric-mini .metric-value {
    display: block;
    font-size: var(--font-size-base);
    font-weight: var(--font-weight-semibold);
}

.metric-mini.quality-good .metric-value { color: var(--color-success); }
.metric-mini.quality-warning .metric-value { color: var(--color-warning); }
.metric-mini.quality-bad .metric-value { color: var(--color-error); }

.column-stats {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: var(--spacing-sm);
    margin-bottom: var(--spacing-md);
}

.column-stats .stat {
    text-align: center;
}

.column-stats .stat-label {
    display: block;
    font-size: var(--font-size-xs);
    color: var(--color-text-secondary);
}

.column-stats .stat-value {
    color: var(--color-text-primary);
    display: block;
    font-size: var(--font-size-sm);
    font-weight: var(--font-weight-medium);
}

.column-patterns {
    display: flex;
    flex-wrap: wrap;
    gap: var(--spacing-xs);
    margin-bottom: var(--spacing-md);
}

.pattern-tag {
    font-size: var(--font-size-xs);
    padding: 0.125rem 0.5rem;
    background-color: var(--color-primary);
    color: white;
    border-radius: var(--border-radius-sm);
}

/* Tables */
.table-container {
    overflow-x: auto;
    margin-bottom: var(--spacing-lg);
}

.table-title {
    font-size: var(--font-size-lg);
    margin-bottom: var(--spacing-sm);
}

.data-table {
    width: 100%;
    border-collapse: collapse;
    font-size: var(--font-size-sm);
}

.data-table th,
.data-table td {
    padding: var(--spacing-sm) var(--spacing-md);
    text-align: left;
    border-bottom: 1px solid var(--color-border);
}

.data-table th {
    background-color: var(--color-surface);
    font-weight: var(--font-weight-semibold);
    color: var(--color-text-primary);
    white-space: nowrap;
}

.data-table tr:hover {
    background-color: var(--color-surface);
}

/* Charts */
.chart-container {
    background-color: var(--color-surface);
    border-radius: var(--border-radius-lg);
    padding: var(--spacing-lg);
    margin-bottom: var(--spacing-md);
}

.chart-title {
    font-size: var(--font-size-lg);
    margin-bottom: var(--spacing-xs);
}

.chart-subtitle {
    font-size: var(--font-size-sm);
    color: var(--color-text-secondary);
    margin-bottom: var(--spacing-md);
}

.charts-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
    gap: var(--spacing-lg);
}

.svg-chart {
    width: 100%;
    height: auto;
}

.svg-chart .chart-label {
    font-size: 12px;
    fill: var(--color-text-secondary);
}

.svg-chart .chart-empty {
    font-size: 14px;
    fill: var(--color-text-secondary);
}

/* ApexCharts Base Styles (Theme-aware) */
.apexcharts-tooltip {
    background-color: var(--color-surface) !important;
    border: 1px solid var(--color-border) !important;
    border-radius: var(--border-radius-md) !important;
    box-shadow: var(--shadow-md) !important;
}

.apexcharts-tooltip-title {
    background-color: var(--color-background) !important;
    border-bottom: 1px solid var(--color-border) !important;
    padding: var(--spacing-xs) var(--spacing-sm) !important;
}

.apexcharts-legend-text {
    color: var(--color-text-primary) !important;
}

.apexcharts-menu {
    background-color: var(--color-surface) !important;
    border: 1px solid var(--color-border) !important;
    border-radius: var(--border-radius-md) !important;
    box-shadow: var(--shadow-md) !important;
}

.apexcharts-menu-item {
    color: var(--color-text-primary) !important;
}

.apexcharts-menu-item:hover {
    background-color: var(--color-background) !important;
}

/* Alerts */
.alerts-container {
    display: flex;
    flex-direction: column;
    gap: var(--spacing-md);
}

.alert {
    padding: var(--spacing-md);
    border-radius: var(--border-radius-md);
    border-left: 4px solid;
}

.alert-info {
    background-color: #eff6ff;
    border-color: var(--color-info);
}

.alert-warning {
    background-color: #fffbeb;
    border-color: var(--color-warning);
}

.alert-error {
    background-color: #fef2f2;
    border-color: var(--color-error);
}

.alert-critical {
    background-color: #fef2f2;
    border-color: #7f1d1d;
}

.alert-header {
    display: flex;
    align-items: center;
    gap: var(--spacing-sm);
    margin-bottom: var(--spacing-xs);
}

.alert-title {
    font-weight: var(--font-weight-semibold);
    color: var(--color-text-primary);
}

.alert-message {
    font-size: var(--font-size-sm);
    color: var(--color-text-secondary);
}

.alert-suggestion {
    margin-top: var(--spacing-sm);
    font-size: var(--font-size-sm);
    color: var(--color-text-secondary);
    font-style: italic;
}

/* Patterns */
.patterns-list {
    display: flex;
    flex-direction: column;
    gap: var(--spacing-sm);
}

.pattern-item {
    background-color: var(--color-surface);
    border-radius: var(--border-radius-md);
    padding: var(--spacing-md);
}

.pattern-header {
    display: flex;
    align-items: center;
    gap: var(--spacing-md);
    flex-wrap: wrap;
}

.pattern-column {
    font-weight: var(--font-weight-semibold);
    color: var(--color-text-primary);
}

.pattern-name {
    font-family: var(--font-family-mono);
    font-size: var(--font-size-sm);
    color: var(--color-secondary);
}

.pattern-match {
    font-size: var(--font-size-sm);
    padding: 0.125rem 0.5rem;
    border-radius: var(--border-radius-sm);
}

.match-high { background-color: #d1fae5; color: #065f46; }
.match-medium { background-color: #fef3c7; color: #92400e; }
.match-low { background-color: #fee2e2; color: #991b1b; }

.pattern-samples {
    margin-top: var(--spacing-sm);
    font-size: var(--font-size-sm);
    color: var(--color-text-secondary);
}

/* Correlations */
.correlations-list {
    display: flex;
    flex-direction: column;
    gap: var(--spacing-sm);
}

.correlation-item {
    display: flex;
    align-items: center;
    gap: var(--spacing-md);
    padding: var(--spacing-md);
    background-color: var(--color-surface);
    border-radius: var(--border-radius-md);
}

.corr-col {
    font-weight: var(--font-weight-medium);
    color: var(--color-text-primary);
}

.corr-arrow {
    color: var(--color-text-secondary);
}

.corr-value {
    font-family: var(--font-family-mono);
    font-weight: var(--font-weight-semibold);
    margin-left: auto;
}

.corr-value.positive { color: var(--color-success); }
.corr-value.negative { color: var(--color-error); }

.corr-strong { border-left: 3px solid var(--color-primary); }
.corr-moderate { border-left: 3px solid var(--color-secondary); }
.corr-weak { border-left: 3px solid var(--color-border); }

/* Quality Scores */
.quality-scores-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: var(--spacing-lg);
}

.quality-score-card {
    text-align: center;
    padding: var(--spacing-lg);
    background-color: var(--color-surface);
    border-radius: var(--border-radius-lg);
}

.score-ring {
    width: 100px;
    height: 100px;
    margin: 0 auto var(--spacing-md);
}

.circular-chart {
    display: block;
    margin: 0 auto;
    max-width: 100%;
    max-height: 100px;
}

.circle-bg {
    fill: none;
    stroke: var(--color-border);
    stroke-width: 3.8;
}

.circle {
    fill: none;
    stroke-width: 2.8;
    stroke-linecap: round;
    animation: progress 1s ease-out forwards;
}

.score-good .circle { stroke: var(--color-success); }
.score-warning .circle { stroke: var(--color-warning); }
.score-bad .circle { stroke: var(--color-error); }

.percentage {
    fill: var(--color-text-primary);
    font-size: 0.5em;
    font-weight: var(--font-weight-bold);
    text-anchor: middle;
}

.score-label {
    font-size: var(--font-size-lg);
    font-weight: var(--font-weight-semibold);
    color: var(--color-text-primary);
}

.score-desc {
    font-size: var(--font-size-sm);
    color: var(--color-text-secondary);
    margin-top: var(--spacing-xs);
}

@keyframes progress {
    0% { stroke-dasharray: 0, 100; }
}

/* Recommendations */
.recommendations-list {
    list-style: none;
}

.recommendation-item {
    padding: var(--spacing-md);
    color: var(--color-text-primary);
    margin-bottom: var(--spacing-sm);
    background-color: var(--color-surface);
    border-radius: var(--border-radius-md);
    border-left: 3px solid var(--color-info);
    font-size: var(--font-size-sm);
}

.validators-section h4 {
    margin-bottom: var(--spacing-md);
}

.validators-list {
    display: flex;
    flex-direction: column;
    gap: var(--spacing-sm);
}

.validator-suggestion {
    display: flex;
    align-items: center;
    gap: var(--spacing-md);
    padding: var(--spacing-sm) var(--spacing-md);
    background-color: var(--color-surface);
    border-radius: var(--border-radius-sm);
}

.validator-column {
    font-weight: var(--font-weight-medium);
    min-width: 120px;
}

.validator-code {
    font-family: var(--font-family-mono);
    font-size: var(--font-size-sm);
}

/* No Data */
.no-data {
    text-align: center;
    padding: var(--spacing-xl);
    color: var(--color-text-secondary);
    font-style: italic;
}

/* Download Button */
.download-button {
    display: inline-flex;
    align-items: center;
    gap: var(--spacing-sm);
    padding: var(--spacing-sm) var(--spacing-md);
    background-color: var(--color-primary);
    color: white;
    border: none;
    border-radius: var(--border-radius-md);
    font-size: var(--font-size-sm);
    font-weight: var(--font-weight-medium);
    cursor: pointer;
    transition: background-color 0.2s ease;
}

.download-button:hover {
    background-color: var(--color-secondary);
}
"""

# =============================================================================
# Responsive CSS
# =============================================================================

RESPONSIVE_CSS = """
/* Responsive Design */
@media (max-width: 1200px) {
    .columns-grid {
        grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
    }
}

@media (max-width: 768px) {
    html {
        font-size: 14px;
    }

    .report-container {
        padding: var(--spacing-md);
    }

    .report-header-main {
        flex-direction: column;
    }

    .metrics-grid {
        grid-template-columns: repeat(2, 1fr);
    }

    .columns-grid {
        grid-template-columns: 1fr;
    }

    .charts-grid {
        grid-template-columns: 1fr;
    }

    .column-stats {
        grid-template-columns: repeat(2, 1fr);
    }

    .quality-scores-grid {
        grid-template-columns: repeat(2, 1fr);
    }

    .correlation-item {
        flex-wrap: wrap;
    }

    .toc-list {
        flex-direction: column;
    }
}

@media (max-width: 480px) {
    .metrics-grid {
        grid-template-columns: 1fr;
    }

    .quality-scores-grid {
        grid-template-columns: 1fr;
    }

    .column-metrics {
        flex-direction: column;
    }
}
"""

# =============================================================================
# Print CSS
# =============================================================================

PRINT_CSS = """
/* Print Styles */
@page {
    size: A4 portrait;
    margin: 16mm 14mm;
}

@media print {
    html {
        font-size: 10.5pt;
    }

    body {
        background-color: white;
        color: black;
        font-size: var(--font-size-base);
        margin: 0;
        padding: 0;
    }

    .report-container,
    .container {
        width: auto;
        max-width: none;
        min-height: auto;
        padding: 0;
        border: 0;
        box-shadow: none;
    }

    .download-button,
    .toc-list a:hover {
        display: none;
    }

    .panel,
    .summary-box,
    .report-summary,
    .quality-overview,
    .alert {
        page-break-inside: avoid;
        break-inside: avoid;
    }

    /* Long sections must fragment in block flow. Flex/grid fragmentation can
       detach headings and split otherwise page-sized column cards. */
    .report-section,
    .section-content {
        display: block;
        page-break-inside: auto;
        break-inside: auto;
    }

    .section-content > * + * {
        margin-top: 4mm;
    }

    .chapter-header,
    .chapter-lead,
    .section-header,
    .appendix-title,
    .appendix-lead,
    .toc-title-professional {
        page-break-after: avoid;
        break-after: avoid;
    }

    .columns-grid,
    .charts-grid {
        display: block;
    }

    .column-card {
        margin-bottom: 4mm;
        page-break-inside: avoid;
        break-inside: avoid;
    }

    .column-header > * {
        min-width: 0;
        overflow-wrap: anywhere;
    }

    .chart-container,
    figure {
        page-break-inside: avoid;
        break-inside: avoid;
    }

    .table-container,
    table,
    .data-table {
        page-break-inside: auto;
        break-inside: auto;
    }

    .data-table {
        table-layout: fixed;
    }

    .data-table th,
    .data-table td {
        white-space: normal;
        overflow-wrap: anywhere;
    }

    tr,
    thead,
    tfoot {
        page-break-inside: avoid;
        break-inside: avoid;
    }

    thead {
        display: table-header-group;
    }

    tfoot {
        display: table-footer-group;
    }

    .no-print {
        display: none !important;
    }

    a {
        color: inherit;
        text-decoration: underline;
    }

    .alert {
        border: 1px solid currentColor;
    }
}
"""

# =============================================================================
# Dark Mode Overrides
# =============================================================================

DARK_MODE_OVERRIDES = """
/* Dark Mode Type Badge Overrides */
.type-numeric { background-color: #1e3a5f; color: #93c5fd; }
.type-string { background-color: #422006; color: #fcd34d; }
.type-datetime { background-color: #064e3b; color: #6ee7b7; }
.type-boolean { background-color: #4c1d95; color: #c4b5fd; }
.type-email { background-color: #450a0a; color: #fca5a5; }
.type-url { background-color: #164e63; color: #67e8f9; }
.type-phone { background-color: #500724; color: #f9a8d4; }
.type-other { background-color: #374151; color: #d1d5db; }

/* Dark Mode Pattern Match */
.match-high { background-color: #064e3b; color: #6ee7b7; }
.match-medium { background-color: #422006; color: #fcd34d; }
.match-low { background-color: #450a0a; color: #fca5a5; }

/* Dark Mode Alerts */
.alert-info { background-color: #1e3a5f; }
.alert-warning { background-color: #422006; }
.alert-error { background-color: #450a0a; }
.alert-critical { background-color: #2d0a0a; }

/* Dark Mode ApexCharts Overrides */
.apexcharts-tooltip {
    background-color: var(--color-surface) !important;
    border-color: var(--color-border) !important;
    color: var(--color-text-primary) !important;
}

.apexcharts-tooltip-title {
    background-color: var(--color-background) !important;
    border-color: var(--color-border) !important;
    color: var(--color-text-primary) !important;
}

.apexcharts-tooltip-text,
.apexcharts-tooltip-text-y-label,
.apexcharts-tooltip-text-y-value,
.apexcharts-tooltip-text-goals-label,
.apexcharts-tooltip-text-goals-value {
    color: var(--color-text-primary) !important;
}

.apexcharts-xaxistooltip,
.apexcharts-yaxistooltip {
    background-color: var(--color-surface) !important;
    border-color: var(--color-border) !important;
    color: var(--color-text-primary) !important;
}

.apexcharts-xaxistooltip-text,
.apexcharts-yaxistooltip-text {
    color: var(--color-text-primary) !important;
}

/* ApexCharts Menu (Download, etc.) */
.apexcharts-menu {
    background-color: var(--color-surface) !important;
    border-color: var(--color-border) !important;
}

.apexcharts-menu-item {
    color: var(--color-text-primary) !important;
}

.apexcharts-menu-item:hover {
    background-color: var(--color-background) !important;
}

/* ApexCharts Legend */
.apexcharts-legend-text {
    color: var(--color-text-primary) !important;
}

/* ApexCharts Toolbar Icons */
.apexcharts-toolbar svg,
.apexcharts-reset-icon svg,
.apexcharts-zoom-icon svg,
.apexcharts-zoomin-icon svg,
.apexcharts-zoomout-icon svg,
.apexcharts-pan-icon svg,
.apexcharts-menu-icon svg,
.apexcharts-selection-icon svg {
    fill: var(--color-text-secondary) !important;
}

.apexcharts-toolbar svg:hover {
    fill: var(--color-text-primary) !important;
}

/* ApexCharts Data Labels */
.apexcharts-datalabel,
.apexcharts-datalabel-label,
.apexcharts-datalabel-value,
.apexcharts-pie-label,
.apexcharts-donut-label {
    fill: var(--color-text-primary) !important;
}

/* ApexCharts Title and Subtitle */
.apexcharts-title-text {
    fill: var(--color-text-primary) !important;
}

.apexcharts-subtitle-text {
    fill: var(--color-text-secondary) !important;
}

/* ApexCharts Axis Labels */
.apexcharts-xaxis text,
.apexcharts-yaxis text,
.apexcharts-xaxis-label,
.apexcharts-yaxis-label {
    fill: var(--color-text-secondary) !important;
}

/* ApexCharts Axis Title */
.apexcharts-xaxis-title text,
.apexcharts-yaxis-title text {
    fill: var(--color-text-secondary) !important;
}

/* ApexCharts Radial Bar (Gauge) Labels */
.apexcharts-radialbar-track text,
.apexcharts-radialbar text {
    fill: var(--color-text-primary) !important;
}

/* ApexCharts Heatmap Labels */
.apexcharts-heatmap-rect text {
    fill: var(--color-text-primary) !important;
}

/* ApexCharts Annotations */
.apexcharts-annotation-label {
    fill: var(--color-text-primary) !important;
}
"""


# =============================================================================
# Complete Stylesheet
# =============================================================================

def get_complete_stylesheet(
    theme_css_vars: str,
    is_dark: bool = False,
    *,
    include_apexcharts: bool = True,
) -> str:
    """Generate the complete stylesheet.

    Args:
        theme_css_vars: CSS custom properties for the theme
        is_dark: Whether this is a dark theme
        include_apexcharts: Whether to include ApexCharts-specific CSS

    Returns:
        Complete CSS stylesheet
    """
    components_css = COMPONENTS_CSS
    dark_overrides = DARK_MODE_OVERRIDES if is_dark else ""

    if not include_apexcharts:
        components_css = _strip_apexcharts_components(components_css)
        dark_overrides = _strip_apexcharts_dark_overrides(dark_overrides)

    return f"""
{theme_css_vars}

{BASE_CSS}

{LAYOUT_CSS}

{REPORT_DOCUMENT_CSS}

{components_css}

{RESPONSIVE_CSS}

{PRINT_CSS}

{dark_overrides}
"""


def _strip_apexcharts_components(css: str) -> str:
    start = "/* ApexCharts Base Styles (Theme-aware) */"
    end = "/* Alerts */"
    if start not in css or end not in css:
        return css
    before, remainder = css.split(start, 1)
    _, after = remainder.split(end, 1)
    return before + end + after


def _strip_apexcharts_dark_overrides(css: str) -> str:
    start = "/* Dark Mode ApexCharts Overrides */"
    if start not in css:
        return css
    before, _ = css.split(start, 1)
    return before
