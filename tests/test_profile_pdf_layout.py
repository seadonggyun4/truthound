"""Real Profile PDF pagination regressions using only synthetic Core inputs."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from io import BytesIO

import polars as pl
import pytest

from truthound.datadocs import HTMLReportBuilder
from truthound.profiler import DataProfiler


def synthetic_profile():
    columns = {f"synthetic_measure_{index:02d}": [index, index + 1] for index in range(8)}
    columns.update(
        {
            "synthetic_category": ["alpha", "beta"],
            "synthetic_flag": [True, False],
            "synthetic_long_" + "column_segment_" * 8: ["alpha", "beta"],
        }
    )
    return (
        DataProfiler().profile(pl.DataFrame(columns).lazy(), name="Synthetic PDF layout").to_dict()
    )


def render_profile(profile, *, theme="light", language="en"):
    weasyprint = pytest.importorskip("weasyprint")
    before = deepcopy(profile)
    builder = HTMLReportBuilder(theme=theme, language=language, _use_svg=True)
    html = builder.build_for_pdf(profile, title="Synthetic Profile PDF layout")
    source = weasyprint.HTML(string=html)
    document = source.render()
    # Exercise actual PDF serialization as well as its pre-paint layout tree.
    pypdf = pytest.importorskip("pypdf")
    pdf = document.write_pdf()
    assert len(pypdf.PdfReader(BytesIO(pdf)).pages) == len(document.pages)
    assert profile == before, "Print rendering must not rewrite the Core result"
    return source, document, pdf


def _walk(box, ancestors=()):
    yield box, ancestors
    for child in getattr(box, "children", ()):
        yield from _walk(child, (*ancestors, box))


def _classes(element):
    return set(element.get("class", "").split()) if element is not None else set()


def _element_pages(document, *, text_only=False):
    result = defaultdict(set)
    for page_index, page in enumerate(document.pages):
        for box, _ in _walk(page._page_box):
            if box.element is not None and (not text_only or getattr(box, "text", "").strip()):
                result[box.element].add(page_index)
    return result


def _first_text_page(element, pages):
    values = [page for child in element.iter() for page in pages.get(child, ())]
    return min(values) if values else None


@pytest.fixture(
    scope="module",
    params=[
        (theme, language) for theme in ("light", "dark", "minimal") for language in ("en", "ko")
    ],
)
def profile_pdf(request):
    theme, language = request.param
    return render_profile(synthetic_profile(), theme=theme, language=language)


def test_profile_pdf_headings_stay_with_their_content(profile_pdf):
    source, document, _ = profile_pdf
    text_pages = _element_pages(document, text_only=True)
    groups = {
        "report-chapter": ("chapter-header", "section-content"),
        "report-section": ("section-header", "section-content"),
        "executive-summary": ("section-header", "executive-summary-grid"),
        "report-toc-professional": ("toc-title-professional", "toc-list-professional"),
        "report-appendix": ("appendix-title", "table-container"),
    }
    orphaned = []
    for element in source.etree_element.iter():
        for group_class, (heading_class, content_class) in groups.items():
            if group_class not in _classes(element):
                continue
            heading = next(
                (child for child in element.iter() if heading_class in _classes(child)), None
            )
            content = next(
                (child for child in element.iter() if content_class in _classes(child)), None
            )
            if heading is None or content is None:
                continue
            heading_page = _first_text_page(heading, text_pages)
            content_page = _first_text_page(content, text_pages)
            if content_page is not None and heading_page != content_page:
                orphaned.append((group_class, heading_page, content_page))
    assert not orphaned, f"Heading-only fragments: {orphaned}"


def test_profile_pdf_column_cards_do_not_fragment(profile_pdf):
    source, document, _ = profile_pdf
    pages = _element_pages(document)
    cards = [
        element for element in source.etree_element.iter() if "column-card" in _classes(element)
    ]
    assert len(cards) == 11
    card_pages = [set().union(*(pages[child] for child in card.iter())) for card in cards]
    fragments = [sorted(values) for values in card_pages if len(values) != 1]
    assert not fragments, f"Short column cards split across pages: {fragments}"


def test_profile_pdf_text_stays_inside_printable_width(profile_pdf):
    _, document, _ = profile_pdf
    out_of_bounds = []
    for index, page in enumerate(document.pages):
        root = page._page_box
        left = root.content_box_x()
        right = left + root.width
        for box, ancestors in _walk(root):
            if not getattr(box, "text", "").strip() or any(
                type(parent).__name__ == "MarginBox" for parent in ancestors
            ):
                continue
            if box.position_x < left - 1 or box.position_x + box.width > right + 1:
                out_of_bounds.append(
                    (
                        index,
                        round(box.position_x - left, 1),
                        round(box.position_x + box.width - right, 1),
                    )
                )
    assert not out_of_bounds, f"Text outside printable page bounds: {out_of_bounds[:12]}"


def test_profile_pdf_does_not_end_with_footer_only_page(profile_pdf):
    _, document, _ = profile_pdf
    substantive = []
    for box, ancestors in _walk(document.pages[-1]._page_box):
        if not getattr(box, "text", "").strip():
            continue
        if any(
            type(parent).__name__ == "MarginBox"
            or "report-footer-professional" in _classes(parent.element)
            for parent in ancestors
        ):
            continue
        substantive.append(box)
    assert substantive, "Report footer must remain with the final content"


def test_profile_pdf_retains_every_column_name(profile_pdf):
    _, _, pdf = profile_pdf
    pypdf = pytest.importorskip("pypdf")
    text = "".join(
        "".join(page.extract_text().split()) for page in pypdf.PdfReader(BytesIO(pdf)).pages
    )
    for column in synthetic_profile()["columns"]:
        assert column["name"] in text
