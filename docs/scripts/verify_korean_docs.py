from __future__ import annotations

import argparse
import html as html_lib
import re
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlsplit

DISALLOWED = (
    "한국어 미러",
    "Truthound Depot",
    "Data Docs Dashboard",
    "truthound-dashboard",
    "/dashboard/",
    "/depot/",
)

# These local Core interfaces belong to the full reference site, not the
# public portal. Allow exact navigation/edit links only; never arbitrary text,
# external applications, or routes that merely contain the same substring.
CORE_UI_ROUTES = {
    "/dashboard/", "/cli/dashboard/", "/guides/datadocs/dashboard/",
}
CORE_UI_SOURCES = {
    "dashboard/index.md", "cli/dashboard.md", "guides/datadocs/dashboard.md",
}
ATTRIBUTE = re.compile(
    r'''(?P<name>[^\s"'<>/=]+)\s*=\s*(?:"(?P<double>[^"]*)"|'(?P<single>[^']*)'|(?P<bare>[^\s"'=<>`]+))''',
    re.DOTALL,
)


def _approved_core_ui_href(value: str, relative_page: str) -> bool:
    base = "https://truthound.netlify.app/" + relative_page.removesuffix("index.html")
    try:
        parsed = urlsplit(urljoin(base, html_lib.unescape(value)))
    except ValueError:
        return False
    if parsed.query or parsed.fragment:
        return False
    if parsed.scheme == "https" and parsed.netloc == "truthound.netlify.app":
        path = parsed.path.removeprefix("/en") if parsed.path.startswith("/en/") else parsed.path
        return path in CORE_UI_ROUTES
    if parsed.scheme == "https" and parsed.netloc == "github.com":
        prefix = "/seadonggyun4/Truthound/edit/main/docs/"
        if not parsed.path.startswith(prefix):
            return False
        source = parsed.path.removeprefix(prefix)
        if source.startswith(("ko/", "en/")):
            source = source[3:]
        return source in CORE_UI_SOURCES
    return False


class CoreUILinkPolicy(HTMLParser):
    def __init__(self, html: str, relative_page: str) -> None:
        super().__init__(convert_charrefs=False)
        self.relative_page = relative_page
        self.line_offsets = [0, *(match.end() for match in re.finditer("\n", html))]
        self.allowed_spans: list[tuple[int, int]] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag not in {"a", "link"}:
            return
        raw = self.get_starttag_text()
        line, column = self.getpos()
        offset = self.line_offsets[line - 1] + column
        for match in ATTRIBUTE.finditer(raw):
            if match.group("name").lower() != "href":
                continue
            value_group = next(name for name in ("double", "single", "bare") if match.group(name) is not None)
            value = match.group(value_group)
            if ("href", html_lib.unescape(value)) not in attrs:
                continue
            if _approved_core_ui_href(value, self.relative_page):
                start, end = match.span(value_group)
                self.allowed_spans.append((offset + start, offset + end))


def _policy_html(html: str, relative_page: str, surface: str) -> str:
    if surface != "full":
        return html
    policy = CoreUILinkPolicy(html, relative_page)
    policy.feed(html)
    for start, end in reversed(policy.allowed_spans):
        html = html[:start] + "__approved_core_ui_link__" + html[end:]
    return html


class TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.skip = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in {"script", "style", "pre", "code", "svg"}:
            self.skip += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "pre", "code", "svg"} and self.skip:
            self.skip -= 1

    def handle_data(self, data: str) -> None:
        if not self.skip:
            self.parts.append(data)


def _visible_text(html: str) -> str:
    parser = TextExtractor()
    parser.feed(html)
    return " ".join(parser.parts)


def _is_korean_page(path: Path, site_dir: Path) -> bool:
    relative = path.relative_to(site_dir).as_posix()
    if relative.startswith("en/"):
        return False
    if relative.startswith(("assets/", "search/")):
        return False
    return path.name == "index.html" or relative in {"404.html"}


def _hangul_count(text: str) -> int:
    return len(re.findall(r"[가-힣]", text))


def _english_word_count(text: str) -> int:
    return len(re.findall(r"\b[A-Za-z][A-Za-z'-]{3,}\b", text))


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify Korean MkDocs pages contain Korean prose.")
    parser.add_argument("--site-dir", type=Path, default=Path("site"))
    parser.add_argument("--surface", choices=("public", "full"), default="public")
    args = parser.parse_args()
    site_dir = args.site_dir.resolve()
    failures: list[str] = []

    if not site_dir.is_dir():
        print("Korean docs verification failed: site directory is missing.")
        return 1
    pages = [path for path in sorted(site_dir.rglob("*.html")) if _is_korean_page(path, site_dir)]
    if not pages:
        print("Korean docs verification failed: no Korean HTML pages were found.")
        return 1

    for html_path in pages:
        html = html_path.read_text(encoding="utf-8")
        text = _visible_text(html)
        policy_html = _policy_html(html, html_path.relative_to(site_dir).as_posix(), args.surface)
        for phrase in DISALLOWED:
            if phrase in policy_html:
                failures.append(f"{html_path.relative_to(site_dir)} contains disallowed phrase: {phrase}")
        hangul = _hangul_count(text)
        english = _english_word_count(text)
        if html_path.name != "404.html" and hangul < 120:
            failures.append(f"{html_path.relative_to(site_dir)} has too little Korean prose: {hangul}")
        if html_path.name != "404.html" and english > max(120, hangul * 0.9):
            failures.append(
                f"{html_path.relative_to(site_dir)} looks English-heavy: english={english}, hangul={hangul}"
            )

    if failures:
        print("Korean docs verification failed:")
        for failure in failures[:80]:
            print(f" - {failure}")
        if len(failures) > 80:
            print(f" - ... {len(failures) - 80} more")
        return 1
    print("Verified Korean docs contain Korean prose and no removed surfaces.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
