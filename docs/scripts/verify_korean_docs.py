from __future__ import annotations

import argparse
import re
import sys
from html.parser import HTMLParser
from pathlib import Path


DISALLOWED = (
    "한국어 미러",
    "Truthound Depot",
    "Data Docs Dashboard",
    "truthound-dashboard",
    "/dashboard/",
    "/depot/",
)


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
    args = parser.parse_args()
    site_dir = args.site_dir.resolve()
    failures: list[str] = []

    for html_path in sorted(site_dir.rglob("*.html")):
        if not _is_korean_page(html_path, site_dir):
            continue
        html = html_path.read_text(encoding="utf-8")
        text = _visible_text(html)
        for phrase in DISALLOWED:
            if phrase in html:
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
