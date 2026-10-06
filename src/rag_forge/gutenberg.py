"""Clean Project Gutenberg texts before chunking.

Illustrated editions often ship a preface, title pages, and a list of
illustrations *after* the legal START marker. Those pages mention the
book title and themes ("pride", "opening") and drown the actual first
chapter in retrieval.
"""

from __future__ import annotations

import re

_START_MARKERS = (
    "*** START OF THIS PROJECT GUTENBERG",
    "*** START OF THE PROJECT GUTENBERG",
    "*END*THE SMALL PRINT",
)
_END_MARKERS = (
    "*** END OF THIS PROJECT GUTENBERG",
    "*** END OF THE PROJECT GUTENBERG",
    "End of Project Gutenberg",
    "End of the Project Gutenberg",
)

# Captions are short; cap the span so a missing ] cannot swallow the book.
_ILLUSTRATION_RE = re.compile(r"\[Illustration(?::[^\]]{0,2000})?\]", re.IGNORECASE)

# First chapter / letter of the work, not later chapters.
_BODY_HEADING_RE = re.compile(r"(?im)(?:^|(?<=\n))[ \t]*(?:chapter|letter)\s+(?:i|1|one)\.?\b")
_FOLLOW_ON_HEADING_RE = re.compile(r"(?i)\b(?:chapter|letter)\s+[ivxlc0-9]+\b")
# Illustration-index rows pad the page number with several spaces.
# Do not treat "Letter 1\\r" as a page number: \\r is regex \\s.
_PAGE_NUMBER_ON_LINE_RE = re.compile(r"[ \t]{2,}\d{1,4}(?:[ \t]+|$)")


def strip_gutenberg_wrappers(text: str) -> str:
    """Remove the legal Gutenberg header and footer, if present."""
    for marker in _START_MARKERS:
        idx = text.find(marker)
        if idx != -1:
            newline_idx = text.find("\n", idx)
            if newline_idx != -1:
                text = text[newline_idx + 1 :]
            break

    for marker in _END_MARKERS:
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx]
            break

    return text


def strip_illustrations(text: str) -> str:
    """Drop [Illustration] / [Illustration: caption] blocks."""
    return _ILLUSTRATION_RE.sub("", text)


def _heading_line(text: str, match: re.Match[str]) -> str:
    end = text.find("\n", match.start())
    if end == -1:
        end = len(text)
    return text[match.start() : end]


def _is_toc_or_index_heading(text: str, match: re.Match[str]) -> bool:
    """True for contents / illustration-index rows, not the chapter itself."""
    line = _heading_line(text, match)
    if _PAGE_NUMBER_ON_LINE_RE.search(line):
        return True

    following = text[match.end() : match.end() + 400]
    return len(_FOLLOW_ON_HEADING_RE.findall(following)) >= 3


def _followed_by_prose(text: str, match: re.Match[str]) -> bool:
    after = text[match.end() : match.end() + 600]
    after = re.sub(r"^[\].:\-\s]+", "", after)
    after = _ILLUSTRATION_RE.sub(" ", after)
    words = re.findall(r"[A-Za-z]{3,}", after)
    return len(words) >= 25


def find_body_start(text: str) -> int:
    """Index of the first real Chapter I / Letter 1, or 0 if none."""
    for match in _BODY_HEADING_RE.finditer(text):
        if _is_toc_or_index_heading(text, match):
            continue
        if _followed_by_prose(text, match):
            return match.start()
    return 0


def clean_gutenberg_text(text: str) -> str:
    """Strip wrappers, skip front matter, and drop illustration markup.

    Safe to run more than once. If no chapter/letter heading is found the
    text is left intact aside from wrappers and illustration tags.
    """
    if not text or not text.strip():
        return text

    text = strip_gutenberg_wrappers(text)
    start = find_body_start(text)
    if start:
        text = text[start:]
        # "[Illustration: ... Chapter I.]" leaves a dangling ] on the heading.
        text = re.sub(
            r"(?im)^[ \t]*((?:chapter|letter)\s+(?:i|1|one)\.?)\s*\]",
            r"\1",
            text,
            count=1,
        )
    text = strip_illustrations(text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
