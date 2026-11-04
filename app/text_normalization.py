from __future__ import annotations

import html
import re


_HTML_TAG = re.compile(r"<[^>]+>")
_WHITESPACE = re.compile(r"\s+")


def normalize_for_prompt(text: str) -> str:
    """Clean display text without changing the stored verbatim evidence."""

    decoded = html.unescape(text or "")
    without_tags = _HTML_TAG.sub(" ", decoded)
    return _WHITESPACE.sub(" ", without_tags).strip()
