from __future__ import annotations

import re
from dataclasses import dataclass


_TERMINATOR = re.compile(r"[。！？!?]|\.(?!\d)")
_TRAILING_CLOSERS = set('"\'”’)]}')
_ABBREVIATIONS = {
    "e.g.",
    "i.e.",
    "et al.",
    "vs.",
    "dr.",
    "mr.",
    "mrs.",
    "prof.",
    "fig.",
    "no.",
}


@dataclass(frozen=True)
class TextSegment:
    text: str
    start: int
    end: int


def _is_abbreviation(text: str, period_index: int) -> bool:
    prefix = text[max(0, period_index - 12) : period_index + 1].lower()
    return any(prefix.endswith(item) for item in _ABBREVIATIONS)


def segment_abstract(text: str) -> list[TextSegment]:
    """Conservatively split an abstract while retaining exact offsets."""

    if not text or not text.strip():
        return []

    boundaries: list[int] = []
    for match in _TERMINATOR.finditer(text):
        if match.group() == "." and _is_abbreviation(text, match.start()):
            continue
        end = match.end()
        while end < len(text) and text[end] in _TRAILING_CLOSERS:
            end += 1
        next_index = end
        while next_index < len(text) and text[next_index].isspace():
            next_index += 1
        if next_index >= len(text):
            boundaries.append(end)
            continue
        next_character = text[next_index]
        if next_character.isupper() or next_character.isdigit() or next_character in "<&[":
            boundaries.append(end)

    segments: list[TextSegment] = []
    cursor = 0
    for boundary in boundaries + [len(text)]:
        raw_start = cursor
        raw_end = boundary
        while raw_start < raw_end and text[raw_start].isspace():
            raw_start += 1
        while raw_end > raw_start and text[raw_end - 1].isspace():
            raw_end -= 1
        if raw_end > raw_start:
            segments.append(TextSegment(text=text[raw_start:raw_end], start=raw_start, end=raw_end))
        cursor = boundary

    if not segments:
        stripped_start = len(text) - len(text.lstrip())
        stripped_end = len(text.rstrip())
        return [TextSegment(text=text[stripped_start:stripped_end], start=stripped_start, end=stripped_end)]
    return segments
