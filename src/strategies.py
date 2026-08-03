"""Domain-aware chunking strategies used in the K4 retrieval comparison."""

from __future__ import annotations

import re

from .chunking import RecursiveChunker


class MarkdownHeadingChunker:
    """Split Markdown by headings and retain the heading in every child chunk.

    TikTok Shop policies are long, hierarchical Markdown documents. Keeping a
    section title next to its text adds useful retrieval terms and avoids joining
    unrelated policy clauses merely because they are adjacent in the file.
    Oversized sections fall back to the recursive strategy.
    """

    HEADING_PATTERN = re.compile(r"(?m)^(#{1,6})\s+(.+?)\s*$")

    def __init__(self, chunk_size: int = 700) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        self.chunk_size = chunk_size
        self._fallback = RecursiveChunker(chunk_size=chunk_size)

    def chunk(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []

        matches = list(self.HEADING_PATTERN.finditer(text))
        if not matches:
            return self._fallback.chunk(text)

        chunks: list[str] = []
        preamble = text[: matches[0].start()].strip()
        if preamble:
            chunks.extend(self._fallback.chunk(preamble))

        heading_stack: list[tuple[int, str]] = []
        for index, match in enumerate(matches):
            section_end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            level = len(match.group(1))
            title = match.group(2).strip()
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
            # Include the full parent path. For example, a "Tài liệu bắt buộc"
            # subsection remains identifiable as belonging to "Thực phẩm bổ sung".
            heading = "\n".join(f"{'#' * item_level} {item_title}" for item_level, item_title in heading_stack)
            body = text[match.end() : section_end].strip()
            section = f"{heading}\n\n{body}" if body else heading
            if len(section) <= self.chunk_size:
                chunks.append(section)
                continue

            # Prefixing the heading to each fragment preserves the section's
            # subject when a long policy section must be divided further.
            available = max(1, self.chunk_size - len(heading) - 2)
            for fragment in RecursiveChunker(chunk_size=available).chunk(body):
                chunks.append(f"{heading}\n\n{fragment}")

        return [chunk for chunk in chunks if chunk.strip()]
