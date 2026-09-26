"""Semantic chunking tuned for financial narratives, covenants, and tables."""

import re
from collections.abc import Iterable

from app.domain.documents import ContentKind, DocumentChunk, SourceElement


class FinancialSemanticChunker:
    """Chunk financial content without splitting table rows or legal clauses.

    Narrative content is packed by paragraph and sentence, retaining a small
    sentence overlap for cross-reference context. Markdown tables are split only
    between data rows and repeat their headers in every resulting chunk.
    """

    def __init__(self, max_characters: int = 1_800, overlap_sentences: int = 1) -> None:
        if max_characters < 256:
            raise ValueError("max_characters must be at least 256")
        if overlap_sentences < 0:
            raise ValueError("overlap_sentences cannot be negative")
        self._max_characters = max_characters
        self._overlap_sentences = overlap_sentences

    def chunk(self, document_id: str, elements: Iterable[SourceElement]) -> list[DocumentChunk]:
        """Convert ordered parser elements into citable, embedding-ready chunks."""

        chunks: list[DocumentChunk] = []
        for element in elements:
            texts = self._split_table(element.text) if element.kind is ContentKind.TABLE else self._split_narrative(element.text)
            for text in texts:
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{document_id}:{len(chunks):06d}",
                        document_id=document_id,
                        text=text,
                        kind=element.kind,
                        ordinal=len(chunks),
                        page_number=element.page_number,
                        section=element.section,
                        metadata=element.metadata,
                    )
                )
        return chunks

    def _split_narrative(self, text: str) -> list[str]:
        paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", text) if paragraph.strip()]
        chunks: list[str] = []
        current = ""
        for paragraph in paragraphs:
            for unit in self._sentence_units(paragraph):
                candidate = f"{current}\n\n{unit}".strip() if current else unit
                if current and len(candidate) > self._max_characters:
                    chunks.append(current)
                    overlap = self._tail_sentences(current)
                    current = f"{overlap} {unit}".strip()
                elif not current and len(unit) > self._max_characters:
                    chunks.extend(self._hard_split(unit))
                    current = ""
                else:
                    current = candidate
        if current:
            chunks.append(current)
        return chunks

    def _split_table(self, table: str) -> list[str]:
        rows = [line.strip() for line in table.splitlines() if line.strip()]
        if len(rows) < 3:
            return self._hard_split(table)
        header = "\n".join(rows[:2])
        chunks: list[str] = []
        current_rows: list[str] = []
        for row in rows[2:]:
            candidate = "\n".join([header, *current_rows, row])
            if current_rows and len(candidate) > self._max_characters:
                chunks.append("\n".join([header, *current_rows]))
                current_rows = [row]
            elif not current_rows and len(candidate) > self._max_characters:
                chunks.extend(self._hard_split(candidate))
            else:
                current_rows.append(row)
        if current_rows:
            chunks.append("\n".join([header, *current_rows]))
        return chunks or [table]

    @staticmethod
    def _sentence_units(paragraph: str) -> list[str]:
        units = re.split(r"(?<=[.!?;:])\s+(?=[A-Z0-9(])", paragraph)
        return [unit.strip() for unit in units if unit.strip()]

    def _tail_sentences(self, text: str) -> str:
        if not self._overlap_sentences:
            return ""
        sentences = self._sentence_units(text)
        return " ".join(sentences[-self._overlap_sentences :])

    def _hard_split(self, text: str) -> list[str]:
        return [text[index : index + self._max_characters] for index in range(0, len(text), self._max_characters)]
