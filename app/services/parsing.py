"""LlamaParse-backed parsers that preserve financial tables as Markdown."""

import csv
from collections.abc import Sequence
from pathlib import Path

from app.domain.documents import ContentKind, SourceElement


class DocumentParsingError(RuntimeError):
    """Raised when a supported document cannot be parsed into source elements."""


class FinancialDocumentParser:
    """Parse PDF and CSV files into source elements with table fidelity.

    PDFs are parsed by LlamaParse in Markdown mode, which retains detected tables
    and headings. CSV files are converted to a Markdown table locally so they
    retain exact row/column values without an external parsing round trip.
    """

    def __init__(self, llama_cloud_api_key: str | None) -> None:
        self._llama_cloud_api_key = llama_cloud_api_key

    async def parse(self, source_path: Path) -> list[SourceElement]:
        """Parse a supported source file into ordered, source-addressable elements."""

        suffix = source_path.suffix.lower()
        if suffix == ".csv":
            return self._parse_csv(source_path)
        if suffix == ".pdf":
            return await self._parse_pdf(source_path)
        raise DocumentParsingError(f"Unsupported document type: {suffix or 'unknown'}")

    def _parse_csv(self, source_path: Path) -> list[SourceElement]:
        try:
            with source_path.open("r", encoding="utf-8-sig", newline="") as source_file:
                rows = list(csv.reader(source_file))
        except (OSError, UnicodeDecodeError, csv.Error) as exc:
            raise DocumentParsingError(f"Unable to parse CSV: {exc}") from exc

        if not rows or not any(any(cell.strip() for cell in row) for row in rows):
            raise DocumentParsingError("CSV contains no tabular data")

        width = max(len(row) for row in rows)
        normalized_rows = [row + [""] * (width - len(row)) for row in rows]
        header = normalized_rows[0]
        divider = ["---"] * width
        markdown_rows = [header, divider, *normalized_rows[1:]]
        table = "\n".join("| " + " | ".join(self._escape_cell(cell) for cell in row) + " |" for row in markdown_rows)
        return [
            SourceElement(
                text=table,
                kind=ContentKind.TABLE,
                section="CSV table",
                metadata={"row_count": max(len(rows) - 1, 0), "column_count": width},
            )
        ]

    async def _parse_pdf(self, source_path: Path) -> list[SourceElement]:
        if not self._llama_cloud_api_key:
            raise DocumentParsingError("LLAMA_CLOUD_API_KEY is required to parse PDF documents")
        try:
            from llama_parse import LlamaParse

            parser = LlamaParse(
                api_key=self._llama_cloud_api_key,
                result_type="markdown",
                verbose=False,
            )
            documents = await parser.aload_data(str(source_path))
        except Exception as exc:
            raise DocumentParsingError(f"LlamaParse failed to parse PDF: {exc}") from exc

        elements: list[SourceElement] = []
        for page_number, document in enumerate(documents, start=1):
            text = str(document.text).strip()
            if text:
                elements.extend(self._elements_from_markdown(text, page_number))
        if not elements:
            raise DocumentParsingError("PDF parser returned no extractable text or tables")
        return elements

    @staticmethod
    def _escape_cell(value: str) -> str:
        return value.replace("|", "\\|").replace("\n", " ").strip()

    @staticmethod
    def _elements_from_markdown(markdown: str, page_number: int) -> Sequence[SourceElement]:
        elements: list[SourceElement] = []
        current_section: str | None = None
        text_lines: list[str] = []
        table_lines: list[str] = []

        def flush_text() -> None:
            if text := "\n".join(text_lines).strip():
                elements.append(SourceElement(text=text, page_number=page_number, section=current_section))
            text_lines.clear()

        def flush_table() -> None:
            if table := "\n".join(table_lines).strip():
                elements.append(
                    SourceElement(text=table, kind=ContentKind.TABLE, page_number=page_number, section=current_section)
                )
            table_lines.clear()

        for line in markdown.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                flush_text()
                flush_table()
                current_section = stripped.lstrip("#").strip() or current_section
            elif stripped.startswith("|") and stripped.endswith("|"):
                flush_text()
                table_lines.append(stripped)
            else:
                flush_table()
                text_lines.append(line)
        flush_text()
        flush_table()
        return elements
