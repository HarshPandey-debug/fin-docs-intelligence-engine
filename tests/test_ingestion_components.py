"""Unit tests for local, deterministic Phase 2 ingestion components."""

import asyncio
from pathlib import Path

from app.domain.documents import ContentKind, SourceElement
from app.services.chunking import FinancialSemanticChunker
from app.services.ingestion import DocumentIngestionService
from app.services.parsing import FinancialDocumentParser


def test_csv_parser_preserves_headers_and_rows(tmp_path: Path) -> None:
    """CSV input becomes one structured Markdown table with exact values."""

    source = tmp_path / "ratios.csv"
    source.write_text("Metric,FY2025\nDebt,1200\nEquity,800\n", encoding="utf-8")

    elements = asyncio.run(FinancialDocumentParser(None).parse(source))

    assert len(elements) == 1
    assert elements[0].kind is ContentKind.TABLE
    assert "| Metric | FY2025 |" in elements[0].text
    assert "| Debt | 1200 |" in elements[0].text


def test_chunker_never_drops_table_headers() -> None:
    """Every table chunk repeats headers, preserving standalone financial meaning."""

    table = "\n".join(
        ["| Metric | Amount |", "| --- | --- |"]
        + [f"| Covenant {index} | {index * 100} |" for index in range(40)]
    )
    chunks = FinancialSemanticChunker(max_characters=256).chunk(
        "document-1", [SourceElement(text=table, kind=ContentKind.TABLE)]
    )

    assert len(chunks) > 1
    assert all(chunk.text.startswith("| Metric | Amount |\n| --- | --- |") for chunk in chunks)
    assert [chunk.ordinal for chunk in chunks] == list(range(len(chunks)))


def test_ingestion_orchestrates_parse_embed_and_upsert(tmp_path: Path) -> None:
    """The service coordinates collaborators and reports a deterministic result."""

    class Parser:
        async def parse(self, _: Path) -> list[SourceElement]:
            return [SourceElement(text="Liquidity remains above the required covenant threshold.")]

    class Embedder:
        async def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [[0.1, 0.2] for _ in texts]

    class Store:
        initialized = False
        received_chunks = 0

        async def ensure_collection(self) -> None:
            self.initialized = True

        async def upsert(self, chunks: list[object], vectors: list[list[float]], _: str) -> int:
            assert self.initialized
            assert len(chunks) == len(vectors)
            self.received_chunks = len(chunks)
            return self.received_chunks

        async def close(self) -> None:
            return None

    store = Store()
    service = DocumentIngestionService(Parser(), FinancialSemanticChunker(), Embedder(), store)  # type: ignore[arg-type]
    result = asyncio.run(service.ingest("doc-123", tmp_path / "unused.csv", "unused.csv"))

    assert result.document_id == "doc-123"
    assert result.chunk_count == 1
    assert result.vector_count == 1
