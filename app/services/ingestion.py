"""End-to-end orchestration for document parsing, embedding, and vector storage."""

from pathlib import Path

from app.domain.documents import IngestionResult
from app.services.chunking import FinancialSemanticChunker
from app.services.embeddings import EmbeddingProvider
from app.services.parsing import FinancialDocumentParser
from app.services.vector_store import QdrantDocumentStore


class DocumentIngestionService:
    """Execute the durable stages of a financial document ingestion workflow."""

    def __init__(
        self,
        parser: FinancialDocumentParser,
        chunker: FinancialSemanticChunker,
        embedder: EmbeddingProvider,
        vector_store: QdrantDocumentStore,
    ) -> None:
        self._parser = parser
        self._chunker = chunker
        self._embedder = embedder
        self._vector_store = vector_store

    async def ingest(self, document_id: str, source_path: Path, source_filename: str) -> IngestionResult:
        """Parse, semantically chunk, embed, and persist one financial document."""

        elements = await self._parser.parse(source_path)
        chunks = self._chunker.chunk(document_id, elements)
        if not chunks:
            raise ValueError("Document did not produce any retrievable chunks")
        vectors = await self._embedder.embed_documents([chunk.text for chunk in chunks])
        await self._vector_store.ensure_collection()
        vector_count = await self._vector_store.upsert(chunks, vectors, source_filename)
        return IngestionResult(document_id=document_id, chunk_count=len(chunks), vector_count=vector_count)

    async def close(self) -> None:
        """Release vector-store resources after a background ingestion task."""

        await self._vector_store.close()
