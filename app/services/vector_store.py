"""Qdrant persistence for source-citable financial document chunks."""

from uuid import NAMESPACE_URL, uuid5

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams

from app.domain.analysis import Citation, RetrievedPassage
from app.domain.documents import DocumentChunk


class VectorStoreError(RuntimeError):
    """Raised when Qdrant collection management or upsert operations fail."""


class QdrantDocumentStore:
    """Create and populate the collection used by the retrieval agent."""

    def __init__(self, url: str, collection_name: str, vector_size: int) -> None:
        self._client = AsyncQdrantClient(url=url)
        self._collection_name = collection_name
        self._vector_size = vector_size

    async def ensure_collection(self) -> None:
        """Create a cosine-distance collection and document filter index if absent."""

        try:
            if not await self._client.collection_exists(self._collection_name):
                await self._client.create_collection(
                    collection_name=self._collection_name,
                    vectors_config=VectorParams(size=self._vector_size, distance=Distance.COSINE),
                )
                await self._client.create_payload_index(
                    collection_name=self._collection_name,
                    field_name="document_id",
                    field_schema="keyword",
                )
        except Exception as exc:
            raise VectorStoreError(f"Unable to initialize Qdrant collection: {exc}") from exc

    async def upsert(self, chunks: list[DocumentChunk], vectors: list[list[float]], source_filename: str) -> int:
        """Upsert chunks and their vectors with citation metadata into Qdrant."""

        if len(chunks) != len(vectors):
            raise VectorStoreError("Chunk and embedding counts must match")
        if any(len(vector) != self._vector_size for vector in vectors):
            raise VectorStoreError("Embedding dimensions do not match Qdrant collection dimensions")
        points = [
            PointStruct(
                id=str(uuid5(NAMESPACE_URL, chunk.chunk_id)),
                vector=vector,
                payload={
                    "document_id": chunk.document_id,
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "kind": chunk.kind.value,
                    "ordinal": chunk.ordinal,
                    "page_number": chunk.page_number,
                    "section": chunk.section,
                    "source_filename": source_filename,
                    **chunk.metadata,
                },
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        try:
            await self._client.upsert(collection_name=self._collection_name, points=points, wait=True)
        except Exception as exc:
            raise VectorStoreError(f"Unable to upsert chunks to Qdrant: {exc}") from exc
        return len(points)

    async def search(self, document_id: str, query_vector: list[float], limit: int) -> list[RetrievedPassage]:
        """Search only the requested document and return source-citable passages."""

        if len(query_vector) != self._vector_size:
            raise VectorStoreError("Query embedding dimension does not match Qdrant collection dimensions")
        try:
            results = await self._client.search(
                collection_name=self._collection_name,
                query_vector=query_vector,
                query_filter=Filter(
                    must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
                ),
                limit=limit,
                with_payload=True,
            )
        except Exception as exc:
            raise VectorStoreError(f"Unable to search Qdrant: {exc}") from exc

        passages: list[RetrievedPassage] = []
        for point in results:
            payload = point.payload or {}
            text = str(payload.get("text", "")).strip()
            chunk_id = str(payload.get("chunk_id", "")).strip()
            source_filename = str(payload.get("source_filename", "")).strip()
            if not text or not chunk_id or not source_filename:
                continue
            passages.append(
                RetrievedPassage(
                    citation=Citation(
                        document_id=document_id,
                        chunk_id=chunk_id,
                        source_filename=source_filename,
                        page_number=payload.get("page_number"),
                        section=payload.get("section"),
                        excerpt=text[:1_000],
                    ),
                    score=point.score,
                    text=text,
                    kind=str(payload.get("kind", "text")),
                )
            )
        return passages

    async def close(self) -> None:
        """Close the underlying asynchronous Qdrant client."""

        await self._client.close()
