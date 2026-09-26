"""Source-cited retrieval agent over the document-scoped Qdrant collection."""

from app.domain.analysis import RetrievedPassage
from app.services.embeddings import EmbeddingProvider
from app.services.vector_store import QdrantDocumentStore


class RetrievalAgent:
    """Embed a query and retrieve the highest-scoring chunks from one document."""

    def __init__(self, embedder: EmbeddingProvider, store: QdrantDocumentStore, limit: int) -> None:
        self._embedder = embedder
        self._store = store
        self._limit = limit

    async def retrieve(self, document_id: str, query: str) -> list[RetrievedPassage]:
        """Return document-filtered passages with their original source citations."""

        vector = await self._embedder.embed_query(query)
        return await self._store.search(document_id, vector, self._limit)

    async def close(self) -> None:
        """Release the short-lived Qdrant client for one query request."""

        await self._store.close()
