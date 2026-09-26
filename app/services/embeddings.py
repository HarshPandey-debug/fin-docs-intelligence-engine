"""Embedding provider abstractions for financial-document retrieval."""

import asyncio
from typing import Protocol


class EmbeddingError(RuntimeError):
    """Raised when an embedding provider cannot return usable vectors."""


class EmbeddingProvider(Protocol):
    """Interface implemented by document and query embedding providers."""

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed text chunks for vector retrieval."""

    async def embed_query(self, text: str) -> list[float]:
        """Embed a user query for similarity search."""


class GeminiEmbeddingProvider:
    """Generate retrieval-document embeddings with Gemini's embedding API."""

    def __init__(self, api_key: str | None, model: str, dimensions: int) -> None:
        self._api_key = api_key
        self._model = model
        self._dimensions = dimensions

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed all chunks in one provider request and validate vector dimensions."""

        if not texts:
            return []
        return await asyncio.to_thread(self._embed_sync, texts, "RETRIEVAL_DOCUMENT")

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single user query using retrieval-query task semantics."""

        if not text.strip():
            raise EmbeddingError("Query text cannot be empty")
        return (await asyncio.to_thread(self._embed_sync, [text], "RETRIEVAL_QUERY"))[0]

    def _embed_sync(self, texts: list[str], task_type: str) -> list[list[float]]:
        if not self._api_key:
            raise EmbeddingError("GEMINI_API_KEY is required for document embeddings")
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=self._api_key)
            response = client.models.embed_content(
                model=self._model,
                contents=texts,
                config=types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=self._dimensions,
                ),
            )
            vectors = [list(embedding.values) for embedding in response.embeddings]
        except Exception as exc:
            raise EmbeddingError(f"Gemini embedding request failed: {exc}") from exc

        if len(vectors) != len(texts):
            raise EmbeddingError("Embedding provider returned a vector count that does not match inputs")
        if any(len(vector) != self._dimensions for vector in vectors):
            raise EmbeddingError("Embedding provider returned a vector with an unexpected dimension")
        return vectors
