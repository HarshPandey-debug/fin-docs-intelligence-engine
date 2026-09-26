"""Typed models for parsed documents and vector-ready chunks."""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ContentKind(StrEnum):
    """Kinds of source content that require different chunk boundaries."""

    TEXT = "text"
    TABLE = "table"


class SourceElement(BaseModel):
    """A location-aware unit emitted by a document parser."""

    text: str = Field(min_length=1)
    kind: ContentKind = ContentKind.TEXT
    page_number: int | None = Field(default=None, ge=1)
    section: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentChunk(BaseModel):
    """A semantically coherent, citable portion of a source document."""

    chunk_id: str
    document_id: str
    text: str = Field(min_length=1)
    kind: ContentKind
    ordinal: int = Field(ge=0)
    page_number: int | None = Field(default=None, ge=1)
    section: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestionResult(BaseModel):
    """Persisted outcome of one document-ingestion run."""

    document_id: str
    chunk_count: int = Field(ge=0)
    vector_count: int = Field(ge=0)
