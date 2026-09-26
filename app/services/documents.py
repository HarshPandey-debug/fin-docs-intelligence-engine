"""Durable document-ingestion status contracts and repository."""

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict
from sqlalchemy import DateTime, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.services.audit import Base


class IngestionStatus(StrEnum):
    """Lifecycle states visible for an uploaded document."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DocumentIngestionRecord(BaseModel):
    """Status of a document's current ingestion run."""

    model_config = ConfigDict(from_attributes=True)

    document_id: str
    filename: str
    status: IngestionStatus
    updated_at: datetime
    chunk_count: int | None = None
    error: str | None = None


class DocumentStatusLog(Base):
    """Durable status record for one uploaded document."""

    __tablename__ = "document_status"

    document_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    chunk_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)


class DocumentRepository:
    """Persist document ingestion state using the application's database."""

    def __init__(self, session_factory: object) -> None:
        self._session_factory = session_factory

    def create(self, document_id: str, filename: str) -> None:
        """Register a newly accepted upload as pending."""

        with self._session_factory.begin() as session:
            session.add(DocumentStatusLog(document_id=document_id, filename=filename, status=IngestionStatus.PENDING.value, updated_at=datetime.now(UTC)))

    def mark_processing(self, document_id: str) -> None:
        """Mark an accepted document as being processed."""

        self._update(document_id, status=IngestionStatus.PROCESSING.value)

    def mark_completed(self, document_id: str, chunk_count: int) -> None:
        """Mark successful vectorization and retain its chunk count."""

        self._update(document_id, status=IngestionStatus.COMPLETED.value, chunk_count=chunk_count, error=None)

    def mark_failed(self, document_id: str, error: str) -> None:
        """Mark a recoverable failure without exposing internal stack traces."""

        self._update(document_id, status=IngestionStatus.FAILED.value, error=error[:500])

    def get(self, document_id: str) -> DocumentIngestionRecord | None:
        """Return the current process-local ingestion record for one document."""

        with self._session_factory() as session:
            record = session.get(DocumentStatusLog, document_id)
            return DocumentIngestionRecord.model_validate(record, from_attributes=True) if record else None

    def _update(self, document_id: str, **changes: object) -> None:
        with self._session_factory.begin() as session:
            record = session.get(DocumentStatusLog, document_id)
            if record is None:
                return
            for field, value in changes.items():
                setattr(record, field, value)
            record.updated_at = datetime.now(UTC)


