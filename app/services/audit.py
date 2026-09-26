"""SQLAlchemy-backed audit trail for FinDocs query processing."""

import json
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import DateTime, Integer, JSON, String, Text, create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.engine.url import make_url
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from app.domain.audit import QueryAuditEvent


class Base(DeclarativeBase):
    """Base class for relational audit-log models."""


class QueryAuditLog(Base):
    """Append-only audit record for one `/api/v1/query` invocation."""

    __tablename__ = "query_audit_logs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    document_id: Mapped[str] = mapped_column(String(128), index=True, nullable=False)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    route: Mapped[str | None] = mapped_column(String(32), nullable=True)
    routing_path: Mapped[list[dict[str, object]]] = mapped_column(JSON, nullable=False)
    prompt_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    final_output: Mapped[dict[str, object] | None] = mapped_column(JSON, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)


class AuditRepository:
    """Initialize and append query audit events using short-lived transactions."""

    def __init__(self, database_url: str) -> None:
        url = make_url(database_url)
        if url.drivername.startswith("sqlite") and url.database and url.database != ":memory:":
            Path(url.database).parent.mkdir(parents=True, exist_ok=True)
        connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
        self._engine: Engine = create_engine(database_url, pool_pre_ping=True, connect_args=connect_args)
        self._session_factory = sessionmaker(bind=self._engine, expire_on_commit=False)

    @property
    def session_factory(self) -> sessionmaker:
        """Expose the configured factory to repositories sharing this database."""

        return self._session_factory

    def initialize(self) -> None:
        """Create the development audit table; production should apply controlled migrations."""

        Base.metadata.create_all(self._engine)

    def check_connection(self) -> None:
        """Run a minimal database query for readiness checks."""

        with self._engine.connect() as connection:
            connection.execute(text("SELECT 1"))

    def record(self, event: QueryAuditEvent) -> None:
        """Persist an immutable event in a committed transaction."""

        routing_path = [step.model_dump(mode="json") for step in event.steps]
        final_output = event.final_output.model_dump(mode="json") if event.final_output else None
        record = QueryAuditLog(
            id=event.request_id,
            document_id=event.document_id,
            query=event.query,
            route=event.route,
            routing_path=json.loads(json.dumps(routing_path)),
            prompt_tokens=event.prompt_tokens,
            completion_tokens=event.completion_tokens,
            total_tokens=event.total_tokens,
            latency_ms=event.latency_ms,
            status=event.status,
            final_output=json.loads(json.dumps(final_output)) if final_output is not None else None,
            error_message=event.error_message,
            created_at=event.created_at or datetime.now(UTC),
        )
        with self._session_factory.begin() as session:
            session.add(record)

    def get(self, request_id: str) -> QueryAuditEvent | None:
        """Return one audit event for operational inspection and test verification."""

        with self._session_factory() as session:
            record = session.get(QueryAuditLog, request_id)
            if record is None:
                return None
            response = None
            if record.final_output:
                from app.domain.analysis import QueryResponse

                response = QueryResponse.model_validate(record.final_output)
            return QueryAuditEvent(
                request_id=record.id,
                document_id=record.document_id,
                query=record.query,
                route=record.route,
                steps=record.routing_path,
                prompt_tokens=record.prompt_tokens,
                completion_tokens=record.completion_tokens,
                total_tokens=record.total_tokens,
                latency_ms=record.latency_ms,
                status=record.status,
                final_output=response,
                error_message=record.error_message,
                created_at=record.created_at,
            )

    def close(self) -> None:
        """Dispose pooled database connections during application shutdown."""

        self._engine.dispose()
