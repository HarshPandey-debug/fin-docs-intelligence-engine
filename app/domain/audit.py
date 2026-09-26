"""Contracts for immutable financial-analysis audit events."""

from datetime import datetime

from pydantic import BaseModel, Field

from app.domain.analysis import AgentStep, QueryResponse


class QueryAuditEvent(BaseModel):
    """Durable evidence of one document-analysis request and its outcome."""

    request_id: str
    document_id: str
    query: str
    route: str | None = None
    steps: list[AgentStep] = Field(default_factory=list)
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    latency_ms: int = Field(ge=0)
    status: str
    final_output: QueryResponse | None = None
    error_message: str | None = None
    created_at: datetime | None = None
