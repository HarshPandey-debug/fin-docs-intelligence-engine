"""Phase 1 smoke tests."""

from fastapi.testclient import TestClient

from app.main import app
from app.services.audit import AuditRepository
from app.services.documents import DocumentRepository, IngestionStatus
from app.core.security import require_api_key
from app.core.config import Settings


def test_health_check() -> None:
    """The API exposes a liveness endpoint."""

    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_metrics_endpoint() -> None:
    """Prometheus can scrape metric exposition data."""

    with TestClient(app) as client:
        client.get("/health")
        response = client.get("/api/v1/metrics")
    assert response.status_code == 200
    assert "findocs_http_requests_total" in response.text


def test_document_status_survives_repository_recreation(tmp_path) -> None:
    """Document status is stored in SQL rather than process memory."""

    audit = AuditRepository(f"sqlite:///{tmp_path / 'documents.db'}")
    audit.initialize()
    first = DocumentRepository(audit.session_factory)
    first.create("doc-1", "filing.csv")
    first.mark_completed("doc-1", 3)

    second = DocumentRepository(audit.session_factory)
    record = second.get("doc-1")

    assert record is not None
    assert record.status is IngestionStatus.COMPLETED
    assert record.chunk_count == 3
    audit.close()


def test_production_requires_api_key() -> None:
    """Production authentication rejects missing and invalid credentials."""

    settings = Settings(environment="production", app_api_key="expected")

    try:
        require_api_key(None, settings)
    except Exception as exc:
        assert getattr(exc, "status_code", None) == 401
    else:
        raise AssertionError("Expected missing API key to be rejected")

    try:
        require_api_key("wrong", settings)
    except Exception as exc:
        assert getattr(exc, "status_code", None) == 401
    else:
        raise AssertionError("Expected invalid API key to be rejected")
