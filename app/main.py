"""FastAPI application entry point for FinDocs."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from qdrant_client import AsyncQdrantClient

from app.api.v1.router import router as v1_router
from app.core.config import get_settings
from app.core.audit_context import AuditContextMiddleware
from app.core.metrics import PrometheusMiddleware
from app.core.rate_limit import RateLimitMiddleware
from app.core.security import ApiKeyMiddleware
from app.services.audit import AuditRepository
from app.services.documents import DocumentRepository

settings = get_settings()
audit_repository = AuditRepository(settings.database_url)
document_repository = DocumentRepository(audit_repository.session_factory)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Initialize and clean up process-level resources."""

    logging.basicConfig(level=settings.log_level.upper())
    audit_repository.initialize()
    app.state.audit_repository = audit_repository
    app.state.document_repository = document_repository
    try:
        yield
    finally:
        audit_repository.close()


app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)
app.add_middleware(PrometheusMiddleware)
app.add_middleware(AuditContextMiddleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(ApiKeyMiddleware)
app.include_router(v1_router)


@app.get("/health", tags=["operations"])
async def health_check() -> JSONResponse:
    """Report API process liveness without checking external dependencies."""

    return JSONResponse({"status": "ok", "environment": settings.environment})


@app.get("/ready", tags=["operations"])
async def readiness_check() -> JSONResponse:
    """Report whether required runtime dependencies are reachable and configured."""

    checks: dict[str, str] = {
        "database": "ok",
        "qdrant": "ok",
        "gemini": "configured" if settings.gemini_api_key else "missing",
        "llama_parse": "configured" if settings.llama_cloud_api_key else "missing",
        "api_auth": "configured" if settings.app_api_key else "missing",
    }
    try:
        audit_repository.check_connection()
    except Exception:
        checks["database"] = "unavailable"

    qdrant = AsyncQdrantClient(url=str(settings.qdrant_url))
    try:
        await qdrant.get_collections()
    except Exception:
        checks["qdrant"] = "unavailable"
    finally:
        await qdrant.close()

    ready = all(value in {"ok", "configured"} for value in checks.values()) and (
        settings.environment.lower() in {"development", "test"} or bool(settings.app_api_key)
    )
    return JSONResponse(
        {"status": "ready" if ready else "not_ready", "checks": checks},
        status_code=200 if ready else 503,
    )
