"""Composition root for the version 1 public API."""

import logging
from pathlib import Path
from time import perf_counter
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Request, UploadFile, status
from pydantic import BaseModel, Field

from app.core.config import Settings, get_settings
from app.core.metrics import ANALYSIS_DURATION_SECONDS, ANALYSIS_REQUESTS_TOTAL, CODE_EXECUTIONS_TOTAL, DOCUMENT_INGESTIONS_TOTAL, LLM_TOKENS_TOTAL
from app.core.metrics import metrics_response
from app.agents.compliance import ComplianceAgent
from app.agents.quantitative import DeterministicCodeInterpreter, QuantitativeAgent
from app.agents.retrieval import RetrievalAgent
from app.agents.supervisor import SupervisorAgent
from app.domain.analysis import QueryResponse
from app.domain.audit import QueryAuditEvent
from app.services.chunking import FinancialSemanticChunker
from app.services.documents import DocumentIngestionRecord, DocumentRepository
from app.services.embeddings import GeminiEmbeddingProvider
from app.services.ingestion import DocumentIngestionService
from app.services.parsing import FinancialDocumentParser
from app.services.vector_store import QdrantDocumentStore
from app.services.audit import AuditRepository
from app.workflows.analysis import FinancialAnalysisWorkflow

router = APIRouter(prefix="/api/v1")
logger = logging.getLogger(__name__)
ALLOWED_EXTENSIONS = {".pdf", ".csv"}


class UploadAcceptedResponse(BaseModel):
    """Response issued immediately after a document upload is accepted."""

    document_id: str
    status: str = "pending"
    filename: str
    message: str = "Document accepted for background ingestion."


class QueryRequest(BaseModel):
    """A document-scoped question submitted to the LangGraph workflow."""

    doc_id: str = Field(min_length=1, max_length=128)
    query: str = Field(min_length=3, max_length=10_000)


def get_ingestion_service(settings: Settings = Depends(get_settings)) -> DocumentIngestionService:
    """Construct a short-lived ingestion service using validated configuration."""

    return DocumentIngestionService(
        parser=FinancialDocumentParser(settings.llama_cloud_api_key),
        chunker=FinancialSemanticChunker(),
        embedder=GeminiEmbeddingProvider(
            api_key=settings.gemini_api_key,
            model=settings.embedding_model,
            dimensions=settings.embedding_dimensions,
        ),
        vector_store=QdrantDocumentStore(
            url=str(settings.qdrant_url),
            collection_name=settings.qdrant_collection,
            vector_size=settings.embedding_dimensions,
        ),
    )


def get_analysis_workflow(settings: Settings = Depends(get_settings)) -> FinancialAnalysisWorkflow:
    """Build a request-scoped compiled graph with retrieval resources to close after use."""

    store = QdrantDocumentStore(
        url=str(settings.qdrant_url),
        collection_name=settings.qdrant_collection,
        vector_size=settings.embedding_dimensions,
    )
    return FinancialAnalysisWorkflow(
        supervisor=SupervisorAgent(settings.gemini_api_key, settings.generation_model),
        retrieval=RetrievalAgent(
            embedder=GeminiEmbeddingProvider(
                api_key=settings.gemini_api_key,
                model=settings.embedding_model,
                dimensions=settings.embedding_dimensions,
            ),
            store=store,
            limit=settings.retrieval_limit,
        ),
        compliance=ComplianceAgent(Path(__file__).resolve().parents[3] / "app" / "rules" / "credit_agreement_rules.json"),
        quantitative=QuantitativeAgent(DeterministicCodeInterpreter(settings.code_execution_timeout_seconds)),
    )


def get_audit_repository(request: Request) -> AuditRepository:
    """Return the lifespan-managed audit repository for this API request."""

    return request.app.state.audit_repository


def get_document_repository(request: Request) -> DocumentRepository:
    """Return the lifespan-managed document repository for this API request."""

    return request.app.state.document_repository


async def _persist_upload(upload: UploadFile, destination: Path, max_bytes: int, extension: str) -> None:
    """Stream an upload to disk while enforcing the configured size limit."""

    bytes_written = 0
    is_first_block = True
    try:
        with destination.open("xb") as destination_file:
            while block := await upload.read(1_048_576):
                if is_first_block and extension == ".pdf" and not block.startswith(b"%PDF-"):
                    raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Uploaded file is not a valid PDF")
                is_first_block = False
                bytes_written += len(block)
                if bytes_written > max_bytes:
                    raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="Upload exceeds size limit")
                destination_file.write(block)
    except HTTPException:
        destination.unlink(missing_ok=True)
        raise
    except OSError as exc:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Unable to persist uploaded document") from exc
    finally:
        await upload.close()


async def _run_ingestion(
    document_id: str,
    source_path: Path,
    filename: str,
    service: DocumentIngestionService,
    repository: DocumentRepository,
) -> None:
    """Execute ingestion in the background and expose only safe failure state."""

    repository.mark_processing(document_id)
    try:
        result = await service.ingest(document_id, source_path, filename)
        repository.mark_completed(document_id, result.chunk_count)
        DOCUMENT_INGESTIONS_TOTAL.labels("completed").inc()
        logger.info("Document ingestion completed", extra={"document_id": document_id, "chunks": result.chunk_count})
    except Exception:
        repository.mark_failed(document_id, "Document ingestion failed. Review service logs for details.")
        DOCUMENT_INGESTIONS_TOTAL.labels("failed").inc()
        logger.exception("Document ingestion failed", extra={"document_id": document_id})
    finally:
        await service.close()


@router.post("/upload", response_model=UploadAcceptedResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    background_tasks: BackgroundTasks,
    document: Annotated[UploadFile, File(description="SEC filing or credit agreement in PDF or CSV format")],
    service: Annotated[DocumentIngestionService, Depends(get_ingestion_service)],
    repository: Annotated[DocumentRepository, Depends(get_document_repository)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> UploadAcceptedResponse:
    """Accept a PDF or CSV and schedule parsing, embedding, and Qdrant upsert."""

    filename = Path(document.filename or "").name
    extension = Path(filename).suffix.lower()
    if not filename or extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Only PDF and CSV documents are supported")

    document_id = str(uuid4())
    settings.document_storage_dir.mkdir(parents=True, exist_ok=True)
    destination = settings.document_storage_dir / f"{document_id}{extension}"
    await _persist_upload(document, destination, settings.max_upload_bytes, extension)
    repository.create(document_id, filename)
    background_tasks.add_task(_run_ingestion, document_id, destination, filename, service, repository)
    return UploadAcceptedResponse(document_id=document_id, filename=filename)


@router.get("/documents/{document_id}", response_model=DocumentIngestionRecord)
async def get_document_status(
    document_id: str,
    repository: Annotated[DocumentRepository, Depends(get_document_repository)],
) -> DocumentIngestionRecord:
    """Return the current ingestion status for a document accepted by this API instance."""

    record = repository.get(document_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document status was not found")
    return record


@router.post("/query", response_model=QueryResponse)
async def query_document(
    payload: QueryRequest,
    http_request: Request,
    workflow: Annotated[FinancialAnalysisWorkflow, Depends(get_analysis_workflow)],
    audit_repository: Annotated[AuditRepository, Depends(get_audit_repository)],
) -> QueryResponse:
    """Run supervisor-routed retrieval, compliance, and deterministic calculation analysis."""

    request_id = str(http_request.state.request_id)
    start_time = float(http_request.state.request_started_at)
    try:
        response = await workflow.invoke(payload.doc_id, payload.query)
    except Exception as exc:
        latency_ms = int((perf_counter() - start_time) * 1_000)
        _record_audit_event(
            audit_repository,
            QueryAuditEvent(
                request_id=request_id,
                document_id=payload.doc_id,
                query=payload.query,
                latency_ms=latency_ms,
                status="failed",
                error_message=f"Workflow exception: {type(exc).__name__}",
            ),
        )
        ANALYSIS_REQUESTS_TOTAL.labels("failed", "unknown").inc()
        logger.exception("Document analysis workflow failed", extra={"document_id": payload.doc_id, "request_id": request_id})
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Document analysis is temporarily unavailable. No financial conclusion was generated.",
        ) from exc

    latency_ms = int((perf_counter() - start_time) * 1_000)
    outcome = "completed_with_errors" if response.errors else "completed"
    _record_audit_event(
        audit_repository,
        QueryAuditEvent(
            request_id=request_id,
            document_id=response.document_id,
            query=response.query,
            route=response.route.value,
            steps=response.steps,
            prompt_tokens=response.token_usage.prompt_tokens,
            completion_tokens=response.token_usage.completion_tokens,
            total_tokens=response.token_usage.total_tokens,
            latency_ms=latency_ms,
            status=outcome,
            final_output=response,
        ),
    )
    ANALYSIS_REQUESTS_TOTAL.labels(outcome, response.route.value).inc()
    ANALYSIS_DURATION_SECONDS.labels(response.route.value).observe(latency_ms / 1_000)
    LLM_TOKENS_TOTAL.labels("prompt").inc(response.token_usage.prompt_tokens)
    LLM_TOKENS_TOTAL.labels("completion").inc(response.token_usage.completion_tokens)
    if response.quantitative_result is not None:
        CODE_EXECUTIONS_TOTAL.labels(response.quantitative_result.status).inc()
    return response


def _record_audit_event(audit_repository: AuditRepository, event: QueryAuditEvent) -> None:
    """Persist audit evidence without replacing a valid analysis response on DB failure."""

    try:
        audit_repository.record(event)
    except Exception:
        logger.exception("Unable to persist query audit event", extra={"request_id": event.request_id})


@router.get("/metrics", include_in_schema=False)
async def metrics() -> object:
    """Expose application metrics for the Prometheus scraper."""

    return metrics_response()
