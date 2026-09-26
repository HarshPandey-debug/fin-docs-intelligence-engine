"""Prometheus metrics and middleware for HTTP-level observability."""

from collections.abc import Awaitable, Callable
from time import perf_counter

from fastapi import Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware

HTTP_REQUESTS_TOTAL = Counter(
    "findocs_http_requests_total",
    "Total HTTP requests processed by FinDocs.",
    ("method", "path", "status_code"),
)
HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "findocs_http_request_duration_seconds",
    "Time spent serving FinDocs HTTP requests.",
    ("method", "path"),
)
DOCUMENT_INGESTIONS_TOTAL = Counter(
    "findocs_document_ingestions_total",
    "Document ingestion jobs grouped by terminal outcome.",
    ("status",),
)
ANALYSIS_REQUESTS_TOTAL = Counter(
    "findocs_analysis_requests_total",
    "Document analysis requests grouped by terminal outcome and route.",
    ("status", "route"),
)
ANALYSIS_DURATION_SECONDS = Histogram(
    "findocs_analysis_duration_seconds",
    "End-to-end LangGraph document analysis duration.",
    ("route",),
)
LLM_TOKENS_TOTAL = Counter(
    "findocs_llm_tokens_total",
    "LLM tokens consumed by FinDocs agent stages.",
    ("token_type",),
)
CODE_EXECUTIONS_TOTAL = Counter(
    "findocs_code_executions_total",
    "Restricted financial calculation executions grouped by outcome.",
    ("status",),
)


class PrometheusMiddleware(BaseHTTPMiddleware):
    """Record request count, outcome, and latency without high-cardinality paths."""

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        start_time = perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            HTTP_REQUESTS_TOTAL.labels(request.method, request.url.path, "500").inc()
            raise
        route = request.scope.get("route")
        path = getattr(route, "path", request.url.path)
        HTTP_REQUEST_DURATION_SECONDS.labels(request.method, path).observe(perf_counter() - start_time)
        HTTP_REQUESTS_TOTAL.labels(request.method, path, str(response.status_code)).inc()
        return response


def metrics_response() -> Response:
    """Return Prometheus' metric exposition payload."""

    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
