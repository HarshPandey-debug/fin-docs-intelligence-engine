"""Request middleware supplying a stable correlation ID to audit logging."""

from collections.abc import Awaitable, Callable
from time import perf_counter
from uuid import uuid4

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware


class AuditContextMiddleware(BaseHTTPMiddleware):
    """Attach a request ID and monotonic start time for durable query audit events."""

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        request.state.request_id = str(uuid4())
        request.state.request_started_at = perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response
