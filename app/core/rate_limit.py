"""Small-process request limiting for the API edge."""

from collections import defaultdict, deque
from time import monotonic

from fastapi import Request, Response, status
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.config import get_settings


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Limit API requests per client process; use a shared edge limiter for replicas."""

    def __init__(self, app: object) -> None:
        super().__init__(app)
        self._requests: dict[str, deque[float]] = defaultdict(deque)

    async def dispatch(self, request: Request, call_next: object) -> Response:
        if not request.url.path.startswith("/api/v1/"):
            return await call_next(request)  # type: ignore[misc]
        client = request.client.host if request.client else "unknown"
        now = monotonic()
        window = self._requests[client]
        while window and now - window[0] >= 60:
            window.popleft()
        settings = get_settings()
        if len(window) >= settings.rate_limit_per_minute:
            return Response(
                content='{"detail":"Rate limit exceeded"}',
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                media_type="application/json",
                headers={"Retry-After": "60"},
            )
        window.append(now)
        return await call_next(request)  # type: ignore[misc]