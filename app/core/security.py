"""Authentication dependencies for protected API routes."""

from hmac import compare_digest

from fastapi import Depends, Header, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from app.core.config import Settings, get_settings


def require_api_key(
    api_key: str | None = Header(default=None, alias="X-API-Key"),
    settings: Settings = Depends(get_settings),
) -> None:
    """Require the configured API key outside development mode."""

    if not settings.app_api_key:
        if settings.environment.lower() in {"development", "test"}:
            return
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="API authentication is not configured")
    if not api_key or not compare_digest(api_key, settings.app_api_key):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")


class ApiKeyMiddleware(BaseHTTPMiddleware):
    """Protect versioned API routes without coupling authentication to router setup."""

    async def dispatch(self, request: Request, call_next: object) -> Response:
        if not request.url.path.startswith("/api/v1/"):
            return await call_next(request)  # type: ignore[misc]
        settings = get_settings()
        supplied = request.headers.get("X-API-Key")
        if not settings.app_api_key and settings.environment.lower() in {"development", "test"}:
            return await call_next(request)  # type: ignore[misc]
        if not settings.app_api_key:
            return JSONResponse({"detail": "API authentication is not configured"}, status_code=503)
        if not supplied or not compare_digest(supplied, settings.app_api_key):
            return JSONResponse({"detail": "Invalid API key"}, status_code=401)
        return await call_next(request)  # type: ignore[misc]