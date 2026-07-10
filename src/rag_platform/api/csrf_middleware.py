from __future__ import annotations

from collections.abc import Awaitable, Callable

from fastapi import Request, Response, status
from starlette.middleware.base import BaseHTTPMiddleware

from rag_platform.config import get_settings


def _extract_host(source: str) -> str:
    if "://" in source:
        source = source.split("://", 1)[1]
    return source.split("/", 0)[0].split(":")[0].lower()


class CsrfProtectionMiddleware(BaseHTTPMiddleware):
    def __init__(
        self,
        app: Callable[..., Awaitable[None]],
        *,
        skip_paths: list[str] | None = None,
    ) -> None:
        super().__init__(app)
        self._settings = get_settings()
        self._skip_paths = skip_paths or []

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        if self._settings.auth_disabled:
            return await call_next(request)

        if self._settings.environment not in {"stage", "prod"}:
            return await call_next(request)

        for skip in self._skip_paths:
            if request.url.path.startswith(skip):
                return await call_next(request)

        if request.method in {"GET", "HEAD", "OPTIONS", "TRACE"}:
            return await call_next(request)

        origin = request.headers.get("Origin", "")
        referer = request.headers.get("Referer", "")
        host = request.headers.get("Host", "")

        source = origin or referer
        if not source:
            return await call_next(request)

        if _extract_host(source) != _extract_host(host):
            return Response(
                content=b'{"detail":"Cross-origin request blocked"}',
                status_code=status.HTTP_403_FORBIDDEN,
                headers={"Content-Type": "application/json"},
            )

        return await call_next(request)
