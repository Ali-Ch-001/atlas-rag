from __future__ import annotations

from collections.abc import Awaitable, Callable, Coroutine
from typing import Any, TypeVar

from fastapi import Request, Response, status
from starlette.middleware.base import BaseHTTPMiddleware

from rag_platform.adapters.cache import CacheStore
from rag_platform.config import get_settings

F = TypeVar("F", bound=Callable[..., Coroutine[Any, Any, Any]])

_RATE_LIMIT_ATTR = "_rate_limit"

DEFAULT_ENDPOINT_LIMITS: dict[str, tuple[int, int]] = {
    "/v1/search": (200, 60),
    "/v1/responses": (100, 60),
    "/v1/documents": (20, 60),
    "/v1/evaluation": (10, 60),
    "/v1/ingestion": (30, 60),
}


class RateLimiter:
    __slots__ = ("_cache",)

    def __init__(self, cache: CacheStore) -> None:
        self._cache = cache

    async def _incr(self, key: str) -> int:
        result = await self._cache.client.incr(key)
        return int(result)

    async def _expire(self, key: str, seconds: int) -> None:
        await self._cache.client.expire(key, seconds)

    async def _ttl(self, key: str) -> int:
        result = await self._cache.client.ttl(key)
        return int(result)

    async def check(
        self, *, namespace: str, route_key: str, max_requests: int, window_seconds: int
    ) -> tuple[bool, int]:
        redis_key = f"ratelimit:{namespace}:{route_key}"
        current = await self._incr(redis_key)
        if current == 1:
            await self._expire(redis_key, window_seconds)
        if current > max_requests:
            ttl = await self._ttl(redis_key)
            return True, max(ttl, 1)
        return False, 0


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(
        self,
        app: Callable[..., Awaitable[None]],
        *,
        cache_store: CacheStore,
        endpoint_limits: dict[str, tuple[int, int]] | None = None,
        skip_paths: list[str] | None = None,
    ) -> None:
        super().__init__(app)
        self._limiter = RateLimiter(cache_store)
        self._endpoint_limits = endpoint_limits or {}
        self._skip_paths = skip_paths or []

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        path = request.url.path
        for skip in self._skip_paths:
            if path.startswith(skip):
                return await call_next(request)

        max_req, window = 100, 60
        route_obj = request.scope.get("route")
        if route_obj is not None:
            endpoint = getattr(route_obj, "endpoint", None)
            if endpoint is not None:
                func_limits = getattr(endpoint, _RATE_LIMIT_ATTR, None)
                if isinstance(func_limits, dict):
                    max_req = func_limits.get("max_requests", max_req)
                    window = func_limits.get("window_seconds", window)

        if max_req == 100 and window == 60:
            for prefix, (limit, w) in self._endpoint_limits.items():
                if path.startswith(prefix):
                    max_req, window = limit, w
                    break

        settings = get_settings()
        if settings.auth_disabled:
            tenant_key = "default"
        else:
            tenant_key = request.headers.get("X-Tenant-ID", "unknown")

        route_key = path.rstrip("/").replace("/", ".")
        limited, retry_after = await self._limiter.check(
            namespace=tenant_key,
            route_key=route_key,
            max_requests=max_req,
            window_seconds=window,
        )

        if limited:
            return Response(
                content=b'{"detail":"Rate limit exceeded"}',
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                headers={
                    "Retry-After": str(retry_after),
                    "Content-Type": "application/json",
                },
            )

        return await call_next(request)


def rate_limit(
    max_requests: int = 100,
    window_seconds: int = 60,
) -> Callable[[F], F]:
    def decorator(func: F) -> F:
        setattr(
            func,
            _RATE_LIMIT_ATTR,
            {"max_requests": max_requests, "window_seconds": window_seconds},
        )
        return func

    return decorator
