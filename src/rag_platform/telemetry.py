from __future__ import annotations

import time

from fastapi import FastAPI, Request
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from prometheus_client import Counter, Histogram

from rag_platform.config import Settings
from rag_platform.db.session import engine

HTTP_REQUESTS = Counter(
    "atlas_http_requests_total",
    "HTTP requests by route and response status",
    ("method", "route", "status"),
)
HTTP_DURATION = Histogram(
    "atlas_http_request_duration_seconds",
    "HTTP request duration by route",
    ("method", "route"),
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30),
)


def configure_prometheus(app: FastAPI) -> None:
    @app.middleware("http")
    async def observe_request(request: Request, call_next):  # type: ignore[no-untyped-def]
        started = time.perf_counter()
        response = await call_next(request)
        route_object = request.scope.get("route")
        route = getattr(route_object, "path", "unmatched")
        HTTP_REQUESTS.labels(request.method, route, str(response.status_code)).inc()
        HTTP_DURATION.labels(request.method, route).observe(time.perf_counter() - started)
        return response


def configure_telemetry(app: FastAPI, settings: Settings) -> None:
    if not settings.otel_enabled:
        return
    provider = TracerProvider(
        resource=Resource.create(
            {
                "service.name": settings.otel_service_name,
                "deployment.environment": settings.environment,
            }
        )
    )
    provider.add_span_processor(
        BatchSpanProcessor(
            OTLPSpanExporter(endpoint=settings.otel_exporter_otlp_endpoint, insecure=True)
        )
    )
    trace.set_tracer_provider(provider)
    FastAPIInstrumentor.instrument_app(app)
    HTTPXClientInstrumentor().instrument()
    SQLAlchemyInstrumentor().instrument(engine=engine.sync_engine)
