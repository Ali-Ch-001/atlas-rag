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
from prometheus_client import Counter, Gauge, Histogram

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

GUARDRAIL_VIOLATIONS = Counter(
    "atlas_guardrail_violations_total",
    "Guardrail violations by type",
    ("violation_type",),
)
DOCUMENTS_INGESTED = Counter(
    "atlas_documents_ingested_total",
    "Documents ingested by state",
    ("state",),
)
CACHE_HITS = Counter(
    "atlas_cache_hits_total",
    "Cache hits by cache type",
    ("cache_type",),
)
RETRIEVAL_REQUESTS = Counter(
    "atlas_retrieval_requests_total",
    "Retrieval requests by outcome",
    ("outcome",),
)
RETRIEVAL_DURATION = Histogram(
    "atlas_retrieval_request_duration_seconds",
    "Retrieval request duration",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 2, 5),
)
INGESTION_STAGE_DURATION = Histogram(
    "atlas_ingestion_stage_duration_seconds",
    "Ingestion stage duration by stage name",
    ("stage",),
    buckets=(0.1, 0.5, 1, 2, 5, 10, 30, 60, 300, 900),
)
INGESTION_QUEUE_AGE = Gauge(
    "atlas_ingestion_queue_age_seconds",
    "Age of the oldest item in the ingestion queue",
)
DB_POOL_ACTIVE = Gauge(
    "atlas_database_pool_active_connections",
    "Active database connections in the pool",
)
TOOL_CALLS = Counter(
    "atlas_tool_calls_total",
    "Tool calls by tool name and outcome",
    ("tool", "outcome"),
)
TENANT_MISMATCH = Counter(
    "atlas_tenant_mismatch_attempts_total",
    "Cross-tenant access attempts blocked",
    ("endpoint",),
)
DEAD_LETTER_MESSAGES = Counter(
    "atlas_dead_letter_messages_total",
    "Messages sent to dead-letter topics after exhausting retries",
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
