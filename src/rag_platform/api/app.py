from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from rag_platform.adapters.events import EventProducer, OutboxPublisher
from rag_platform.api.dependencies import (
    get_cache_store,
    get_graph_store,
    get_object_store,
    get_vector_store,
)
from rag_platform.api.documents import router as documents_router
from rag_platform.api.health import router as health_router
from rag_platform.api.ingestion import router as ingestion_router
from rag_platform.api.responses import router as responses_router
from rag_platform.api.search import router as search_router
from rag_platform.config import get_settings
from rag_platform.logging import configure_logging
from rag_platform.telemetry import configure_prometheus, configure_telemetry

settings = get_settings()
configure_logging(settings.log_level)
logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    object_store = get_object_store()
    vector_store = get_vector_store()
    cache = get_cache_store()
    await object_store.ensure_buckets()
    await vector_store.ensure_collection()
    await cache.ensure_semantic_index()
    graph = get_graph_store()
    if graph:
        await graph.ensure_schema()
    producer = EventProducer(settings)
    await producer.start()
    publisher = OutboxPublisher(producer)
    publisher_task = asyncio.create_task(publisher.run(), name="outbox-publisher")
    app.state.outbox_publisher = publisher
    logger.info("application_started", environment=settings.environment)
    try:
        yield
    finally:
        publisher.stop()
        await publisher_task
        await producer.stop()
        await cache.close()
        if graph:
            await graph.close()
        logger.info("application_stopped")


app = FastAPI(
    title="Atlas RAG API",
    version="0.1.0",
    default_response_class=ORJSONResponse,
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=[
        "Authorization",
        "Content-Type",
        "Idempotency-Key",
        "X-Tenant-ID",
        "X-Subject-ID",
    ],
)
app.include_router(health_router)
app.include_router(documents_router)
app.include_router(ingestion_router)
app.include_router(search_router)
app.include_router(responses_router)
configure_prometheus(app)
configure_telemetry(app, settings)


@app.get("/metrics", include_in_schema=False)
async def prometheus_metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
