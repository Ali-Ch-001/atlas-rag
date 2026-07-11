"""End-to-end retrieval pipeline test using testcontainers for PostgreSQL + Redis.

Verifies the full ingestion → retrieval pipeline: a PDF with known text is
uploaded, ingested, and a search returns results containing expected content
with citations.
"""

from __future__ import annotations

from datetime import date
from io import BytesIO
from typing import Any
from uuid import UUID, uuid4

import fitz
import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from rag_platform.config import Settings
from rag_platform.security.auth import AuthContext


@pytest.fixture(scope="module")
def pg_container():
    from testcontainers.postgres import PostgresContainer

    container = PostgresContainer(
        "postgres:16.9-alpine",
        username="rag",
        password="rag-test-password",  # noqa: S106
        dbname="rag_test",
    )
    container.start()
    yield container
    container.stop()


@pytest.fixture(scope="module")
def redis_container():
    from testcontainers.redis import RedisContainer

    container = RedisContainer("redis:7.4-alpine")
    container.start()
    yield container
    container.stop()


@pytest.fixture(scope="module")
def test_settings(pg_container, redis_container):
    redis_port = redis_container.get_exposed_port(6379)
    pg_port = pg_container.get_exposed_port(5432)
    return Settings(
        environment="test",
        database_url=(
            f"postgresql+asyncpg://rag:rag-test-password@"
            f"{pg_container.get_container_host_ip()}:{pg_port}/rag_test"
        ),
        redis_url=f"redis://{redis_container.get_container_host_ip()}:{redis_port}/0",
        model_provider="deterministic",
        reranker_provider="lexical",
        ocr_enabled=False,
        clamav_enabled=False,
        otel_enabled=False,
    )


@pytest.fixture(scope="module")
def test_engine(test_settings):
    engine = create_async_engine(test_settings.database_url)
    return engine


@pytest.fixture(autouse=True)
async def create_tables(test_engine):
    from rag_platform.db.models import (
        Chunk,
        Document,
        DocumentVersion,
        IngestionStage,
        OutboxEvent,
        RetrievalLog,
        RetrievalRequestLog,
        Section,
    )

    async with test_engine.begin() as conn:
        await conn.run_sync(Document.__table__.create, checkfirst=True)
        await conn.run_sync(DocumentVersion.__table__.create, checkfirst=True)
        await conn.run_sync(Section.__table__.create, checkfirst=True)
        await conn.run_sync(Chunk.__table__.create, checkfirst=True)
        await conn.run_sync(IngestionStage.__table__.create, checkfirst=True)
        await conn.run_sync(OutboxEvent.__table__.create, checkfirst=True)
        await conn.run_sync(RetrievalLog.__table__.create, checkfirst=True)
        await conn.run_sync(RetrievalRequestLog.__table__.create, checkfirst=True)

    yield

    async with test_engine.begin() as conn:
        for table in (
            Chunk,
            Section,
            IngestionStage,
            OutboxEvent,
            RetrievalLog,
            RetrievalRequestLog,
            DocumentVersion,
            Document,
        ):
            await conn.run_sync(table.__table__.drop, checkfirst=True)


def create_test_pdf() -> bytes:
    text = (
        "The capital of France is Paris. "
        "The currency is the Euro. "
        "France is a member of the European Union."
    )
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), text, fontsize=12)
    output = document.write()
    document.close()
    return output


class InMemoryObjectStore:
    def __init__(self) -> None:
        self._store: dict[str, bytes] = {}

    async def upload_fileobj(
        self,
        bucket: str,
        key: str,
        fileobj: Any,
        *,
        content_type: str = "application/octet-stream",
        metadata: dict[str, str] | None = None,
    ) -> None:
        fileobj.seek(0)
        self._store[key] = fileobj.read()

    async def upload_bytes(
        self,
        bucket: str,
        key: str,
        content: bytes,
        *,
        content_type: str = "application/octet-stream",
        metadata: dict[str, str] | None = None,
    ) -> None:
        self._store[key] = content

    async def download_bytes(self, bucket: str, key: str) -> bytes:
        if key not in self._store:
            raise FileNotFoundError(f"Object not found: {key}")
        return self._store[key]

    async def copy(
        self,
        source_bucket: str,
        source_key: str,
        target_bucket: str,
        target_key: str,
    ) -> None:
        if source_key in self._store:
            self._store[target_key] = self._store[source_key]

    async def delete(self, bucket: str, key: str) -> None:
        self._store.pop(key, None)


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._points: list[dict[str, Any]] = []
        self.upsert_chunks_called = False

    async def upsert_chunks(self, chunks: list[Any], vectors: list[list[float]]) -> None:
        self.upsert_chunks_called = True
        for chunk, vector in zip(chunks, vectors, strict=True):
            self._points.append(
                {
                    "chunk_id": str(chunk.chunk_id),
                    "vector": vector,
                    "tenant_id": str(chunk.tenant_id),
                    "corpus_id": str(chunk.corpus_id),
                    "document_id": str(chunk.document_id),
                    "version_id": str(chunk.version_id),
                    "document_type": chunk.document_type,
                    "language": chunk.language,
                    "source_date": (chunk.source_date.isoformat() if chunk.source_date else None),
                    "classification": chunk.classification,
                    "acl_groups": chunk.acl_groups,
                }
            )

    async def search(
        self,
        vector: list[float],
        auth: Any,
        corpus_ids: list[UUID],
        document_types: list[str],
        date_from: date | None,
        date_to: date | None,
        limit: int,
    ) -> list[Any]:
        from rag_platform.adapters.vector_store import DenseHit

        results = []
        for point in self._points:
            if point["tenant_id"] != str(auth.tenant_id):
                continue
            if str(point["corpus_id"]) not in [str(cid) for cid in corpus_ids]:
                continue
            if document_types and point["document_type"] not in document_types:
                continue
            if point.get("deleted") is True:
                continue
            results.append(DenseHit(chunk_id=UUID(point["chunk_id"]), score=0.75))
        return results[:limit]


@pytest.fixture
def auth():
    return AuthContext(
        tenant_id=Settings().dev_tenant_id,
        subject_id="test-user",
        groups=("testers",),
        permissions=frozenset({"documents:read", "documents:write", "agents:run"}),
        clearance=100,
    )


@pytest.fixture
async def setup_session(test_engine):
    factory = async_sessionmaker(test_engine, expire_on_commit=False, class_=AsyncSession)
    async with factory() as session:
        yield session


@pytest.mark.asyncio
async def test_upload_ingest_search_returns_paris_citation(
    monkeypatch,
    test_settings,
    test_engine,
    setup_session,
    auth,
) -> None:
    from rag_platform.adapters.backpressure import BackpressureController
    from rag_platform.adapters.cache import CacheStore
    from rag_platform.adapters.embeddings import DeterministicEmbeddingProvider
    from rag_platform.db.models import OutboxEvent
    from rag_platform.domain.models import (
        IngestionEvent,
        SearchFilters,
        SearchRequest,
    )
    from rag_platform.ingestion.service import IngestionService
    from rag_platform.retrieval.reranker import LexicalReranker
    from rag_platform.services.documents import DocumentService
    from rag_platform.services.retrieval import RetrievalService

    # --- Mock external dependencies ---
    object_store = InMemoryObjectStore()
    vector_store = InMemoryVectorStore()

    embeddings = DeterministicEmbeddingProvider(test_settings.qdrant_vector_size)
    reranker = LexicalReranker(test_settings)
    cache = CacheStore(test_settings)
    await cache.ping()

    backpressure = BackpressureController(test_settings)

    retrieval_service = RetrievalService(
        test_settings, embeddings, vector_store, cache, reranker, backpressure
    )

    document_service = DocumentService(test_settings, object_store)

    # --- Override SessionFactory for ingestion ---
    factory = async_sessionmaker(test_engine, expire_on_commit=False, class_=AsyncSession)

    async def mock_session_factory():
        return factory()

    monkeypatch.setattr("rag_platform.ingestion.service.SessionFactory", mock_session_factory)

    # --- Create and upload PDF ---
    pdf_bytes = create_test_pdf()

    corpus_id = uuid4()
    response = await document_service.create(
        setup_session,
        auth,
        BytesIO(pdf_bytes),
        corpus_id=corpus_id,
        title="France Facts",
        document_type="report",
        source_date=date(2025, 1, 1),
    )
    assert response.state.value == "QUARANTINED"

    # --- Grab the IngestionEvent from outbox ---
    from sqlalchemy import select as sa_select

    outbox = (
        await setup_session.execute(
            sa_select(OutboxEvent).where(
                OutboxEvent.tenant_id == auth.tenant_id,
                OutboxEvent.topic == "document.accepted.v1",
            )
        )
    ).scalar_one()
    event = IngestionEvent.model_validate(outbox.payload)

    # --- Run ingestion pipeline ---
    ingestion = IngestionService(
        test_settings,
        object_store,
        embeddings,
        vector_store,
        cache,
        backpressure=backpressure,
    )
    await ingestion.process(event)
    assert vector_store.upsert_chunks_called is True

    # --- Search ---
    search_request = SearchRequest(
        query="What is the capital of France?",
        filters=SearchFilters(corpus_ids=[corpus_id]),
        top_k=5,
    )
    search_response = await retrieval_service.search(setup_session, auth, search_request)

    assert len(search_response.results) > 0, "Search should return results"
    top_content = search_response.results[0].content.lower()
    assert "paris" in top_content, f"Top result should mention Paris, got: {top_content}"

    citations = [result.citation for result in search_response.results]
    assert len(citations) > 0, "At least one result should have a citation"
    assert citations[0].citation_id is not None

    # --- Cleanup ---
    await cache.close()
