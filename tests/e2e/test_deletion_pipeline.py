from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from rag_platform.config import Settings
from rag_platform.db.models import Chunk, Document, DocumentVersion, OutboxEvent, RetrievalLog
from rag_platform.domain.models import DocumentState
from rag_platform.security.auth import AuthContext
from rag_platform.services.documents import DocumentService


class _MockObjectStore:
    async def upload_fileobj(self, bucket: str, key: str, file: Any, **kwargs: Any) -> None:
        pass


@pytest.fixture(scope="module")
def _deletion_test_pg() -> Any:
    from testcontainers.postgres import PostgresContainer  # type: ignore[import-not-found]

    container = PostgresContainer(
        "postgres:16.9-alpine",
        username="rag",
        password="rag-test-password",  # noqa: S106
        dbname="rag_test",
    )
    container.start()
    yield container
    container.stop()


@pytest.fixture
async def deletion_test_session(_deletion_test_pg: Any) -> Any:
    url = (
        "postgresql+asyncpg://rag:rag-test-password@"
        f"{_deletion_test_pg.get_container_host_ip()}:"
        f"{_deletion_test_pg.get_exposed_port(5432)}/rag_test"
    )
    engine = create_async_engine(url)

    async with engine.begin() as conn:
        await conn.run_sync(Document.__table__.create)
        await conn.run_sync(DocumentVersion.__table__.create)
        await conn.run_sync(Chunk.__table__.create)
        await conn.run_sync(OutboxEvent.__table__.create)
        await conn.run_sync(RetrievalLog.__table__.create)

    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with factory() as session:
        yield session
    await engine.dispose()


async def _insert_document_and_chunks(
    session: AsyncSession,
    tenant_id: Any,
) -> tuple[Any, Any, Any]:
    document_id = uuid4()
    corpus_id = uuid4()
    version_id = uuid4()

    document = Document(
        tenant_id=tenant_id,
        document_id=document_id,
        corpus_id=corpus_id,
        current_version_id=None,
        document_type="policy",
        title="Deletion Test Document",
        classification=0,
    )
    session.add(document)

    version = DocumentVersion(
        tenant_id=tenant_id,
        version_id=version_id,
        document_id=document_id,
        version_number=1,
        source_sha256=b"\x00" * 31 + b"\x01",
        normalized_text_sha256=None,
        object_key=f"quarantine/{tenant_id}/{document_id}/{version_id}/hash.pdf",
        source_date=None,
        pipeline_version="2026-07-10.1",
        parser_version="pymupdf-1",
        state=DocumentState.active.value,
    )
    session.add(version)

    chunk_a = Chunk(
        tenant_id=tenant_id,
        chunk_id=uuid4(),
        corpus_id=corpus_id,
        document_id=document_id,
        version_id=version_id,
        section_id=None,
        title="Section A",
        heading_path=None,
        content="Content of chunk A for deletion lifecycle test.",
        content_sha256=b"\x01" * 32,
        ordinal=1,
        page_from=1,
        page_to=1,
        token_count=10,
        language="en",
        document_type="policy",
        acl_groups=[],
        classification=0,
        source_spans=[],
    )
    session.add(chunk_a)

    chunk_b = Chunk(
        tenant_id=tenant_id,
        chunk_id=uuid4(),
        corpus_id=corpus_id,
        document_id=document_id,
        version_id=version_id,
        section_id=None,
        title="Section B",
        heading_path=None,
        content="Content of chunk B for deletion lifecycle test.",
        content_sha256=b"\x02" * 32,
        ordinal=2,
        page_from=2,
        page_to=2,
        token_count=10,
        language="en",
        document_type="policy",
        acl_groups=[],
        classification=0,
        source_spans=[],
    )
    session.add(chunk_b)

    return document_id, corpus_id, chunk_a.chunk_id


async def _insert_retrieval_log(
    session: AsyncSession,
    tenant_id: Any,
    chunk_id: Any,
) -> None:
    log = RetrievalLog(
        log_id=uuid4(),
        request_id=uuid4(),
        tenant_id=tenant_id,
        query_hash="abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
        chunk_id=chunk_id,
        dense_score=None,
        sparse_score=None,
        rrf_score=0.85,
        reranker_score=0.90,
        final_rank=None,
    )
    session.add(log)


async def _tombstone_chunks(session: AsyncSession, document_id: Any) -> None:
    result = await session.execute(select(Chunk).where(Chunk.document_id == document_id))
    for chunk in result.scalars().all():
        chunk.deleted_at = datetime.now(UTC)
    await session.commit()


async def _count_visibly_tombstoned_chunks(
    session: AsyncSession, document_id: Any, tenant_id: Any
) -> int:
    result = await session.execute(
        select(Chunk)
        .join(
            Document,
            (Document.tenant_id == Chunk.tenant_id) & (Document.document_id == Chunk.document_id),
        )
        .where(
            Chunk.document_id == document_id,
            Chunk.tenant_id == tenant_id,
            Chunk.deleted_at.isnot(None),
            Document.deleted_at.isnot(None),
        )
    )
    return len(result.scalars().all())


async def _count_retrieval_excluded_chunks(
    session: AsyncSession, document_id: Any, tenant_id: Any
) -> int:
    result = await session.execute(
        select(Chunk)
        .join(
            Document,
            (Document.tenant_id == Chunk.tenant_id) & (Document.document_id == Chunk.document_id),
        )
        .where(
            Chunk.document_id == document_id,
            Chunk.tenant_id == tenant_id,
            Document.deleted_at.is_(None),
        )
    )
    return len(result.scalars().all())


async def test_deletion_lifecycle_end_to_end(deletion_test_session: AsyncSession) -> None:
    tenant_id = uuid4()
    auth = AuthContext(
        tenant_id=tenant_id,
        subject_id="test-user",
        groups=(),
        permissions=frozenset(),
        clearance=0,
    )
    settings = Settings(environment="test")
    service = DocumentService(settings, _MockObjectStore())

    document_id, _corpus_id, chunk_id = await _insert_document_and_chunks(
        deletion_test_session, tenant_id
    )
    await _insert_retrieval_log(deletion_test_session, tenant_id, chunk_id)
    await deletion_test_session.commit()

    result = await service.delete(deletion_test_session, auth, document_id)
    assert result is True, "delete() should return True for an active document"

    document = await deletion_test_session.get(
        Document, {"tenant_id": tenant_id, "document_id": document_id}
    )
    assert document is not None
    assert document.deleted_at is not None, "Document must be tombstoned (deleted_at set)"

    versions_result = await deletion_test_session.execute(
        select(DocumentVersion).where(
            DocumentVersion.tenant_id == tenant_id,
            DocumentVersion.document_id == document_id,
        )
    )
    for version in versions_result.scalars().all():
        assert version.state == DocumentState.deleted.value, (
            f"Version {version.version_id} must be marked DELETED"
        )

    outbox_result = await deletion_test_session.execute(
        select(OutboxEvent).where(
            OutboxEvent.tenant_id == tenant_id,
            OutboxEvent.topic == "document.delete.requested.v1",
        )
    )
    outbox_events = outbox_result.scalars().all()
    assert len(outbox_events) == 1, "Must create exactly one deletion outbox event"
    assert outbox_events[0].event_key == str(document_id)

    await _tombstone_chunks(deletion_test_session, document_id)

    visible_count = await _count_visibly_tombstoned_chunks(
        deletion_test_session, document_id, tenant_id
    )
    assert visible_count == 2, "Both chunks must be tombstoned (deleted_at set)"

    excluded_count = await _count_retrieval_excluded_chunks(
        deletion_test_session, document_id, tenant_id
    )
    assert excluded_count == 0, (
        "Retrieval queries joining on documents.deleted_at IS NULL must exclude tombstoned chunks"
    )
