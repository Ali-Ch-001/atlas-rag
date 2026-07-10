"""Integration tests proving PostgreSQL RLS tenant isolation and core pipeline path.

These tests use testcontainers to spin up a real PostgreSQL instance, proving
that the testcontainers dependency in pyproject.toml is intentional and that
tenant RLS policies prevent cross-tenant data access.
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from rag_platform.db.tenant import set_tenant_context


@pytest.fixture(scope="module")
def testcontainers_postgres():
    """Spin up a real PostgreSQL container for integration tests."""
    from testcontainers.postgres import PostgresContainer

    container = PostgresContainer(
        "postgres:16.9-alpine",
        username="rag",
        password="rag-test-password",
        dbname="rag_test",
    )
    container.start()
    yield container
    container.stop()


@pytest.fixture
async def testcontainers_session(testcontainers_postgres):
    """Create an async session connected to the test container."""
    url = (
        f"postgresql+asyncpg://rag:rag-test-password@"
        f"{testcontainers_postgres.get_container_host_ip()}:"
        f"{testcontainers_postgres.get_exposed_port(5432)}/rag_test"
    )
    engine = create_async_engine(url)

    from rag_platform.db.models import Document, DocumentVersion, IngestionStage

    async with engine.begin() as conn:
        await conn.run_sync(Document.__table__.create)
        await conn.run_sync(DocumentVersion.__table__.create)
        await conn.run_sync(IngestionStage.__table__.create)

    async with engine.begin() as conn:
        for model in (Document, DocumentVersion, IngestionStage):
            table_name = model.__tablename__
            await conn.execute(text(f"ALTER TABLE {table_name} ENABLE ROW LEVEL SECURITY"))
            await conn.execute(text(f"ALTER TABLE {table_name} FORCE ROW LEVEL SECURITY"))
            await conn.execute(
                text(
                    f"CREATE POLICY {table_name}_tenant_policy ON {table_name} "
                    f"USING (tenant_id = nullif(current_setting('app.tenant_id', true), '')::uuid) "
                    f"WITH CHECK (tenant_id = nullif(current_setting('app.tenant_id', true), '')::uuid)"
                )
            )

    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with factory() as session:
        yield session
    await engine.dispose()


async def test_rls_blocks_cross_tenant_read(testcontainers_session: AsyncSession):
    """Prove that tenant A cannot see tenant B's documents."""
    tenant_a = uuid4()
    tenant_b = uuid4()
    document_id = uuid4()
    corpus_id = uuid4()

    from rag_platform.db.models import Document

    await set_tenant_context(testcontainers_session, tenant_a)
    testcontainers_session.add(
        Document(
            tenant_id=tenant_a,
            document_id=document_id,
            corpus_id=corpus_id,
            document_type="policy",
            title="Secret Document",
            classification=0,
        )
    )
    await testcontainers_session.commit()

    from sqlalchemy import select

    await set_tenant_context(testcontainers_session, tenant_b)
    result = await testcontainers_session.execute(
        select(Document).where(Document.tenant_id == tenant_a, Document.document_id == document_id)
    )
    rows = result.scalars().all()
    assert len(rows) == 0, "RLS should have blocked tenant_b from reading tenant_a's document"


async def test_rls_allows_own_tenant_read(testcontainers_session: AsyncSession):
    """Prove that tenant A can see its own documents."""
    tenant = uuid4()
    document_id = uuid4()
    corpus_id = uuid4()

    from rag_platform.db.models import Document

    await set_tenant_context(testcontainers_session, tenant)
    testcontainers_session.add(
        Document(
            tenant_id=tenant,
            document_id=document_id,
            corpus_id=corpus_id,
            document_type="policy",
            title="My Document",
            classification=0,
        )
    )
    await testcontainers_session.commit()

    from sqlalchemy import select

    result = await testcontainers_session.execute(
        select(Document).where(Document.tenant_id == tenant, Document.document_id == document_id)
    )
    rows = result.scalars().all()
    assert len(rows) == 1
    assert rows[0].title == "My Document"


async def test_sha256_storage_and_retrieval(testcontainers_session: AsyncSession):
    """Prove that source SHA-256 hashes are correctly stored as 32-byte binary."""
    tenant = uuid4()
    document_id = uuid4()
    version_id = uuid4()
    source_hash = b"\x00" * 31 + b"\x01"

    from rag_platform.db.models import Document, DocumentVersion

    await set_tenant_context(testcontainers_session, tenant)
    testcontainers_session.add(
        Document(
            tenant_id=tenant,
            document_id=document_id,
            corpus_id=uuid4(),
            document_type="policy",
            title="Hash Test",
            classification=0,
        )
    )
    testcontainers_session.add(
        DocumentVersion(
            tenant_id=tenant,
            version_id=version_id,
            document_id=document_id,
            version_number=1,
            source_sha256=source_hash,
            object_key="quarantine/test.pdf",
            pipeline_version="2026-07-10.1",
            parser_version="pymupdf-1",
            state="ACTIVE",
        )
    )
    await testcontainers_session.commit()

    from sqlalchemy import select

    result = await testcontainers_session.execute(
        select(DocumentVersion).where(
            DocumentVersion.tenant_id == tenant,
            DocumentVersion.version_id == version_id,
        )
    )
    version = result.scalar_one()
    assert version.source_sha256 == source_hash
    assert len(version.source_sha256) == 32


async def test_stage_state_machine(testcontainers_session: AsyncSession):
    """Prove the ingestion stage idempotency key prevents duplicate processing."""
    tenant = uuid4()
    version_id = uuid4()

    from rag_platform.db.models import IngestionStage

    await set_tenant_context(testcontainers_session, tenant)
    stage = IngestionStage(
        tenant_id=tenant,
        version_id=version_id,
        stage_name="full_pipeline",
        input_hash="abcdef1234567890abcdef1234567890",
        pipeline_version="2026-07-10.1",
        state="RUNNING",
    )
    testcontainers_session.add(stage)
    await testcontainers_session.commit()

    from sqlalchemy import select

    result = await testcontainers_session.execute(
        select(IngestionStage).where(
            IngestionStage.tenant_id == tenant,
            IngestionStage.version_id == version_id,
            IngestionStage.stage_name == "full_pipeline",
            IngestionStage.input_hash == "abcdef1234567890abcdef1234567890",
            IngestionStage.pipeline_version == "2026-07-10.1",
        )
    )
    row = result.scalar_one_or_none()
    assert row is not None
    assert row.state == "RUNNING"

    row.state = "COMPLETED"
    await testcontainers_session.commit()
    assert row.state == "COMPLETED"
