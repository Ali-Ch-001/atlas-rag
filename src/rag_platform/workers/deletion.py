from __future__ import annotations

import asyncio
import signal
from uuid import UUID

import structlog
from sqlalchemy import select

from rag_platform.adapters.cache import CacheStore
from rag_platform.adapters.events import EventConsumer
from rag_platform.adapters.graph_store import GraphStore
from rag_platform.adapters.object_store import ObjectStore
from rag_platform.adapters.vector_store import VectorStore
from rag_platform.config import get_settings
from rag_platform.db.models import DocumentVersion
from rag_platform.db.session import SessionFactory
from rag_platform.db.tenant import set_tenant_context
from rag_platform.logging import configure_logging


async def delete_projections(payload: dict[str, object]) -> None:
    settings = get_settings()
    tenant_id = UUID(str(payload["tenant_id"]))
    document_id = UUID(str(payload["document_id"]))
    corpus_id = UUID(str(payload["corpus_id"]))
    raw_version_ids = payload.get("version_ids", [])
    if not isinstance(raw_version_ids, list):
        raise ValueError("Deletion event version_ids must be a list")
    version_ids = [UUID(str(value)) for value in raw_version_ids]
    object_store = ObjectStore(settings)
    vectors = VectorStore(settings)
    cache = CacheStore(settings)
    graph = GraphStore(settings) if settings.neo4j_enabled else None

    async with SessionFactory() as session, session.begin():
        await set_tenant_context(session, tenant_id)
        versions = list(
            (
                await session.scalars(
                    select(DocumentVersion).where(
                        DocumentVersion.tenant_id == tenant_id,
                        DocumentVersion.version_id.in_(version_ids),
                    )
                )
            ).all()
        )
        object_keys = [version.object_key for version in versions]

    for version_id in version_ids:
        await vectors.delete_version(tenant_id, version_id)
    for key in object_keys:
        await object_store.delete(settings.s3_clean_bucket, key)
    await object_store.delete_prefix(
        settings.s3_derived_bucket, f"derived/{tenant_id}/{document_id}/"
    )
    if graph:
        await graph.delete_document(tenant_id, document_id)
        await graph.close()
    await cache.bump_corpus_epoch(tenant_id, corpus_id)
    await cache.close()


async def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    logger = structlog.get_logger(__name__)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)

    async with EventConsumer(
        settings, "document.delete.requested.v1", "rag-deletion-v1"
    ) as consumer:
        async for payload in consumer.events():
            if stop.is_set():
                break
            try:
                await delete_projections(payload)
            except Exception:
                logger.exception("document_deletion_failed", payload=payload)
                await asyncio.sleep(5)
                continue
            await consumer.commit()


if __name__ == "__main__":
    asyncio.run(main())
