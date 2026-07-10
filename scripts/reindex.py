"""Reindex all active documents under a new pipeline version.

Usage:
    uv run python -m rag_platform.scripts.reindex \
        --tenant-id 00000000-0000-0000-0000-000000000001 \
        --pipeline-version 2026-07-15.1

This script reads every ACTIVE DocumentVersion, downloads its source PDF from
clean object storage, re-processes it through the current pipeline version, and
creates new version rows. Old versions remain searchable during processing.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
from datetime import UTC, datetime
from uuid import UUID, uuid4

import structlog
from sqlalchemy import select

from rag_platform.adapters.cache import CacheStore
from rag_platform.adapters.embeddings import create_embedding_provider
from rag_platform.adapters.graph_store import GraphStore
from rag_platform.adapters.object_store import ObjectStore
from rag_platform.adapters.vector_store import VectorStore
from rag_platform.config import get_settings
from rag_platform.db.models import (
    Chunk,
    Document,
    DocumentVersion,
    OutboxEvent,
    Section,
)
from rag_platform.db.session import SessionFactory
from rag_platform.db.tenant import set_tenant_context
from rag_platform.domain.models import DocumentState
from rag_platform.ingestion.chunker import ChunkerConfig, SemanticChunker
from rag_platform.ingestion.parser import parse_pdf
from rag_platform.ingestion.scanner import DocumentScanner
from rag_platform.ingestion.service import PARSER_VERSION, PIPELINE_VERSION
from rag_platform.logging import configure_logging

logger = structlog.get_logger(__name__)


async def reindex_tenant(tenant_id: UUID, pipeline_version: str) -> dict[str, int]:
    settings = get_settings()
    object_store = ObjectStore(settings)
    embeddings = create_embedding_provider(settings)
    vector_store = VectorStore(settings)
    cache = CacheStore(settings)
    graph = GraphStore(settings) if settings.neo4j_enabled else None
    scanner = DocumentScanner(settings)
    chunker = SemanticChunker(
        ChunkerConfig(
            target_tokens=settings.chunk_target_tokens,
            max_tokens=settings.chunk_max_tokens,
            overlap_tokens=settings.chunk_overlap_tokens,
            document_max_tokens=settings.chunker_document_max_tokens,
        )
    )

    stats = {"processed": 0, "skipped": 0, "failed": 0}

    async with SessionFactory() as session, session.begin():
        await set_tenant_context(session, tenant_id)
        versions = list(
            (
                await session.scalars(
                    select(DocumentVersion).where(
                        DocumentVersion.tenant_id == tenant_id,
                        DocumentVersion.state == DocumentState.active.value,
                    )
                )
            ).all()
        )

    logger.info("reindex_starting", tenant_id=str(tenant_id), version_count=len(versions))

    for old_version in versions:
        try:
            existing = False
            async with SessionFactory() as session, session.begin():
                await set_tenant_context(session, tenant_id)
                row = await session.execute(
                    select(DocumentVersion).where(
                        DocumentVersion.tenant_id == tenant_id,
                        DocumentVersion.document_id == old_version.document_id,
                        DocumentVersion.pipeline_version == pipeline_version,
                    )
                )
                existing = row.scalar_one_or_none() is not None

            if existing:
                stats["skipped"] += 1
                continue

            content = await object_store.download_bytes(
                settings.s3_clean_bucket, old_version.object_key
            )
            actual_hash = hashlib.sha256(content).hexdigest()
            if actual_hash != old_version.source_sha256.hex():
                logger.error(
                    "reindex_hash_mismatch",
                    version_id=str(old_version.version_id),
                    document_id=str(old_version.document_id),
                )
                stats["failed"] += 1
                continue

            await scanner.scan(content)
            parsed = await asyncio.to_thread(parse_pdf, content, settings.max_pdf_pages)
            drafts = chunker.chunk(parsed)
            if not drafts:
                stats["failed"] += 1
                continue

            new_version_id = uuid4()
            async with SessionFactory() as session, session.begin():
                await set_tenant_context(session, tenant_id)

                last_version = (
                    await session.execute(
                        select(DocumentVersion.version_number)
                        .where(
                            DocumentVersion.tenant_id == tenant_id,
                            DocumentVersion.document_id == old_version.document_id,
                        )
                        .order_by(DocumentVersion.version_number.desc())
                        .limit(1)
                    )
                ).scalar_one_or_none()

                new_version = DocumentVersion(
                    tenant_id=tenant_id,
                    version_id=new_version_id,
                    document_id=old_version.document_id,
                    version_number=(last_version or 0) + 1,
                    source_sha256=old_version.source_sha256,
                    object_key=old_version.object_key,
                    source_date=old_version.source_date,
                    pipeline_version=pipeline_version,
                    parser_version=PARSER_VERSION,
                    state=DocumentState.processing.value,
                    page_count=parsed.page_count,
                )
                session.add(new_version)

            sections = _build_sections(tenant_id, new_version_id, old_version, drafts)
            chunks = _build_chunks(tenant_id, old_version, new_version_id, sections, drafts)

            async with SessionFactory() as session, session.begin():
                await set_tenant_context(session, tenant_id)
                session.add_all(sections)
                session.add_all(chunks)

            inputs = [
                "\n".join(filter(None, [chunk.title, chunk.heading_path, chunk.content]))
                for chunk in chunks
            ]
            vectors: list[list[float]] = []
            for start in range(0, len(inputs), 128):
                vectors.extend(await embeddings.embed_documents(inputs[start : start + 128]))
            await vector_store.upsert_chunks(chunks, vectors)

            if graph:
                await graph.index_chunks(chunks)

            async with SessionFactory() as session, session.begin():
                await set_tenant_context(session, tenant_id)
                new_version = await session.get(
                    DocumentVersion,
                    {"tenant_id": tenant_id, "version_id": new_version_id},
                )
                document = await session.get(
                    Document,
                    {"tenant_id": tenant_id, "document_id": old_version.document_id},
                )
                if new_version and document:
                    new_version.state = DocumentState.active.value
                    new_version.token_count = sum(c.token_count for c in chunks)
                    document.current_version_id = new_version_id
                session.add(
                    OutboxEvent(
                        tenant_id=tenant_id,
                        topic="document.ready.v1",
                        event_key=str(old_version.document_id),
                        payload={
                            "event_type": "document.ready.v1",
                            "tenant_id": str(tenant_id),
                            "document_id": str(old_version.document_id),
                            "version_id": str(new_version_id),
                            "pipeline_version": pipeline_version,
                            "activated_at": datetime.now(UTC).isoformat(),
                        },
                    )
                )

            await cache.bump_corpus_epoch(
                tenant_id,
                old_version.corpus_id if hasattr(old_version, "corpus_id") else UUID(int=0),
            )

            stats["processed"] += 1
            logger.info(
                "reindex_version_complete",
                document_id=str(old_version.document_id),
                new_version_id=str(new_version_id),
            )
        except Exception:
            logger.exception(
                "reindex_version_failed",
                version_id=str(old_version.version_id),
                document_id=str(old_version.document_id),
            )
            stats["failed"] += 1

    if graph:
        await graph.close()
    await cache.close()

    logger.info("reindex_complete", tenant_id=str(tenant_id), **stats)
    return stats


def _build_sections(tenant_id, version_id, old_version, drafts):
    from collections import defaultdict
    from uuid import uuid4

    from rag_platform.db.models import Section

    pages_by_heading: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for draft in drafts:
        key = tuple(draft.heading_path)
        pages_by_heading[key].extend([draft.page_from, draft.page_to])

    section_map: dict[tuple[str, ...], UUID] = {}
    sections: list[Section] = []
    for ordinal, (heading_path, pages) in enumerate(pages_by_heading.items()):
        section_id = uuid4()
        section_map[heading_path] = section_id
        sections.append(
            Section(
                tenant_id=tenant_id,
                section_id=section_id,
                version_id=version_id,
                heading=heading_path[-1] if heading_path else old_version.document_type,
                heading_path=list(heading_path),
                section_level=max(1, len(heading_path)),
                ordinal=ordinal,
                page_from=min(pages),
                page_to=max(pages),
            )
        )
    return sections


def _build_chunks(tenant_id, old_version, new_version_id, sections, drafts):
    from uuid import uuid4

    section_map = {tuple(s.heading_path): s.section_id for s in sections}

    return [
        Chunk(
            tenant_id=tenant_id,
            chunk_id=uuid4(),
            corpus_id=getattr(old_version, "corpus_id", UUID(int=0)),
            document_id=old_version.document_id,
            version_id=new_version_id,
            section_id=section_map.get(tuple(draft.heading_path)),
            title=old_version.document_type,
            heading_path=" > ".join(draft.heading_path) or None,
            content=draft.content,
            content_sha256=bytes.fromhex(draft.content_sha256),
            ordinal=draft.ordinal,
            page_from=draft.page_from,
            page_to=draft.page_to,
            token_count=draft.token_count,
            language="en",
            document_type=old_version.document_type
            if hasattr(old_version, "document_type")
            else "document",
            source_date=old_version.source_date,
            acl_groups=[],
            classification=0,
            source_spans=[span.model_dump(mode="json") for span in draft.source_spans],
        )
        for draft in drafts
    ]


def main():
    parser = argparse.ArgumentParser(description="Reindex documents under a new pipeline version.")
    parser.add_argument("--tenant-id", required=True, type=UUID, help="Target tenant UUID")
    parser.add_argument("--pipeline-version", default=PIPELINE_VERSION)
    args = parser.parse_args()

    configure_logging(get_settings().log_level)
    stats = asyncio.run(reindex_tenant(args.tenant_id, args.pipeline_version))
    print(f"Reindex complete: {stats}")


if __name__ == "__main__":
    main()
