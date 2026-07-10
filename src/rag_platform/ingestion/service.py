from __future__ import annotations

import hashlib
from collections import defaultdict
from datetime import UTC, datetime
from uuid import UUID, uuid4

import anyio
import orjson
import structlog
from sqlalchemy import delete, select

from rag_platform.adapters.cache import CacheStore
from rag_platform.adapters.embeddings import EmbeddingProvider
from rag_platform.adapters.graph_store import GraphStore
from rag_platform.adapters.object_store import ObjectStore
from rag_platform.adapters.vector_store import VectorStore
from rag_platform.config import Settings
from rag_platform.db.models import (
    Chunk,
    Document,
    DocumentVersion,
    IngestionStage,
    OutboxEvent,
    Section,
)
from rag_platform.db.session import SessionFactory
from rag_platform.db.tenant import set_tenant_context
from rag_platform.domain.models import ChunkDraft, DocumentState, IngestionEvent
from rag_platform.ingestion.chunker import ChunkerConfig, SemanticChunker, canonicalize_text
from rag_platform.ingestion.parser import parse_pdf
from rag_platform.ingestion.scanner import DocumentScanner

logger = structlog.get_logger(__name__)
PIPELINE_VERSION = "2026-07-10.1"
PARSER_VERSION = "pymupdf-1"


class IngestionService:
    def __init__(
        self,
        settings: Settings,
        object_store: ObjectStore,
        embeddings: EmbeddingProvider,
        vector_store: VectorStore,
        cache: CacheStore,
        graph: GraphStore | None = None,
    ) -> None:
        self.settings = settings
        self.object_store = object_store
        self.embeddings = embeddings
        self.vector_store = vector_store
        self.cache = cache
        self.graph = graph
        self.scanner = DocumentScanner(settings)
        self.chunker = SemanticChunker(
            ChunkerConfig(
                target_tokens=settings.chunk_target_tokens,
                max_tokens=settings.chunk_max_tokens,
                overlap_tokens=settings.chunk_overlap_tokens,
                document_max_tokens=settings.chunker_document_max_tokens,
            )
        )

    async def process(self, event: IngestionEvent) -> None:
        claimed = await self._claim(event)
        if not claimed:
            logger.info("ingestion_duplicate_skipped", version_id=str(event.version_id))
            return
        try:
            content = await self.object_store.download_bytes(
                self.settings.s3_quarantine_bucket, event.object_key
            )
            actual_hash = hashlib.sha256(content).hexdigest()
            if actual_hash != event.source_sha256:
                raise ValueError("Source object checksum does not match ingestion event")
            await self.scanner.scan(content)

            clean_key = f"clean/{event.tenant_id}/{event.document_id}/{event.version_id}/source.pdf"
            await self.object_store.copy(
                self.settings.s3_quarantine_bucket,
                event.object_key,
                self.settings.s3_clean_bucket,
                clean_key,
            )
            parsed = await anyio.to_thread.run_sync(
                parse_pdf,
                content,
                self.settings.max_pdf_pages,
                self.settings.ocr_enabled,
                self.settings.ocr_dpi,
                self.settings.ocr_language,
            )
            drafts = self.chunker.chunk(parsed)
            if not drafts:
                raise ValueError("Semantic chunking produced no content")

            normalized_hash = hashlib.sha256(
                canonicalize_text("\n".join(draft.content for draft in drafts)).encode()
            ).digest()
            chunks = await self._persist_projection(
                event, clean_key, parsed.page_count, normalized_hash, drafts
            )
            inputs = [
                "\n".join(filter(None, [chunk.title, chunk.heading_path, chunk.content]))
                for chunk in chunks
            ]
            vectors: list[list[float]] = []
            for start in range(0, len(inputs), 128):
                vectors.extend(await self.embeddings.embed_documents(inputs[start : start + 128]))
            await self.vector_store.upsert_chunks(chunks, vectors)
            if self.graph:
                await self.graph.index_chunks(chunks)

            manifest_key = (
                f"derived/{event.tenant_id}/{event.document_id}/{event.version_id}/"
                f"{PIPELINE_VERSION}/manifest.json"
            )
            await self.object_store.upload_bytes(
                self.settings.s3_derived_bucket,
                manifest_key,
                orjson.dumps(
                    {
                        "schema_version": 1,
                        "pipeline_version": PIPELINE_VERSION,
                        "parser_version": PARSER_VERSION,
                        "embedding_model": self.embeddings.model_name,
                        "source_sha256": event.source_sha256,
                        "normalized_text_sha256": normalized_hash.hex(),
                        "page_count": parsed.page_count,
                        "chunk_count": len(chunks),
                        "chunks": [
                            {
                                "chunk_id": str(chunk.chunk_id),
                                "content_sha256": chunk.content_sha256.hex(),
                                "ordinal": chunk.ordinal,
                                "page_from": chunk.page_from,
                                "page_to": chunk.page_to,
                            }
                            for chunk in chunks
                        ],
                    },
                    option=orjson.OPT_INDENT_2,
                ),
                content_type="application/json",
            )
            await self._activate(event, manifest_key)
            await self.cache.bump_corpus_epoch(event.tenant_id, event.corpus_id)
            await self.object_store.delete(self.settings.s3_quarantine_bucket, event.object_key)
            logger.info(
                "ingestion_completed",
                tenant_id=str(event.tenant_id),
                document_id=str(event.document_id),
                version_id=str(event.version_id),
                chunks=len(chunks),
            )
        except Exception as exc:
            await self._fail(event, exc)
            raise

    async def _claim(self, event: IngestionEvent) -> bool:
        async with SessionFactory() as session, session.begin():
            await set_tenant_context(session, event.tenant_id)
            existing = await session.get(
                IngestionStage,
                {
                    "tenant_id": event.tenant_id,
                    "version_id": event.version_id,
                    "stage_name": "full_pipeline",
                    "input_hash": event.source_sha256,
                    "pipeline_version": PIPELINE_VERSION,
                },
            )
            if existing and existing.state == "COMPLETED":
                return False
            if existing:
                existing.state = "RUNNING"
                existing.attempt += 1
                existing.error_detail = None
            else:
                session.add(
                    IngestionStage(
                        tenant_id=event.tenant_id,
                        version_id=event.version_id,
                        stage_name="full_pipeline",
                        input_hash=event.source_sha256,
                        pipeline_version=PIPELINE_VERSION,
                        state="RUNNING",
                    )
                )
            version = await session.get(
                DocumentVersion,
                {"tenant_id": event.tenant_id, "version_id": event.version_id},
            )
            if version:
                version.state = DocumentState.processing.value
            return True

    async def _persist_projection(
        self,
        event: IngestionEvent,
        clean_key: str,
        page_count: int,
        normalized_hash: bytes,
        drafts: list[ChunkDraft],
    ) -> list[Chunk]:
        async with SessionFactory() as session, session.begin():
            await set_tenant_context(session, event.tenant_id)
            await session.execute(
                delete(Chunk).where(
                    Chunk.tenant_id == event.tenant_id, Chunk.version_id == event.version_id
                )
            )
            await session.execute(
                delete(Section).where(
                    Section.tenant_id == event.tenant_id,
                    Section.version_id == event.version_id,
                )
            )
            section_map: dict[tuple[str, ...], UUID] = {}
            sections: list[Section] = []
            pages_by_heading: dict[tuple[str, ...], list[int]] = defaultdict(list)
            for draft in drafts:
                key = tuple(draft.heading_path)
                pages_by_heading[key].extend([draft.page_from, draft.page_to])
            for ordinal, (heading_path, pages) in enumerate(pages_by_heading.items()):
                section_id = uuid4()
                section_map[heading_path] = section_id
                sections.append(
                    Section(
                        tenant_id=event.tenant_id,
                        section_id=section_id,
                        version_id=event.version_id,
                        heading=heading_path[-1] if heading_path else event.title,
                        heading_path=list(heading_path),
                        section_level=max(1, len(heading_path)),
                        ordinal=ordinal,
                        page_from=min(pages),
                        page_to=max(pages),
                    )
                )
            session.add_all(sections)
            chunks = [
                Chunk(
                    tenant_id=event.tenant_id,
                    chunk_id=uuid4(),
                    corpus_id=event.corpus_id,
                    document_id=event.document_id,
                    version_id=event.version_id,
                    section_id=section_map[tuple(draft.heading_path)],
                    title=event.title,
                    heading_path=" > ".join(draft.heading_path) or None,
                    content=draft.content,
                    content_sha256=bytes.fromhex(draft.content_sha256),
                    ordinal=draft.ordinal,
                    page_from=draft.page_from,
                    page_to=draft.page_to,
                    token_count=draft.token_count,
                    language="en",
                    document_type=event.document_type,
                    source_date=event.source_date,
                    acl_groups=[],
                    classification=0,
                    source_spans=[span.model_dump(mode="json") for span in draft.source_spans],
                )
                for draft in drafts
            ]
            session.add_all(chunks)
            version = await session.get(
                DocumentVersion,
                {"tenant_id": event.tenant_id, "version_id": event.version_id},
            )
            if version is None:
                raise ValueError("Document version no longer exists")
            version.object_key = clean_key
            version.page_count = page_count
            version.token_count = sum(chunk.token_count for chunk in chunks)
            version.normalized_text_sha256 = normalized_hash
            version.state = DocumentState.processing.value
            await session.flush()
            return chunks

    async def _activate(self, event: IngestionEvent, manifest_key: str) -> None:
        async with SessionFactory() as session, session.begin():
            await set_tenant_context(session, event.tenant_id)
            version = await session.get(
                DocumentVersion,
                {"tenant_id": event.tenant_id, "version_id": event.version_id},
            )
            document = await session.get(
                Document,
                {"tenant_id": event.tenant_id, "document_id": event.document_id},
            )
            if version is None or document is None:
                raise ValueError("Document was removed before activation")
            version.state = DocumentState.active.value
            document.current_version_id = event.version_id
            stage = await session.get(
                IngestionStage,
                {
                    "tenant_id": event.tenant_id,
                    "version_id": event.version_id,
                    "stage_name": "full_pipeline",
                    "input_hash": event.source_sha256,
                    "pipeline_version": PIPELINE_VERSION,
                },
            )
            if stage:
                stage.state = "COMPLETED"
                stage.output_uri = manifest_key
            session.add(
                OutboxEvent(
                    tenant_id=event.tenant_id,
                    topic="document.ready.v1",
                    event_key=str(event.document_id),
                    payload={
                        "event_type": "document.ready.v1",
                        "tenant_id": str(event.tenant_id),
                        "document_id": str(event.document_id),
                        "version_id": str(event.version_id),
                        "activated_at": datetime.now(UTC).isoformat(),
                    },
                )
            )

    async def _fail(self, event: IngestionEvent, exc: Exception) -> None:
        async with SessionFactory() as session, session.begin():
            await set_tenant_context(session, event.tenant_id)
            version = await session.get(
                DocumentVersion,
                {"tenant_id": event.tenant_id, "version_id": event.version_id},
            )
            if version:
                version.state = DocumentState.failed.value
                version.error_code = type(exc).__name__
                version.error_detail = str(exc)[:2000]
            stage = await session.scalar(
                select(IngestionStage).where(
                    IngestionStage.tenant_id == event.tenant_id,
                    IngestionStage.version_id == event.version_id,
                    IngestionStage.stage_name == "full_pipeline",
                    IngestionStage.input_hash == event.source_sha256,
                    IngestionStage.pipeline_version == PIPELINE_VERSION,
                )
            )
            if stage:
                stage.state = "FAILED"
                stage.error_detail = str(exc)[:2000]
