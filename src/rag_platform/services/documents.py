from __future__ import annotations

import hashlib
from datetime import UTC, date, datetime
from tempfile import SpooledTemporaryFile
from typing import BinaryIO, cast
from uuid import UUID, uuid4

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_platform.adapters.object_store import ObjectStore
from rag_platform.api.audit import audit_delete, audit_document_access
from rag_platform.config import Settings
from rag_platform.db.models import Document, DocumentVersion, OutboxEvent
from rag_platform.db.tenant import set_tenant_context
from rag_platform.domain.models import (
    DocumentListResponse,
    DocumentResponse,
    DocumentState,
    IngestionEvent,
)
from rag_platform.ingestion.service import PARSER_VERSION, PIPELINE_VERSION
from rag_platform.security.auth import AuthContext


class UploadTooLargeError(ValueError):
    pass


async def spool_and_hash(
    file: BinaryIO, max_bytes: int
) -> tuple[SpooledTemporaryFile[bytes], str, int]:
    spool = SpooledTemporaryFile(max_size=min(max_bytes, 16 * 1024 * 1024), mode="w+b")
    digest = hashlib.sha256()
    size = 0
    while True:
        chunk = file.read(1024 * 1024)
        if not chunk:
            break
        size += len(chunk)
        if size > max_bytes:
            spool.close()
            raise UploadTooLargeError(f"Upload exceeds {max_bytes} bytes")
        digest.update(chunk)
        spool.write(chunk)
    spool.seek(0)
    return spool, digest.hexdigest(), size


class DocumentService:
    def __init__(self, settings: Settings, object_store: ObjectStore) -> None:
        self.settings = settings
        self.object_store = object_store

    async def create(
        self,
        session: AsyncSession,
        auth: AuthContext,
        file: BinaryIO,
        *,
        corpus_id: UUID,
        title: str,
        document_type: str,
        source_date: date | None,
    ) -> DocumentResponse:
        await set_tenant_context(session, auth.tenant_id)
        spool, source_hash, _ = await spool_and_hash(file, self.settings.max_upload_bytes)
        try:
            duplicate = await session.execute(
                select(DocumentVersion, Document)
                .join(
                    Document,
                    (Document.tenant_id == DocumentVersion.tenant_id)
                    & (Document.document_id == DocumentVersion.document_id),
                )
                .where(
                    DocumentVersion.tenant_id == auth.tenant_id,
                    DocumentVersion.source_sha256 == bytes.fromhex(source_hash),
                    DocumentVersion.state == DocumentState.active.value,
                    Document.corpus_id == corpus_id,
                    Document.deleted_at.is_(None),
                )
                .limit(1)
            )
            duplicate_row = duplicate.first()
            if duplicate_row:
                version, document = duplicate_row
                return self._response(document, version, deduplicated=True)

            document_id = uuid4()
            version_id = uuid4()
            object_key = f"quarantine/{auth.tenant_id}/{document_id}/{version_id}/{source_hash}.pdf"
            await self.object_store.upload_fileobj(
                self.settings.s3_quarantine_bucket,
                object_key,
                cast(BinaryIO, spool),
                content_type="application/pdf",
                metadata={
                    "tenant-id": str(auth.tenant_id),
                    "document-id": str(document_id),
                    "version-id": str(version_id),
                    "sha256": source_hash,
                },
            )
            document = Document(
                tenant_id=auth.tenant_id,
                document_id=document_id,
                corpus_id=corpus_id,
                current_version_id=None,
                document_type=document_type,
                title=title,
                classification=0,
            )
            version = DocumentVersion(
                tenant_id=auth.tenant_id,
                version_id=version_id,
                document_id=document_id,
                version_number=1,
                source_sha256=bytes.fromhex(source_hash),
                object_key=object_key,
                source_date=source_date,
                pipeline_version=PIPELINE_VERSION,
                parser_version=PARSER_VERSION,
                state=DocumentState.quarantined.value,
            )
            event = IngestionEvent(
                event_id=uuid4(),
                event_type="document.accepted.v1",
                tenant_id=auth.tenant_id,
                corpus_id=corpus_id,
                document_id=document_id,
                version_id=version_id,
                object_key=object_key,
                source_sha256=source_hash,
                document_type=document_type,
                title=title,
                source_date=source_date,
                created_at=datetime.now(UTC),
            )
            session.add_all(
                [
                    document,
                    version,
                    OutboxEvent(
                        tenant_id=auth.tenant_id,
                        event_id=event.event_id,
                        topic=event.event_type,
                        event_key=str(version_id),
                        payload=event.model_dump(mode="json"),
                    ),
                ]
            )
            await session.commit()
            audit_document_access(
                tenant_id=auth.tenant_id,
                document_id=document_id,
                user_id=auth.subject_id,
                action="upload",
            )
            return self._response(document, version)
        finally:
            spool.close()

    async def list(
        self,
        session: AsyncSession,
        auth: AuthContext,
        *,
        corpus_id: UUID | None,
        state: DocumentState | None,
        query: str | None,
        limit: int,
        offset: int,
    ) -> DocumentListResponse:
        await set_tenant_context(session, auth.tenant_id)
        conditions = [Document.tenant_id == auth.tenant_id, Document.deleted_at.is_(None)]
        if corpus_id:
            conditions.append(Document.corpus_id == corpus_id)
        if query:
            conditions.append(Document.title.contains(query.replace("%", "")))
        latest_version_id = (
            select(DocumentVersion.version_id)
            .where(
                DocumentVersion.tenant_id == Document.tenant_id,
                DocumentVersion.document_id == Document.document_id,
            )
            .order_by(DocumentVersion.version_number.desc())
            .limit(1)
            .correlate(Document)
            .scalar_subquery()
        )
        statement = (
            select(Document, DocumentVersion)
            .join(
                DocumentVersion,
                (DocumentVersion.tenant_id == Document.tenant_id)
                & (DocumentVersion.version_id == latest_version_id),
            )
            .where(*conditions)
            .order_by(Document.created_at.desc())
        )
        if state:
            statement = statement.where(DocumentVersion.state == state.value)
        total = await session.scalar(
            select(func.count()).select_from(statement.order_by(None).subquery())
        )
        rows = (await session.execute(statement.limit(limit).offset(offset))).all()
        return DocumentListResponse(
            items=[self._response(document, version) for document, version in rows],
            total=total or 0,
            limit=limit,
            offset=offset,
        )

    async def get(
        self, session: AsyncSession, auth: AuthContext, document_id: UUID
    ) -> DocumentResponse | None:
        await set_tenant_context(session, auth.tenant_id)
        row = (
            await session.execute(
                select(Document, DocumentVersion)
                .join(
                    DocumentVersion,
                    (DocumentVersion.tenant_id == Document.tenant_id)
                    & (DocumentVersion.document_id == Document.document_id),
                )
                .where(
                    Document.tenant_id == auth.tenant_id,
                    Document.document_id == document_id,
                    Document.deleted_at.is_(None),
                )
                .order_by(DocumentVersion.version_number.desc())
                .limit(1)
            )
        ).first()
        return self._response(*row) if row else None

    async def delete(self, session: AsyncSession, auth: AuthContext, document_id: UUID) -> bool:
        await set_tenant_context(session, auth.tenant_id)
        document = await session.get(
            Document, {"tenant_id": auth.tenant_id, "document_id": document_id}
        )
        if document is None or document.deleted_at is not None:
            return False
        document.deleted_at = datetime.now(UTC)
        versions = list(
            (
                await session.scalars(
                    select(DocumentVersion).where(
                        DocumentVersion.tenant_id == auth.tenant_id,
                        DocumentVersion.document_id == document_id,
                    )
                )
            ).all()
        )
        for version in versions:
            version.state = DocumentState.deleted.value
        session.add(
            OutboxEvent(
                tenant_id=auth.tenant_id,
                topic="document.delete.requested.v1",
                event_key=str(document_id),
                payload={
                    "event_type": "document.delete.requested.v1",
                    "tenant_id": str(auth.tenant_id),
                    "document_id": str(document_id),
                    "corpus_id": str(document.corpus_id),
                    "version_ids": [str(version.version_id) for version in versions],
                    "deleted_at": document.deleted_at.isoformat(),
                },
            )
        )
        await session.commit()
        audit_delete(
            tenant_id=auth.tenant_id,
            document_id=document_id,
            user_id=auth.subject_id,
        )
        return True

    @staticmethod
    def _response(
        document: Document, version: DocumentVersion, *, deduplicated: bool = False
    ) -> DocumentResponse:
        return DocumentResponse(
            document_id=document.document_id,
            version_id=version.version_id,
            corpus_id=document.corpus_id,
            title=document.title,
            document_type=document.document_type,
            state=DocumentState(version.state),
            source_date=version.source_date,
            page_count=version.page_count,
            token_count=version.token_count,
            created_at=version.created_at,
            deduplicated=deduplicated,
        )
