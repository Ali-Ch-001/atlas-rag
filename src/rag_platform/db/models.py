from __future__ import annotations

from datetime import date, datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import (
    BigInteger,
    Boolean,
    Computed,
    Date,
    DateTime,
    Float,
    ForeignKeyConstraint,
    Index,
    Integer,
    LargeBinary,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column

from rag_platform.db.base import Base


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )


class Document(Base, TimestampMixin):
    __tablename__ = "documents"

    tenant_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    document_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    corpus_id: Mapped[UUID] = mapped_column(Uuid, nullable=False, index=True)
    current_version_id: Mapped[UUID | None] = mapped_column(Uuid)
    document_type: Mapped[str] = mapped_column(String(100), nullable=False)
    title: Mapped[str] = mapped_column(String(1000), nullable=False)
    classification: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=0)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        Index("ix_documents_tenant_corpus", "tenant_id", "corpus_id"),
        UniqueConstraint("tenant_id", "current_version_id"),
    )


class DocumentVersion(Base, TimestampMixin):
    __tablename__ = "document_versions"

    tenant_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    version_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    document_id: Mapped[UUID] = mapped_column(Uuid, nullable=False)
    version_number: Mapped[int] = mapped_column(Integer, nullable=False)
    source_sha256: Mapped[bytes] = mapped_column(LargeBinary(32), nullable=False)
    normalized_text_sha256: Mapped[bytes | None] = mapped_column(LargeBinary(32))
    object_key: Mapped[str] = mapped_column(Text, nullable=False)
    source_date: Mapped[date | None] = mapped_column(Date)
    pipeline_version: Mapped[str] = mapped_column(String(100), nullable=False)
    parser_version: Mapped[str] = mapped_column(String(100), nullable=False)
    state: Mapped[str] = mapped_column(String(30), nullable=False)
    page_count: Mapped[int | None] = mapped_column(Integer)
    token_count: Mapped[int | None] = mapped_column(BigInteger)
    error_code: Mapped[str | None] = mapped_column(String(100))
    error_detail: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (
        ForeignKeyConstraint(
            ["tenant_id", "document_id"],
            ["documents.tenant_id", "documents.document_id"],
            ondelete="CASCADE",
        ),
        UniqueConstraint("tenant_id", "document_id", "version_number"),
        Index("ix_versions_tenant_source_hash", "tenant_id", "source_sha256"),
    )


class Section(Base, TimestampMixin):
    __tablename__ = "sections"

    tenant_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    section_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    version_id: Mapped[UUID] = mapped_column(Uuid, nullable=False)
    parent_section_id: Mapped[UUID | None] = mapped_column(Uuid)
    heading: Mapped[str | None] = mapped_column(Text)
    heading_path: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    section_level: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=1)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    page_from: Mapped[int] = mapped_column(Integer, nullable=False)
    page_to: Mapped[int] = mapped_column(Integer, nullable=False)

    __table_args__ = (
        ForeignKeyConstraint(
            ["tenant_id", "version_id"],
            ["document_versions.tenant_id", "document_versions.version_id"],
            ondelete="CASCADE",
        ),
        Index("ix_sections_tenant_version", "tenant_id", "version_id"),
    )


class Chunk(Base, TimestampMixin):
    __tablename__ = "chunks"

    tenant_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    chunk_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    corpus_id: Mapped[UUID] = mapped_column(Uuid, nullable=False)
    document_id: Mapped[UUID] = mapped_column(Uuid, nullable=False)
    version_id: Mapped[UUID] = mapped_column(Uuid, nullable=False)
    section_id: Mapped[UUID | None] = mapped_column(Uuid)
    title: Mapped[str] = mapped_column(String(1000), nullable=False)
    heading_path: Mapped[str | None] = mapped_column(Text)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_sha256: Mapped[bytes] = mapped_column(LargeBinary(32), nullable=False)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    page_from: Mapped[int] = mapped_column(Integer, nullable=False)
    page_to: Mapped[int] = mapped_column(Integer, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    language: Mapped[str] = mapped_column(String(16), nullable=False, default="en")
    document_type: Mapped[str] = mapped_column(String(100), nullable=False)
    source_date: Mapped[date | None] = mapped_column(Date)
    acl_groups: Mapped[list[str]] = mapped_column(ARRAY(String(255)), nullable=False, default=list)
    classification: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=0)
    source_spans: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False, default=list)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    search_tsv: Mapped[Any] = mapped_column(
        TSVECTOR,
        Computed(
            "setweight(to_tsvector('english', coalesce(heading_path, '')), 'A') || "
            "setweight(to_tsvector('english', content), 'B')",
            persisted=True,
        ),
    )

    __table_args__ = (
        ForeignKeyConstraint(
            ["tenant_id", "version_id"],
            ["document_versions.tenant_id", "document_versions.version_id"],
            ondelete="CASCADE",
        ),
        Index("ix_chunks_tenant_document_version", "tenant_id", "document_id", "version_id"),
        Index(
            "ix_chunks_filters",
            "tenant_id",
            "corpus_id",
            "document_type",
            "source_date",
        ),
        Index("ix_chunks_content_hash", "tenant_id", "content_sha256"),
        Index("ix_chunks_search_tsv", "search_tsv", postgresql_using="gin"),
    )


class IngestionStage(Base, TimestampMixin):
    __tablename__ = "ingestion_stages"

    tenant_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    version_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    stage_name: Mapped[str] = mapped_column(String(50), primary_key=True)
    input_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    pipeline_version: Mapped[str] = mapped_column(String(100), primary_key=True)
    state: Mapped[str] = mapped_column(String(20), nullable=False)
    attempt: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    output_uri: Mapped[str | None] = mapped_column(Text)
    lease_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error_detail: Mapped[str | None] = mapped_column(Text)


class OutboxEvent(Base):
    __tablename__ = "outbox_events"

    event_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    tenant_id: Mapped[UUID] = mapped_column(Uuid, nullable=False, index=True)
    topic: Mapped[str] = mapped_column(String(200), nullable=False)
    event_key: Mapped[str] = mapped_column(String(200), nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        Index(
            "ix_outbox_unpublished",
            "created_at",
            postgresql_where=published_at.is_(None),
        ),
    )


class RetrievalLog(Base):
    __tablename__ = "retrieval_logs"

    log_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    request_id: Mapped[UUID] = mapped_column(Uuid, nullable=False, index=True)
    tenant_id: Mapped[UUID] = mapped_column(Uuid, nullable=False)
    query_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    chunk_id: Mapped[UUID] = mapped_column(Uuid, nullable=False)
    dense_score: Mapped[float | None]
    sparse_score: Mapped[float | None]
    rrf_score: Mapped[float] = mapped_column(nullable=False)
    reranker_score: Mapped[float] = mapped_column(nullable=False)
    final_rank: Mapped[int | None]
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class RetrievalRequestLog(Base):
    __tablename__ = "retrieval_request_logs"

    request_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    tenant_id: Mapped[UUID] = mapped_column(Uuid, nullable=False, index=True)
    query_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    total_latency_ms: Mapped[float] = mapped_column(Float, nullable=False)
    cache_hit: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    cache_type: Mapped[str | None] = mapped_column(String(20))
    partial: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    result_count: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (Index("ix_retrieval_requests_tenant_created", "tenant_id", "created_at"),)
