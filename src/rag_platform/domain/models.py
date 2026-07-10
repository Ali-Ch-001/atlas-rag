from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class DocumentState(StrEnum):
    uploading = "UPLOADING"
    quarantined = "QUARANTINED"
    processing = "PROCESSING"
    active = "ACTIVE"
    failed = "FAILED"
    deleted = "DELETED"


class DocumentResponse(BaseModel):
    document_id: UUID
    version_id: UUID
    corpus_id: UUID
    title: str
    document_type: str
    state: DocumentState
    source_date: date | None = None
    page_count: int | None = None
    token_count: int | None = None
    created_at: datetime
    deduplicated: bool = False


class DocumentListResponse(BaseModel):
    items: list[DocumentResponse]
    total: int
    limit: int
    offset: int


class ExtractionMethod(StrEnum):
    native = "native"
    ocr = "ocr"


class SourceSpan(BaseModel):
    page: int = Field(ge=1)
    bbox: tuple[float, float, float, float] | None = None
    char_start: int | None = None
    char_end: int | None = None
    extraction_method: ExtractionMethod = ExtractionMethod.native
    confidence: float | None = Field(default=None, ge=0, le=1)


class ChunkDraft(BaseModel):
    section_title: str | None = None
    heading_path: list[str] = Field(default_factory=list)
    content: str
    content_sha256: str
    ordinal: int = Field(ge=0)
    page_from: int = Field(ge=1)
    page_to: int = Field(ge=1)
    token_count: int = Field(ge=1)
    source_spans: list[SourceSpan]


class IngestionEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: UUID
    event_type: str
    schema_version: int = 1
    tenant_id: UUID
    corpus_id: UUID
    document_id: UUID
    version_id: UUID
    object_key: str
    source_sha256: str
    document_type: str
    title: str
    source_date: date | None = None
    traceparent: str | None = None
    created_at: datetime
    attempt: int = Field(default=1, ge=1, le=10)


class SearchFilters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    corpus_ids: list[UUID] = Field(min_length=1, max_length=8)
    document_types: list[str] = Field(default_factory=list, max_length=10)
    date_from: date | None = None
    date_to: date | None = None
    language: str | None = None
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=10, ge=1, le=100)


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=4000)
    filters: SearchFilters
    top_k: int = Field(default=5, ge=1, le=20)


class Citation(BaseModel):
    citation_id: str
    source_id: str
    document_id: UUID
    version_id: UUID
    title: str
    section: str | None
    pages: list[int]
    source_spans: list[dict[str, Any]]
    content_sha256: str


class SearchResult(BaseModel):
    chunk_id: UUID
    document_id: UUID
    version_id: UUID
    content: str
    compressed_content: str
    title: str
    heading_path: str | None
    page_from: int
    page_to: int
    dense_score: float | None = None
    sparse_score: float | None = None
    rrf_score: float = 0
    reranker_score: float = 0
    citation: Citation


class SearchResponse(BaseModel):
    request_id: UUID
    results: list[SearchResult]
    cache_hit: bool = False
    partial: bool = False
    timings_ms: dict[str, float] = Field(default_factory=dict)
    total_count: int = 0
    page: int = 1
    page_size: int = 10


class AgentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=8000)
    corpus_ids: list[UUID] = Field(default_factory=list, max_length=8)
    response_mode: str = Field(default="grounded", pattern="^(grounded|concise)$")
    disconnect_behavior: str = Field(default="cancel", pattern="^(cancel|continue)$")
