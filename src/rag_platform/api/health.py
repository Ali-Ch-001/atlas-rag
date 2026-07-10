from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import Integer, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from rag_platform.adapters.cache import CacheStore
from rag_platform.adapters.vector_store import VectorStore
from rag_platform.api.dependencies import get_cache_store, get_vector_store
from rag_platform.db.models import Chunk, Document, DocumentVersion, RetrievalRequestLog
from rag_platform.db.session import get_session
from rag_platform.db.tenant import set_tenant_context
from rag_platform.security.auth import AuthContext, require_permission

router = APIRouter(tags=["health"])


@router.get("/health/live")
async def live() -> dict[str, str]:
    return {"status": "ok", "time": datetime.now(UTC).isoformat()}


@router.get("/health/ready")
async def ready(
    session: AsyncSession = Depends(get_session),
    cache: CacheStore = Depends(get_cache_store),
    vectors: VectorStore = Depends(get_vector_store),
) -> dict[str, object]:
    checks: dict[str, object] = {}
    try:
        await session.execute(text("SELECT 1"))
        checks["postgres"] = "healthy"
    except Exception:
        checks["postgres"] = "unreachable"
    try:
        checks["redis"] = "healthy" if await cache.ping() else "unreachable"
    except Exception:
        checks["redis"] = "unreachable"
    try:
        await vectors.client.get_collections()
        checks["qdrant"] = "healthy"
    except Exception:
        checks["qdrant"] = "unreachable"
    try:
        await object_store.get_object("not-a-real-key-check", "health-check")
        checks["minio"] = "healthy"
    except Exception as exc:
        error = str(exc)
        if any(code in error.lower() for code in ("nosuchkey", "not found", "404", "no such")):
            checks["minio"] = "healthy"
        else:
            checks["minio"] = "unreachable"
    healthy = [v for v in checks.values() if v == "healthy"]
    if len(healthy) < 3:
        raise HTTPException(
            status_code=503,
            detail={"status": "not_ready", "checks": checks, "healthy_count": len(healthy)},
        )
    return {"status": "ready", "checks": checks, "healthy_count": len(healthy)}


@router.get("/metrics/summary")
async def metrics_summary(
    auth: AuthContext = Depends(require_permission("documents:read")),
    session: AsyncSession = Depends(get_session),
    vectors: VectorStore = Depends(get_vector_store),
) -> dict[str, object]:
    await set_tenant_context(session, auth.tenant_id)
    now = datetime.now(UTC)
    day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    history_start = now - timedelta(hours=24)
    documents = await session.scalar(
        select(func.count())
        .select_from(Document)
        .where(Document.tenant_id == auth.tenant_id, Document.deleted_at.is_(None))
    )
    chunks = await session.scalar(
        select(func.count())
        .select_from(Chunk)
        .where(Chunk.tenant_id == auth.tenant_id, Chunk.deleted_at.is_(None))
    )
    corpora = await session.scalar(
        select(func.count(func.distinct(Document.corpus_id))).where(
            Document.tenant_id == auth.tenant_id, Document.deleted_at.is_(None)
        )
    )
    queries_today = await session.scalar(
        select(func.count())
        .select_from(RetrievalRequestLog)
        .where(
            RetrievalRequestLog.tenant_id == auth.tenant_id,
            RetrievalRequestLog.created_at >= day_start,
        )
    )
    ingestion_last_hour = await session.scalar(
        select(func.count())
        .select_from(DocumentVersion)
        .where(
            DocumentVersion.tenant_id == auth.tenant_id,
            DocumentVersion.state == "ACTIVE",
            DocumentVersion.updated_at >= now - timedelta(hours=1),
        )
    )
    percentile_row = (
        await session.execute(
            select(
                func.percentile_cont(0.50)
                .within_group(RetrievalRequestLog.total_latency_ms)
                .label("p50"),
                func.percentile_cont(0.95)
                .within_group(RetrievalRequestLog.total_latency_ms)
                .label("p95"),
                func.avg(func.cast(RetrievalRequestLog.cache_hit, Integer)).label("cache_rate"),
            ).where(
                RetrievalRequestLog.tenant_id == auth.tenant_id,
                RetrievalRequestLog.created_at >= history_start,
            )
        )
    ).one()
    retrieval_series_rows = (
        await session.execute(
            select(
                func.date_trunc("hour", RetrievalRequestLog.created_at).label("bucket"),
                func.avg(RetrievalRequestLog.total_latency_ms).label("latency"),
            )
            .where(
                RetrievalRequestLog.tenant_id == auth.tenant_id,
                RetrievalRequestLog.created_at >= history_start,
            )
            .group_by("bucket")
            .order_by("bucket")
        )
    ).all()
    ingestion_series_rows = (
        await session.execute(
            select(
                func.date_trunc("hour", DocumentVersion.updated_at).label("bucket"),
                func.count().label("documents"),
            )
            .where(
                DocumentVersion.tenant_id == auth.tenant_id,
                DocumentVersion.state == "ACTIVE",
                DocumentVersion.updated_at >= history_start,
            )
            .group_by("bucket")
            .order_by("bucket")
        )
    ).all()
    vector_stats = await vectors.stats()
    state_rows = (
        await session.execute(
            select(DocumentVersion.state, func.count())
            .where(DocumentVersion.tenant_id == auth.tenant_id)
            .group_by(DocumentVersion.state)
        )
    ).all()
    states: dict[str, int] = {state: int(count) for state, count in state_rows}
    return {
        "source": "live",
        "corpus_count": corpora or 0,
        "documents": documents or 0,
        "chunks": chunks or 0,
        "index_size_gb": vector_stats.estimated_size_gb,
        "indexed_vectors": vector_stats.indexed_vectors_count,
        "document_states": states,
        "queries_today": queries_today or 0,
        "ingestion_per_hour": ingestion_last_hour or 0,
        "retrieval_p50_ms": round(float(percentile_row.p50 or 0), 2),
        "retrieval_p95_ms": round(float(percentile_row.p95 or 0), 2),
        "cache_hit_rate": round(float(percentile_row.cache_rate or 0), 4),
        "throughput": [
            {"time": bucket.isoformat(), "value": int(count)}
            for bucket, count in ingestion_series_rows
        ],
        "latency": [
            {"time": bucket.isoformat(), "value": round(float(value), 2)}
            for bucket, value in retrieval_series_rows
        ],
        "generated_at": now.isoformat(),
    }
