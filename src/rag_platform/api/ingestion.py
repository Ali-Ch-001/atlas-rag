from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_platform.db.models import DocumentVersion, IngestionStage
from rag_platform.db.session import get_session
from rag_platform.db.tenant import set_tenant_context
from rag_platform.security.auth import AuthContext, require_permission

router = APIRouter(prefix="/v1/ingestion", tags=["ingestion"])


@router.get("/summary")
async def ingestion_summary(
    auth: AuthContext = Depends(require_permission("documents:read")),
    session: AsyncSession = Depends(get_session),
) -> dict[str, object]:
    await set_tenant_context(session, auth.tenant_id)

    now = datetime.now(UTC)
    last_24h = now.replace(hour=0, minute=0, second=0, microsecond=0)

    active = await session.scalar(
        select(func.count())
        .select_from(DocumentVersion)
        .where(
            DocumentVersion.tenant_id == auth.tenant_id,
            DocumentVersion.state == "ACTIVE",
        )
    )

    state_rows = (
        await session.execute(
            select(DocumentVersion.state, func.count())
            .where(DocumentVersion.tenant_id == auth.tenant_id)
            .group_by(DocumentVersion.state)
        )
    ).all()
    states: dict[str, int] = {state: int(count) for state, count in state_rows}

    stage_counts_result = (
        await session.execute(
            select(IngestionStage.stage_name, IngestionStage.state, func.count())
            .where(IngestionStage.tenant_id == auth.tenant_id)
            .group_by(IngestionStage.stage_name, IngestionStage.state)
        )
    ).all()

    stages: dict[str, dict[str, int]] = {}
    for stage_name, state, count in stage_counts_result:
        entry = stages.setdefault(stage_name, {"RUNNING": 0, "FAILED": 0, "COMPLETED": 0})
        entry[str(state)] = count

    today_active = await session.scalar(
        select(func.count())
        .select_from(DocumentVersion)
        .where(
            DocumentVersion.tenant_id == auth.tenant_id,
            DocumentVersion.state == "ACTIVE",
            DocumentVersion.updated_at >= last_24h,
        )
    )

    failed_24h = await session.scalar(
        select(func.count())
        .select_from(DocumentVersion)
        .where(
            DocumentVersion.tenant_id == auth.tenant_id,
            DocumentVersion.state == "FAILED",
            DocumentVersion.updated_at >= last_24h,
        )
    )

    failures = (
        await session.execute(
            select(
                DocumentVersion.document_id,
                DocumentVersion.error_code,
                DocumentVersion.error_detail,
                DocumentVersion.updated_at,
            )
            .where(
                DocumentVersion.tenant_id == auth.tenant_id,
                DocumentVersion.state == "FAILED",
            )
            .order_by(DocumentVersion.updated_at.desc())
            .limit(20)
        )
    ).all()

    return {
        "source": "live",
        "active_documents": active or 0,
        "document_states": {str(k): int(v) for k, v in states.items()},
        "queue_depth": {stage: entry.get("RUNNING", 0) for stage, entry in stages.items()},
        "retry_queue": {stage: entry.get("FAILED", 0) for stage, entry in stages.items()},
        "today_processed": today_active or 0,
        "failed_24h": failed_24h or 0,
        "recent_failures": [
            {
                "job_id": str(document_id)[:8],
                "stage": error_code or "unknown",
                "reason": (error_detail or "Unspecified error")[:200],
                "at": updated_at.isoformat() if updated_at else None,
            }
            for document_id, error_code, error_detail, updated_at in failures
        ],
        "pipeline_stages": [
            {
                "name": stage_name,
                "count": stages.get(stage_name, {}).get("RUNNING", 0),
                "completed": stages.get(stage_name, {}).get("COMPLETED", 0),
                "failed": stages.get(stage_name, {}).get("FAILED", 0),
            }
            for stage_name in (
                "quarantine",
                "scan",
                "parse",
                "ocr",
                "chunk",
                "embed",
                "index",
                "full_pipeline",
            )
            if stage_name in stages
        ],
        "generated_at": now.isoformat(),
    }
