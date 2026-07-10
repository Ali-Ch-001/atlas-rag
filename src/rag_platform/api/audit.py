from __future__ import annotations

from uuid import UUID

import structlog

_audit_logger = structlog.get_logger("rag_platform.audit")


def audit_document_access(tenant_id: UUID, document_id: UUID, user_id: str, action: str) -> None:
    _audit_logger.info(
        "document_access",
        tenant_id=str(tenant_id),
        document_id=str(document_id),
        user_id=user_id,
        action=action,
    )


def audit_search(tenant_id: UUID, user_id: str, query_hash: str) -> None:
    _audit_logger.info(
        "search",
        tenant_id=str(tenant_id),
        user_id=user_id,
        query_hash=query_hash,
    )


def audit_delete(tenant_id: UUID, document_id: UUID, user_id: str) -> None:
    _audit_logger.info(
        "document_delete",
        tenant_id=str(tenant_id),
        document_id=str(document_id),
        user_id=user_id,
    )
