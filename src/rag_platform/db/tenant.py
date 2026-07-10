from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


async def set_tenant_context(session: AsyncSession, tenant_id: UUID) -> None:
    """Scope RLS policies to one tenant for the duration of the database session.

    Uses session-scoped (not transaction-scoped) set_config so the tenant
    context survives commits and applies to subsequent queries within the
    same session. Each request handler creates a fresh session via get_session,
    so there is no cross-request tenant leakage.
    """
    await session.execute(
        text("SELECT set_config('app.tenant_id', :tenant_id, false)"),
        {"tenant_id": str(tenant_id)},
    )
