from __future__ import annotations

import asyncio
import hashlib
import hmac
import re
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from rag_platform.adapters.cache import CacheStore
from rag_platform.adapters.graph_store import GraphStore
from rag_platform.adapters.llm import GenerationProvider
from rag_platform.adapters.web_search import TavilySearch
from rag_platform.agent.router import Route, SemanticRouter, calculate
from rag_platform.api.audit import audit_search
from rag_platform.config import Settings
from rag_platform.domain.models import AgentRequest, SearchFilters, SearchRequest
from rag_platform.security.auth import AuthContext
from rag_platform.security.guardrails import GuardrailViolation, inspect_input, inspect_output
from rag_platform.services.retrieval import RetrievalService


class AgentService:
    def __init__(
        self,
        settings: Settings,
        router: SemanticRouter,
        retrieval: RetrievalService,
        web: TavilySearch,
        generator: GenerationProvider,
        cache: CacheStore,
        graph: GraphStore | None = None,
    ) -> None:
        self.settings = settings
        self.router = router
        self.retrieval = retrieval
        self.web = web
        self.generator = generator
        self.cache = cache
        self.graph = graph

    async def stream(
        self,
        session: AsyncSession,
        auth: AuthContext,
        request: AgentRequest,
        response_id: UUID,
    ) -> AsyncIterator[dict[str, Any]]:
        seq = 0

        def event(event_type: str, data: dict[str, Any]) -> dict[str, Any]:
            nonlocal seq
            seq += 1
            return {
                "v": 1,
                "event_id": f"{response_id}:{seq}",
                "seq": seq,
                "type": event_type,
                "response_id": str(response_id),
                "created_at": datetime.now(UTC).isoformat(),
                "data": data,
            }

        yield event(
            "response.created",
            {"response_id": str(response_id), "mode": request.response_mode},
        )
        try:
            inspected = inspect_input(request.message, reject_injection=True)
        except GuardrailViolation as exc:
            yield event(
                "error",
                {"code": "INPUT_GUARDRAIL", "detail": str(exc), "retryable": False},
            )
            yield event("response.completed", {"outcome": "rejected"})
            return

        decision = self.router.route(inspected.text, bool(request.corpus_ids))
        yield event(
            "status",
            {"stage": "routing", "route": decision.route, "reason": decision.reason},
        )

        if decision.route == Route.clarify:
            message = "Select at least one authorized corpus, or ask for a current web search."
            yield event("token", {"delta": message, "char_start": 0})
            yield event("response.completed", {"outcome": "clarification_required"})
            return

        if decision.route == Route.calculator:
            try:
                answer = f"{calculate(inspected.text):,.10g}"
            except (SyntaxError, ValueError, ZeroDivisionError, OverflowError) as exc:
                yield event(
                    "error",
                    {
                        "code": "INVALID_EXPRESSION",
                        "detail": str(exc),
                        "retryable": False,
                    },
                )
                yield event("response.completed", {"outcome": "failed"})
                return
            yield event("token", {"delta": answer, "char_start": 0})
            yield event("usage", {"tool_calls": 1, "input_tokens": 0, "output_tokens": 0})
            yield event("response.completed", {"outcome": "completed"})
            return

        evidence: list[dict[str, str]] = []
        if decision.route in {Route.internal, Route.composite, Route.graph}:
            yield event("status", {"stage": "retrieving"})
            graph_hits = None
            if decision.route == Route.graph:
                if self.graph is None:
                    raise RuntimeError("GraphRAG is not configured")
                yield event("tool.started", {"tool_name": "graph.search", "call_id": "graph_1"})
                graph_hits = await self.graph.search(
                    inspected.text, auth, request.corpus_ids, max_hops=2
                )
                yield event(
                    "tool.completed",
                    {"call_id": "graph_1", "result_count": len(graph_hits)},
                )
            search_response = await self.retrieval.search(
                session,
                auth,
                SearchRequest(
                    query=inspected.text,
                    filters=SearchFilters(corpus_ids=request.corpus_ids),
                    top_k=5,
                ),
                graph_hits=graph_hits,
            )
            audit_search(
                auth.tenant_id,
                auth.subject_id,
                hashlib.sha256(inspected.text.encode()).hexdigest(),
            )
            for result in search_response.results:
                evidence.append(
                    {
                        "citation_id": result.citation.citation_id,
                        "source_id": result.citation.source_id,
                        "title": result.title,
                        "content": result.compressed_content,
                        "locator": f"pages {result.page_from}-{result.page_to}",
                        "document_id": str(result.document_id),
                        "section": result.heading_path or "",
                        "page": str(result.page_from),
                    }
                )

        if decision.route in {Route.web, Route.composite}:
            yield event("tool.started", {"tool_name": "web.search", "call_id": "web_1"})
            web_results = await self.web.search(inspected.text, max_results=5)
            next_number = len(evidence) + 1
            for index, web_result in enumerate(web_results, start=next_number):
                digest = hmac.new(
                    self.settings.citation_hmac_secret.encode(),
                    (
                        f"{web_result.url}|"
                        f"{hashlib.sha256(web_result.content.encode()).hexdigest()}"
                    ).encode(),
                    hashlib.sha256,
                ).hexdigest()[:32]
                evidence.append(
                    {
                        "citation_id": f"C{index}",
                        "source_id": f"web_{digest}",
                        "title": web_result.title,
                        "content": web_result.content,
                        "locator": web_result.url,
                    }
                )
            yield event("tool.completed", {"call_id": "web_1", "result_count": len(web_results)})

        if not evidence:
            message = "I could not find enough authorized evidence to answer reliably."
            yield event("token", {"delta": message, "char_start": 0})
            yield event(
                "response.completed",
                {"outcome": "abstained", "reason": "insufficient_evidence"},
            )
            return

        for item in evidence:
            yield event(
                "source",
                {
                    "source_id": item["source_id"],
                    "citation_id": item["citation_id"],
                    "title": item["title"],
                    "locator": item["locator"],
                    "document_id": item.get("document_id"),
                    "section": item.get("section"),
                    "page": int(item["page"]) if item.get("page") else None,
                },
            )
        yield event("status", {"stage": "generating", "evidence_count": len(evidence)})
        generated = await self.generator.generate(inspected.text, evidence)
        answer = inspect_output(generated.text)
        allowed = {item["citation_id"] for item in evidence}
        referenced = set(re.findall(r"\[(C\d+)\]", answer))
        if not referenced or not referenced.issubset(allowed):
            answer = "I could not validate the answer's citations against the authorized evidence."
            outcome = "abstained"
        else:
            outcome = "completed"

        char_start = 0
        for token in re.findall(r"\S+\s*", answer):
            if await self.cache.is_cancelled(response_id):
                yield event("response.cancelled", {"reason": "client_cancelled"})
                return
            yield event("token", {"delta": token, "char_start": char_start})
            char_start += len(token)
            await asyncio.sleep(0)
        for citation_id in sorted(referenced & allowed):
            source = next(item for item in evidence if item["citation_id"] == citation_id)
            yield event(
                "citation",
                {"citation_id": citation_id, "source_ids": [source["source_id"]]},
            )
        yield event(
            "usage",
            {
                "input_tokens": generated.input_tokens,
                "output_tokens": generated.output_tokens,
                "model": generated.model,
                "tool_calls": int(decision.route in {Route.web, Route.composite})
                + int(decision.route in {Route.internal, Route.composite}),
            },
        )
        yield event("response.completed", {"outcome": outcome})
