from __future__ import annotations

import asyncio
import hashlib
import hmac
import time
from contextlib import nullcontext
from dataclasses import dataclass
from uuid import UUID, uuid4

import structlog
from opentelemetry import trace
from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_platform.adapters.backpressure import BackpressureController
from rag_platform.adapters.cache import CacheStore
from rag_platform.adapters.embeddings import EmbeddingProvider
from rag_platform.adapters.graph_store import GraphHit
from rag_platform.adapters.vector_store import DenseHit, VectorStore
from rag_platform.api.audit import audit_search
from rag_platform.config import Settings
from rag_platform.db.models import Chunk, Document, RetrievalLog, RetrievalRequestLog
from rag_platform.db.tenant import set_tenant_context
from rag_platform.domain.models import Citation, SearchRequest, SearchResponse, SearchResult
from rag_platform.retrieval.context import extractive_compress, select_mmr
from rag_platform.retrieval.reranker import RerankCandidate, Reranker
from rag_platform.security.auth import AuthContext
from rag_platform.telemetry import CACHE_HITS, RETRIEVAL_DURATION, RETRIEVAL_REQUESTS

logger = structlog.get_logger(__name__)
tracer = trace.get_tracer(__name__)


@dataclass(frozen=True, slots=True)
class SparseHit:
    chunk_id: UUID
    score: float


class RetrievalService:
    def __init__(
        self,
        settings: Settings,
        embeddings: EmbeddingProvider,
        vectors: VectorStore,
        cache: CacheStore,
        reranker: Reranker,
        backpressure: BackpressureController | None = None,
    ) -> None:
        self.settings = settings
        self.embeddings = embeddings
        self.vectors = vectors
        self.cache = cache
        self.reranker = reranker
        self.backpressure = backpressure

    def _span(self, name: str):  # type: ignore[no-untyped-def]
        if self.settings.otel_enabled:
            return trace.use_span(tracer.start_span(name), end_on_exit=True)
        return nullcontext()

    async def search(
        self,
        session: AsyncSession,
        auth: AuthContext,
        request: SearchRequest,
        graph_hits: list[GraphHit] | None = None,
    ) -> SearchResponse:
        started = time.perf_counter()
        await set_tenant_context(session, auth.tenant_id)
        epoch = await self.cache.corpus_epoch(auth.tenant_id, request.filters.corpus_ids)
        acl_fingerprint = hashlib.sha256("\x1f".join(sorted(auth.groups)).encode()).hexdigest()[:24]
        graph_scope = "graph" if graph_hits is not None else "hybrid"
        scope_hash = hashlib.sha256(
            request.filters.model_dump_json().encode() + epoch.encode() + graph_scope.encode()
        ).hexdigest()
        canonical_query = " ".join(request.query.lower().split())
        query_hash = hashlib.sha256(canonical_query.encode()).hexdigest()
        exact_key = hashlib.sha256(
            (
                f"{auth.tenant_id}|{acl_fingerprint}|{scope_hash}|{canonical_query}|"
                f"{graph_scope}|v1"
            ).encode()
        ).hexdigest()
        cache_available = self.backpressure is None or self.backpressure.redis_circuit.before_call()
        if cache_available:
            try:
                if cached := await self.cache.get_exact(exact_key):
                    CACHE_HITS.labels("exact").inc()
                    response = SearchResponse.model_validate_json(cached)
                    if self.backpressure:
                        self.backpressure.redis_circuit.on_success()
                    return await self._record_cache_hit(
                        session, auth, response, query_hash, started, "exact"
                    )
            except Exception as exc:
                if self.backpressure:
                    self.backpressure.redis_circuit.on_failure()
                    logger.warning("redis_cache_exact_failed", error=str(exc))
        else:
            logger.warning("redis_circuit_open_skipping_cache")

        query_vector = None
        if cache_available:
            try:
                query_vector = await self.cache.get_embedding(
                    self.embeddings.model_name, query_hash
                )
            except Exception:
                if self.backpressure:
                    self.backpressure.redis_circuit.on_failure()
        if query_vector is None:
            query_vector = await self.embeddings.embed_query(request.query)
            if cache_available:
                try:
                    await self.cache.set_embedding(
                        self.embeddings.model_name, query_hash, query_vector
                    )
                except Exception:
                    if self.backpressure:
                        self.backpressure.redis_circuit.on_failure()

        if cache_available:
            try:
                semantic = await self.cache.get_semantic(
                    tenant_id=auth.tenant_id,
                    acl_fingerprint=acl_fingerprint,
                    scope_hash=scope_hash,
                    query_vector=query_vector,
                )
            except Exception:
                if self.backpressure:
                    self.backpressure.redis_circuit.on_failure()
                semantic = None
            if semantic:
                CACHE_HITS.labels("semantic").inc()
                response = SearchResponse.model_validate_json(semantic.payload)
                response.timings_ms["semantic_cache_similarity"] = semantic.similarity
                if self.backpressure:
                    self.backpressure.redis_circuit.on_success()
                return await self._record_cache_hit(
                    session, auth, response, query_hash, started, "semantic"
                )

        sparse_task = asyncio.create_task(self._sparse_search(session, auth, request))
        dense_available = (
            self.backpressure is None or self.backpressure.qdrant_circuit.before_call()
        )
        if dense_available:
            dense_task = asyncio.create_task(
                self.vectors.search(
                    query_vector,
                    auth,
                    request.filters.corpus_ids,
                    request.filters.document_types,
                    request.filters.date_from,
                    request.filters.date_to,
                    self.settings.dense_candidates,
                )
            )
        else:
            logger.warning("qdrant_circuit_open_skipping_dense")

            async def _empty_dense() -> list[DenseHit]:
                return []

            dense_task = asyncio.create_task(_empty_dense())
        sparse_hits: list[SparseHit] = []
        dense_hits: list[DenseHit] = []
        partial = False
        dense_start = time.perf_counter()
        results = await asyncio.gather(sparse_task, dense_task, return_exceptions=True)
        dense_latency = (time.perf_counter() - dense_start) * 1000
        if isinstance(results[0], BaseException):
            partial = True
            logger.exception("sparse_retrieval_failed", exc_info=results[0])
        else:
            sparse_hits = results[0]
        if isinstance(results[1], BaseException):
            partial = True
            logger.exception("dense_retrieval_failed", exc_info=results[1])
            if self.backpressure:
                self.backpressure.qdrant_circuit.on_failure()
        else:
            dense_hits = results[1]
            if self.backpressure and dense_available:
                self.backpressure.qdrant_circuit.on_success()
        if not sparse_hits and not dense_hits and partial:
            raise RuntimeError("All retrieval backends failed")
        if self.settings.otel_enabled:
            span = tracer.start_span("sparse_search")
            span.set_attributes(
                {
                    "sparse_hit_count": len(sparse_hits),
                    "sparse_failed": isinstance(results[0], BaseException),
                }
            )
            span.end()
            span = tracer.start_span("dense_search")
            span.set_attributes(
                {
                    "dense_hit_count": len(dense_hits),
                    "dense_latency_ms": dense_latency,
                    "dense_failed": isinstance(results[1], BaseException),
                }
            )
            span.end()

        with self._span("rrf_fusion") as rrf_span:
            fused = self._rrf(sparse_hits, dense_hits, graph_hits)
            if rrf_span is not None:
                rrf_span.set_attributes(
                    {
                        "sparse_count": len(sparse_hits),
                        "dense_count": len(dense_hits),
                        "graph_count": len(graph_hits) if graph_hits else 0,
                        "fused_count": len(fused),
                    }
                )

        chunk_ids = [item[0] for item in fused]
        with self._span("candidate_hydration") as hydrate_span:
            chunks = await self._hydrate(session, auth, chunk_ids)
            if hydrate_span is not None:
                hydrate_span.set_attribute("hydrated_count", len(chunks))

        chunk_map = {chunk.chunk_id: chunk for chunk in chunks}
        sparse_scores = {item.chunk_id: item.score for item in sparse_hits}
        dense_scores = {item.chunk_id: item.score for item in dense_hits}
        candidates = [
            RerankCandidate(
                chunk_id=str(chunk_id),
                content=chunk_map[chunk_id].content,
                title=chunk_map[chunk_id].title,
                heading_path=chunk_map[chunk_id].heading_path,
                dense_score=dense_scores.get(chunk_id),
                sparse_score=sparse_scores.get(chunk_id),
                rrf_score=score,
            )
            for chunk_id, score in fused
            if chunk_id in chunk_map
        ]

        with self._span("rerank") as rerank_span:
            reranked = await self.reranker.rerank(request.query, candidates)
            if rerank_span is not None:
                rerank_span.set_attribute("reranker_candidate_count", len(reranked))

        offset = (request.filters.page - 1) * request.filters.page_size
        page_candidates = reranked[offset : offset + request.filters.page_size]

        with self._span("mmr_select") as mmr_span:
            selected = select_mmr(
                page_candidates,
                min(request.top_k, self.settings.final_context_chunks),
                self.settings.mmr_lambda,
                lambda item: str(chunk_map[UUID(item.chunk_id)].document_id),
            )
            if mmr_span is not None:
                mmr_span.set_attributes(
                    {
                        "mmr_input_count": len(page_candidates),
                        "mmr_selected_count": len(selected),
                        "mmr_lambda": self.settings.mmr_lambda,
                    }
                )

        with self._span("extractive_compress") as compress_span:
            response_results = [
                self._result(request.query, item, chunk_map[UUID(item.chunk_id)], rank)
                for rank, item in enumerate(selected, start=1)
            ]
            if compress_span is not None:
                compress_span.set_attribute("compressed_count", len(response_results))

        request_id = uuid4()
        total_latency_ms = round((time.perf_counter() - started) * 1000, 2)
        session.add_all(
            [
                RetrievalLog(
                    request_id=request_id,
                    tenant_id=auth.tenant_id,
                    query_hash=query_hash,
                    chunk_id=UUID(item.chunk_id),
                    dense_score=item.dense_score,
                    sparse_score=item.sparse_score,
                    rrf_score=item.rrf_score,
                    reranker_score=item.reranker_score,
                    final_rank=(selected.index(item) + 1) if item in selected else None,
                    metadata_json={"partial": partial, "model": self.settings.reranker_model},
                )
                for item in reranked
            ]
            + [
                RetrievalRequestLog(
                    request_id=request_id,
                    tenant_id=auth.tenant_id,
                    query_hash=query_hash,
                    total_latency_ms=total_latency_ms,
                    cache_hit=False,
                    cache_type=None,
                    partial=partial,
                    result_count=len(response_results),
                )
            ]
        )
        await session.commit()
        outcome = "partial" if partial else "success"
        RETRIEVAL_REQUESTS.labels(outcome).inc()
        RETRIEVAL_DURATION.observe(total_latency_ms / 1000.0)
        audit_search(auth.tenant_id, auth.subject_id, query_hash)
        response = SearchResponse(
            request_id=request_id,
            results=response_results,
            partial=partial,
            timings_ms={"total": total_latency_ms},
            total_count=len(reranked),
            page=request.filters.page,
            page_size=request.filters.page_size,
        )
        payload = response.model_dump_json()
        if cache_available:
            try:
                await self.cache.set_exact(exact_key, payload)
                await self.cache.set_semantic(
                    tenant_id=auth.tenant_id,
                    acl_fingerprint=acl_fingerprint,
                    scope_hash=scope_hash,
                    query_vector=query_vector,
                    payload=payload,
                )
                if self.backpressure:
                    self.backpressure.redis_circuit.on_success()
            except Exception:
                if self.backpressure:
                    self.backpressure.redis_circuit.on_failure()
                    logger.warning("redis_cache_write_failed")
        return response

    async def _record_cache_hit(
        self,
        session: AsyncSession,
        auth: AuthContext,
        response: SearchResponse,
        query_hash: str,
        started: float,
        cache_type: str,
    ) -> SearchResponse:
        response.request_id = uuid4()
        response.cache_hit = True
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        response.timings_ms = {**response.timings_ms, "total": latency_ms}
        session.add(
            RetrievalRequestLog(
                request_id=response.request_id,
                tenant_id=auth.tenant_id,
                query_hash=query_hash,
                total_latency_ms=latency_ms,
                cache_hit=True,
                cache_type=cache_type,
                partial=response.partial,
                result_count=len(response.results),
            )
        )
        await session.commit()
        return response

    async def _sparse_search(
        self, session: AsyncSession, auth: AuthContext, request: SearchRequest
    ) -> list[SparseHit]:
        query = func.websearch_to_tsquery("english", request.query)
        score = func.ts_rank_cd(Chunk.search_tsv, query).label("score")
        conditions = [
            Chunk.tenant_id == auth.tenant_id,
            Chunk.corpus_id.in_(request.filters.corpus_ids),
            Chunk.deleted_at.is_(None),
            Chunk.classification <= auth.clearance,
            Chunk.search_tsv.op("@@")(query),
            Document.current_version_id == Chunk.version_id,
            Document.deleted_at.is_(None),
        ]
        if request.filters.document_types:
            conditions.append(Chunk.document_type.in_(request.filters.document_types))
        if request.filters.date_from:
            conditions.append(Chunk.source_date >= request.filters.date_from)
        if request.filters.date_to:
            conditions.append(Chunk.source_date <= request.filters.date_to)
        if request.filters.language:
            conditions.append(Chunk.language == request.filters.language)
        if auth.groups:
            conditions.append(
                or_(
                    func.cardinality(Chunk.acl_groups) == 0,
                    Chunk.acl_groups.overlap(list(auth.groups)),
                )
            )
        else:
            conditions.append(func.cardinality(Chunk.acl_groups) == 0)
        rows = (
            await session.execute(
                select(Chunk.chunk_id, score)
                .join(
                    Document,
                    (Document.tenant_id == Chunk.tenant_id)
                    & (Document.document_id == Chunk.document_id),
                )
                .where(*conditions)
                .order_by(score.desc())
                .limit(self.settings.sparse_candidates)
            )
        ).all()
        return [SparseHit(chunk_id=row[0], score=float(row[1])) for row in rows]

    def _rrf(
        self,
        sparse: list[SparseHit],
        dense: list[DenseHit],
        graph: list[GraphHit] | None = None,
    ) -> list[tuple[UUID, float]]:
        scores: dict[UUID, float] = {}
        for rank, sparse_hit in enumerate(sparse, start=1):
            scores[sparse_hit.chunk_id] = scores.get(sparse_hit.chunk_id, 0.0) + 1.0 / (
                self.settings.rrf_k + rank
            )
        for rank, dense_hit in enumerate(dense, start=1):
            scores[dense_hit.chunk_id] = scores.get(dense_hit.chunk_id, 0.0) + 1.0 / (
                self.settings.rrf_k + rank
            )
        for rank, graph_hit in enumerate(graph or [], start=1):
            scores[graph_hit.chunk_id] = scores.get(graph_hit.chunk_id, 0.0) + 1.0 / (
                self.settings.rrf_k + rank
            )
        return sorted(scores.items(), key=lambda item: item[1], reverse=True)[
            : self.settings.fusion_candidates
        ]

    async def _hydrate(
        self, session: AsyncSession, auth: AuthContext, chunk_ids: list[UUID]
    ) -> list[Chunk]:
        if not chunk_ids:
            return []
        return list(
            (
                await session.scalars(
                    select(Chunk)
                    .join(
                        Document,
                        (Document.tenant_id == Chunk.tenant_id)
                        & (Document.document_id == Chunk.document_id),
                    )
                    .where(
                        Chunk.tenant_id == auth.tenant_id,
                        Chunk.chunk_id.in_(chunk_ids),
                        Chunk.deleted_at.is_(None),
                        Document.current_version_id == Chunk.version_id,
                        Document.deleted_at.is_(None),
                    )
                )
            ).all()
        )

    def _result(
        self, query: str, candidate: RerankCandidate, chunk: Chunk, rank: int
    ) -> SearchResult:
        source_material = (
            f"{chunk.tenant_id}|{chunk.document_id}|{chunk.version_id}|{chunk.chunk_id}|"
            f"{chunk.content_sha256.hex()}"
        )
        source_id = hmac.new(
            self.settings.citation_hmac_secret.encode(),
            source_material.encode(),
            hashlib.sha256,
        ).hexdigest()[:32]
        citation = Citation(
            citation_id=f"C{rank}",
            source_id=f"src_{source_id}",
            document_id=chunk.document_id,
            version_id=chunk.version_id,
            title=chunk.title,
            section=chunk.heading_path,
            pages=list(range(chunk.page_from, chunk.page_to + 1)),
            source_spans=chunk.source_spans,
            content_sha256=chunk.content_sha256.hex(),
        )
        return SearchResult(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            version_id=chunk.version_id,
            content=chunk.content,
            compressed_content=extractive_compress(query, chunk.content, chunk.token_count),
            title=chunk.title,
            heading_path=chunk.heading_path,
            page_from=chunk.page_from,
            page_to=chunk.page_to,
            dense_score=candidate.dense_score,
            sparse_score=candidate.sparse_score,
            rrf_score=candidate.rrf_score,
            reranker_score=candidate.reranker_score,
            citation=citation,
        )
