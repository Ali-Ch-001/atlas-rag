from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import anyio

from rag_platform.config import Settings


@dataclass(slots=True)
class RerankCandidate:
    chunk_id: str
    content: str
    title: str
    heading_path: str | None
    dense_score: float | None
    sparse_score: float | None
    rrf_score: float
    reranker_score: float = 0.0


def terms(text: str) -> set[str]:
    return {term.lower() for term in re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]+", text)}


class Reranker(ABC):
    @abstractmethod
    async def rerank(self, query: str, candidates: list[RerankCandidate]) -> list[RerankCandidate]:
        raise NotImplementedError


class LexicalReranker(Reranker):
    def __init__(self, settings: Settings) -> None:
        self._term_weight = settings.reranker_term_weight
        self._title_weight = settings.reranker_title_weight
        self._rrf_weight = settings.reranker_rrf_weight
        self._rrf_scale = settings.reranker_rrf_scale

    async def rerank(self, query: str, candidates: list[RerankCandidate]) -> list[RerankCandidate]:
        query_terms = terms(query)
        for candidate in candidates:
            title_terms = terms(f"{candidate.title} {candidate.heading_path or ''}")
            content_terms = terms(candidate.content)
            overlap = len(query_terms & content_terms) / max(1, len(query_terms))
            title_overlap = len(query_terms & title_terms) / max(1, len(query_terms))
            candidate.reranker_score = (
                self._term_weight * overlap
                + self._title_weight * title_overlap
                + self._rrf_weight * math.tanh(candidate.rrf_score * self._rrf_scale)
            )
        return sorted(candidates, key=lambda item: item.reranker_score, reverse=True)


class CrossEncoderReranker(Reranker):
    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import CrossEncoder  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "Cross-encoder dependencies are missing; run `uv sync --extra ml`"
            ) from exc
        self._model: Any = CrossEncoder(model_name, activation_fn=None)

    async def rerank(self, query: str, candidates: list[RerankCandidate]) -> list[RerankCandidate]:
        pairs = [
            (query, f"{item.title}\n{item.heading_path or ''}\n{item.content}")
            for item in candidates
        ]
        scores = await anyio.to_thread.run_sync(self._model.predict, pairs)
        for candidate, score in zip(candidates, scores, strict=True):
            candidate.reranker_score = float(score)
        return sorted(candidates, key=lambda item: item.reranker_score, reverse=True)


def create_reranker(settings: Settings) -> Reranker:
    if settings.reranker_provider == "cross_encoder":
        return CrossEncoderReranker(settings.reranker_model)
    return LexicalReranker(settings)
