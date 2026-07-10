from __future__ import annotations

from uuid import uuid4

from rag_platform.adapters.graph_store import GraphHit
from rag_platform.adapters.vector_store import DenseHit
from rag_platform.config import Settings
from rag_platform.services.retrieval import RetrievalService, SparseHit


def _make_service(rrf_k: int = 60, fusion_candidates: int = 20) -> RetrievalService:
    service = object.__new__(RetrievalService)
    service.settings = Settings(rrf_k=rrf_k, fusion_candidates=fusion_candidates)
    return service


def test_rrf_with_sparse_dense_and_graph_hits() -> None:
    service = _make_service()
    c1 = uuid4()
    c2 = uuid4()
    c3 = uuid4()
    fused = service._rrf(
        [SparseHit(c1, 0.9), SparseHit(c2, 0.5)],
        [DenseHit(c2, 0.8), DenseHit(c3, 0.7)],
        [GraphHit(c3, 0.6), GraphHit(c1, 0.4)],
    )
    ids = [item[0] for item in fused]
    assert c1 in ids
    assert c2 in ids
    assert c3 in ids
    scores = {item[0]: item[1] for item in fused}
    assert scores[c1] == 1.0 / (60 + 1) + 1.0 / (60 + 2)
    assert scores[c2] == 1.0 / (60 + 2) + 1.0 / (60 + 1)
    assert scores[c3] == 1.0 / (60 + 2) + 1.0 / (60 + 1)


def test_rrf_with_empty_dense_preserves_sparse() -> None:
    service = _make_service()
    c1 = uuid4()
    c2 = uuid4()
    fused = service._rrf(
        [SparseHit(c1, 0.9), SparseHit(c2, 0.5)],
        [],
    )
    assert len(fused) == 2
    scores = {item[0]: item[1] for item in fused}
    assert scores[c1] == 1.0 / (60 + 1)
    assert scores[c2] == 1.0 / (60 + 2)


def test_rrf_ordering_highest_score_first() -> None:
    service = _make_service()
    c1 = uuid4()
    c2 = uuid4()
    c3 = uuid4()
    fused = service._rrf(
        [SparseHit(c1, 0.9), SparseHit(c2, 0.5), SparseHit(c3, 0.2)],
        [DenseHit(c3, 0.8)],
    )
    scores = [item[1] for item in fused]
    assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))
