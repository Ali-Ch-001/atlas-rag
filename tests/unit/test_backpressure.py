from __future__ import annotations

import pytest

from rag_platform.adapters.backpressure import BackpressureController, ConcurrencyGuard, TokenBucket
from rag_platform.config import Settings


class TestConcurrencyGuard:
    async def test_acquire_release_limits_concurrency(self) -> None:
        guard = ConcurrencyGuard(name="test", maximum=2)
        acquired1 = await guard.acquire()
        assert acquired1 is True
        assert guard._in_flight == 1
        acquired2 = await guard.acquire()
        assert acquired2 is True
        assert guard._in_flight == 2
        assert guard.available == 0
        guard.release()
        assert guard._in_flight == 1
        assert guard.available == 1
        guard.release()
        assert guard._in_flight == 0


class TestTokenBucket:
    async def test_consumes_tokens_and_refills(self) -> None:
        bucket = TokenBucket(rate_per_second=10.0, burst=5)
        assert bucket.available_tokens == pytest.approx(5.0)
        consumed = await bucket.consume(3)
        assert consumed is True
        assert bucket.available_tokens == pytest.approx(2.0, rel=0.02)

    async def test_consumes_up_to_burst(self) -> None:
        bucket = TokenBucket(rate_per_second=1000.0, burst=1)
        consumed = await bucket.consume(1)
        assert consumed is True
        assert bucket.available_tokens < 1.0


class TestBackpressureController:
    def test_snapshot_shows_all_fields(self) -> None:
        settings = Settings(embedding_max_concurrency=20)
        ctrl = BackpressureController(settings)
        snap = ctrl.snapshot()
        assert "embedding_in_flight" in snap
        assert "embedding_available" in snap
        assert "embedding_max" in snap
        assert "qdrant_write_tokens" in snap
        assert "qdrant_write_rate" in snap
        assert "ingestion_paused" in snap
        assert "retrieval_p95_ms" in snap
        assert snap["embedding_max"] == 20

    def test_embedding_guard_has_correct_default_max(self) -> None:
        settings = Settings(embedding_max_concurrency=48)
        ctrl = BackpressureController(settings)
        assert ctrl.embedding_guard.maximum == 48
