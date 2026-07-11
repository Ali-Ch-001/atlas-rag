from __future__ import annotations

import time

import pytest

from rag_platform.adapters.backpressure import (
    BackpressureController,
    CircuitBreaker,
    CircuitState,
    ConcurrencyGuard,
    TokenBucket,
)
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


class TestCircuitBreaker:
    def test_initial_state_is_closed(self) -> None:
        cb = CircuitBreaker(name="test")
        assert cb.state == CircuitState.CLOSED
        assert cb.consecutive_failures == 0

    def test_before_call_allows_when_closed(self) -> None:
        cb = CircuitBreaker(name="test")
        assert cb.before_call() is True

    def test_opens_after_consecutive_failures(self) -> None:
        cb = CircuitBreaker(name="test", failure_threshold=5)
        for _ in range(5):
            cb.on_failure()
        assert cb.state == CircuitState.OPEN
        assert cb.consecutive_failures == 5
        assert cb.before_call() is False

    def test_half_opens_after_recovery_timeout(self) -> None:
        cb = CircuitBreaker(name="test", failure_threshold=3, recovery_timeout=0.01)
        for _ in range(3):
            cb.on_failure()
        assert cb.state == CircuitState.OPEN
        time.sleep(0.02)
        assert cb.before_call() is True
        assert cb.state == CircuitState.HALF_OPEN

    def test_closes_after_successes_in_half_open(self) -> None:
        cb = CircuitBreaker(
            name="test",
            failure_threshold=2,
            recovery_timeout=0.01,
            success_threshold=2,
        )
        for _ in range(2):
            cb.on_failure()
        assert cb.state == CircuitState.OPEN
        time.sleep(0.02)
        assert cb.before_call() is True
        assert cb.state == CircuitState.HALF_OPEN
        cb.on_success()
        assert cb.state == CircuitState.HALF_OPEN
        cb.on_success()
        assert cb.state == CircuitState.CLOSED
        assert cb.consecutive_failures == 0

    def test_opens_on_failure_in_half_open(self) -> None:
        cb = CircuitBreaker(name="test", failure_threshold=2, recovery_timeout=0.01)
        for _ in range(2):
            cb.on_failure()
        assert cb.state == CircuitState.OPEN
        time.sleep(0.02)
        assert cb.before_call() is True
        assert cb.state == CircuitState.HALF_OPEN
        cb.on_failure()
        assert cb.state == CircuitState.OPEN
        assert cb.consecutive_failures == 1

    def test_failure_mid_stream_resets_when_not_at_threshold(self) -> None:
        cb = CircuitBreaker(name="test", failure_threshold=10)
        for _ in range(3):
            cb.on_failure()
        assert cb.state == CircuitState.CLOSED
        cb.on_success()
        assert cb.consecutive_failures == 0

    def test_snapshot_contains_all_fields(self) -> None:
        cb = CircuitBreaker(name="test")
        cb.on_failure()
        snap = cb.snapshot()
        assert snap["name"] == "test"
        assert snap["state"] == CircuitState.CLOSED.name
        assert snap["consecutive_failures"] == 1
        assert snap["total_failures"] == 1
        assert snap["total_successes"] == 0
        assert "last_failure_time" in snap
        assert "last_state_change" in snap

    def test_consecutive_successes_not_counted_when_closed(self) -> None:
        cb = CircuitBreaker(name="test")
        cb.on_success()
        cb.on_success()
        assert cb.total_successes == 2
        assert cb.consecutive_successes == 0
        assert cb.state == CircuitState.CLOSED


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

    def test_circuit_breakers_created(self) -> None:
        settings = Settings()
        ctrl = BackpressureController(settings)
        assert ctrl.qdrant_circuit.name == "qdrant"
        assert ctrl.redis_circuit.name == "redis"
        assert ctrl.openai_circuit.name == "openai"
        assert ctrl.qdrant_circuit.state == CircuitState.CLOSED
        assert ctrl.redis_circuit.state == CircuitState.CLOSED
        assert ctrl.openai_circuit.state == CircuitState.CLOSED

    def test_snapshot_includes_circuits(self) -> None:
        settings = Settings()
        ctrl = BackpressureController(settings)
        snap = ctrl.snapshot()
        assert "circuits" in snap
        assert snap["circuits"]["qdrant"]["state"] == CircuitState.CLOSED.name
        assert snap["circuits"]["redis"]["name"] == "redis"
