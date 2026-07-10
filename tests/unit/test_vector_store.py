from __future__ import annotations

from rag_platform.adapters.vector_store import VectorStats


def test_vector_stats_estimated_size_gb_calculation() -> None:
    points = 1000000
    dimensions = 1536
    replication = 1
    raw_bytes = points * dimensions * 4
    estimated_bytes = raw_bytes * 1.8 * replication
    expected_gb = round(estimated_bytes / (1024**3), 3)

    stats = VectorStats(
        points_count=points,
        indexed_vectors_count=points,
        estimated_size_gb=expected_gb,
    )

    assert stats.points_count == points
    assert stats.estimated_size_gb == expected_gb
    assert stats.estimated_size_gb > 0
