from __future__ import annotations


def _normalize_document_state(raw_state: str) -> str:
    state = raw_state.strip().upper()
    if state in ("ACTIVE", "READY"):
        return "ready"
    if state == "FAILED":
        return "failed"
    if state == "PROCESSING":
        return "processing"
    return "queued"


def test_normalize_active_state_maps_to_ready() -> None:
    assert _normalize_document_state("ACTIVE") == "ready"
    assert _normalize_document_state("active") == "ready"


def test_normalize_failed_state_maps_to_failed() -> None:
    assert _normalize_document_state("FAILED") == "failed"
    assert _normalize_document_state("failed") == "failed"


def test_normalize_unknown_state_maps_to_queued() -> None:
    assert _normalize_document_state("UPLOADING") == "queued"
    assert _normalize_document_state("quarantined") == "queued"
    assert _normalize_document_state("DELETED") == "queued"
