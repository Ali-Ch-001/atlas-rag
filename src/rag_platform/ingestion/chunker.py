from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass

import structlog
import tiktoken

from rag_platform.domain.models import ChunkDraft, ExtractionMethod, SourceSpan
from rag_platform.ingestion.schemas import ParsedDocument, TextBlock

logger = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ChunkerConfig:
    target_tokens: int
    max_tokens: int
    overlap_tokens: int
    document_max_tokens: int = 50000


def canonicalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


class SemanticChunker:
    def __init__(self, config: ChunkerConfig) -> None:
        self.config = config
        self.encoder = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        return len(self.encoder.encode(text, disallowed_special=()))

    def _split_oversized(self, block: TextBlock) -> list[TextBlock]:
        if self.count_tokens(block.text) <= self.config.max_tokens:
            return [block]
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", block.text)
        parts: list[str] = []
        current: list[str] = []
        current_tokens = 0
        for sentence in sentences:
            tokens = self.count_tokens(sentence)
            if tokens > self.config.max_tokens:
                encoded = self.encoder.encode(sentence, disallowed_special=())
                for start in range(0, len(encoded), self.config.max_tokens):
                    if current:
                        parts.append(" ".join(current))
                        current = []
                        current_tokens = 0
                    parts.append(
                        self.encoder.decode(encoded[start : start + self.config.max_tokens])
                    )
                continue
            if current and current_tokens + tokens > self.config.max_tokens:
                parts.append(" ".join(current))
                current = []
                current_tokens = 0
            current.append(sentence)
            current_tokens += tokens
        if current:
            parts.append(" ".join(current))
        return [
            TextBlock(
                text=part,
                page=block.page,
                bbox=block.bbox,
                font_size=block.font_size,
                heading_level=None,
                extraction_method=block.extraction_method,
                confidence=block.confidence,
            )
            for part in parts
            if part.strip()
        ]

    def chunk(self, parsed: ParsedDocument) -> list[ChunkDraft]:
        heading_stack: list[str] = []
        expanded: list[tuple[TextBlock, tuple[str, ...]]] = []
        for original in parsed.blocks:
            if original.heading_level is not None:
                level = original.heading_level
                heading_stack = heading_stack[: level - 1]
                heading_stack.append(canonicalize_text(original.text))
                continue
            for block in self._split_oversized(original):
                expanded.append((block, tuple(heading_stack)))

        drafts: list[ChunkDraft] = []
        current: list[TextBlock] = []
        current_headings: tuple[str, ...] = ()
        cumulative_tokens = 0
        budget_exceeded = False
        truncation_marker = "\n\n[TRUNCATED: document token budget exceeded]"

        def flush() -> None:
            nonlocal current, cumulative_tokens, budget_exceeded
            if not current:
                return
            content = canonicalize_text("\n\n".join(block.text for block in current))
            if not content:
                current = []
                return
            canonical_input = "\n".join((*current_headings, content))
            digest = hashlib.sha256(canonical_input.encode()).hexdigest()
            chunk_tokens = self.count_tokens(content)
            if budget_exceeded:
                content = content + truncation_marker
                chunk_tokens = self.count_tokens(content)
            cumulative_tokens += chunk_tokens
            drafts.append(
                ChunkDraft(
                    section_title=current_headings[-1] if current_headings else None,
                    heading_path=list(current_headings),
                    content=content,
                    content_sha256=digest,
                    ordinal=len(drafts),
                    page_from=min(block.page for block in current),
                    page_to=max(block.page for block in current),
                    token_count=chunk_tokens,
                    source_spans=[
                        SourceSpan(
                            page=block.page,
                            bbox=block.bbox,
                            extraction_method=ExtractionMethod(block.extraction_method),
                            confidence=block.confidence,
                        )
                        for block in current
                    ],
                )
            )
            if self.config.overlap_tokens <= 0:
                current = []
                return
            overlap: list[TextBlock] = []
            overlap_count = 0
            for block in reversed(current):
                tokens = self.count_tokens(block.text)
                if overlap and overlap_count + tokens > self.config.overlap_tokens:
                    break
                overlap.insert(0, block)
                overlap_count += tokens
            current = overlap

        for block, headings in expanded:
            if budget_exceeded:
                break
            if current and headings != current_headings:
                flush()
                current = []
            current_headings = headings
            projected = self.count_tokens(
                "\n\n".join([*(item.text for item in current), block.text])
            )
            if current and projected > self.config.target_tokens:
                flush()
            if cumulative_tokens + self.count_tokens(block.text) > self.config.document_max_tokens:
                logger.warning(
                    "chunker_document_token_budget_exceeded",
                    budget=self.config.document_max_tokens,
                    cumulative_tokens=cumulative_tokens,
                    remaining_blocks=len(expanded),
                )
                budget_exceeded = True
                if current:
                    flush()
                break
            current.append(block)
            current_size = self.count_tokens("\n\n".join(item.text for item in current))
            if current_size >= self.config.max_tokens:
                flush()
        flush()

        unique: list[ChunkDraft] = []
        seen: set[str] = set()
        for draft in drafts:
            if draft.content_sha256 in seen:
                continue
            seen.add(draft.content_sha256)
            unique.append(draft.model_copy(update={"ordinal": len(unique)}))
        return unique
