from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field
from uuid import UUID, uuid4

import structlog
from sqlalchemy import Float, desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_platform.adapters.llm import GenerationProvider
from rag_platform.config import Settings
from rag_platform.db.models import EvaluationMetric, EvaluationRun
from rag_platform.db.tenant import set_tenant_context
from rag_platform.domain.models import SearchFilters, SearchRequest
from rag_platform.evaluation import golden_dataset
from rag_platform.evaluation.sample_dataset import GoldenCase
from rag_platform.security.auth import AuthContext
from rag_platform.services.retrieval import RetrievalService

logger = structlog.get_logger(__name__)


@dataclass
class PerCaseMetrics:
    question_hash: str
    context_precision: float
    context_recall: float
    faithfulness: float
    answer_relevancy: float
    answer_correctness: float
    retrieved_count: int
    expected_count: int
    latency_ms: float = 0.0


@dataclass
class EvaluationResult:
    run_id: UUID
    dataset_name: str
    corpus_id: UUID
    case_count: int
    status: str
    created_at: str
    completed_at: str | None = None
    case_metrics: list[PerCaseMetrics] = field(default_factory=list)

    @property
    def recall_at_20(self) -> float:
        return self._mean("context_recall")

    @property
    def faithfulness(self) -> float:
        return self._mean("faithfulness")

    @property
    def citation_precision(self) -> float:
        return self._mean("context_precision")

    @property
    def answer_relevancy(self) -> float:
        return self._mean("answer_relevancy")

    @property
    def answer_correctness(self) -> float:
        return self._mean("answer_correctness")

    @property
    def p95_latency_s(self) -> float:
        if not self.case_metrics:
            return 0.0
        sorted_latencies = sorted(m.latency_ms for m in self.case_metrics)
        idx = math.ceil(0.95 * len(sorted_latencies)) - 1
        return sorted_latencies[max(0, idx)] / 1000.0

    def _mean(self, attr: str) -> float:
        if not self.case_metrics:
            return 0.0
        values = [getattr(m, attr) for m in self.case_metrics]
        return float(round(sum(values) / len(values), 4))


@dataclass
class RunSummary:
    run_id: UUID
    dataset_name: str
    corpus_id: UUID
    case_count: int
    status: str
    recall_at_20: float
    faithfulness: float
    citation_precision: float
    answer_relevancy: float
    answer_correctness: float
    p95_latency_s: float
    created_at: str
    completed_at: str | None = None
    trends: list[dict[str, object]] = field(default_factory=list)
    release_gates: list[dict[str, str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _question_hash(question: str) -> str:
    return hashlib.sha256(question.lower().strip().encode()).hexdigest()[:16]


def _token_overlap(retrieved_texts: list[str], expected_texts: list[str]) -> tuple[float, float]:
    if not expected_texts and not retrieved_texts:
        return 1.0, 1.0
    if not expected_texts:
        return 1.0, 0.0
    if not retrieved_texts:
        return 0.0, 0.0

    def tokens(text: str) -> set[str]:
        return set(text.lower().split())

    expected_tokens: set[str] = set()
    for text in expected_texts:
        expected_tokens |= tokens(text)

    retrieved_tokens: set[str] = set()
    for text in retrieved_texts:
        retrieved_tokens |= tokens(text)

    intersection = expected_tokens & retrieved_tokens
    precision = len(intersection) / len(retrieved_tokens) if retrieved_tokens else 0.0
    recall = len(intersection) / len(expected_tokens) if expected_tokens else 0.0
    return round(precision, 4), round(recall, 4)


def _linear_faithfulness_score(answer: str, retrieved_texts: list[str]) -> tuple[float, str]:
    if not retrieved_texts:
        return 0.0, "No evidence available."
    combined = " ".join(retrieved_texts).lower()
    answer_lower = answer.lower()
    answer_words = set(answer_lower.split())
    evidence_words = set(combined.split())
    supported = answer_words & evidence_words
    if not answer_words:
        return 0.0, "Empty answer."
    ratio = len(supported) / len(answer_words)
    score = min(1.0, ratio * 1.3)
    return round(score, 4), ""


def _answer_relevancy_score(question: str, answer: str) -> tuple[float, str]:
    if not answer:
        return 0.0, "Empty answer."
    q_words = set(question.lower().split())
    a_words = set(answer.lower().split())
    overlap = q_words & a_words
    if not a_words:
        return 0.0, "Empty answer."
    ratio = len(overlap) / (len(a_words) ** 0.7)
    score = min(1.0, round(ratio * 0.8, 4))
    return score, ""


def _answer_correctness_score(answer: str, reference: str) -> tuple[float, str]:
    if not reference:
        return 1.0, ""
    a_tokens = set(answer.lower().split())
    r_tokens = set(reference.lower().split())
    if not r_tokens and not a_tokens:
        return 1.0, ""
    if not r_tokens:
        return 0.0, "No reference answer."
    intersection = a_tokens & r_tokens
    recall = len(intersection) / len(r_tokens)
    precision = len(intersection) / len(a_tokens) if a_tokens else 0.0
    if precision + recall == 0:
        return 0.0, "No token overlap."
    f1 = 2 * precision * recall / (precision + recall)
    return round(f1, 4), ""


EVALUATION_FAITHFULNESS_PROMPT = """\
You are an evaluation judge. Score how faithful a generated answer is to the evidence context.

Question: {question}

Evidence context:
{context}

Generated answer: {answer}

Rate faithfulness from 0 to 10:
- 0: The answer contradicts or fabricates information not in the evidence.
- 10: Every factual claim in the answer is directly supported by the evidence.

Respond with ONLY the integer score."""


EVALUATION_RELEVANCY_PROMPT = """\
You are an evaluation judge. Score how relevant a generated answer is to the question.

Question: {question}

Generated answer: {answer}

Rate relevancy from 0 to 10:
- 0: The answer is completely off-topic or unrelated to the question.
- 10: The answer directly and completely addresses the question.

Respond with ONLY the integer score."""


EVALUATION_CORRECTNESS_PROMPT = """\
You are an evaluation judge. Compare a generated answer with a reference answer.

Question: {question}

Generated answer: {answer}

Reference answer: {reference}

Rate correctness from 0 to 10:
- 0: The generated answer is factually wrong or contradicts the reference.
- 10: The generated answer matches the reference in all key factual claims.

Respond with ONLY the integer score."""


class EvaluationService:
    def __init__(
        self,
        settings: Settings,
        retrieval: RetrievalService,
        llm: GenerationProvider,
    ) -> None:
        self.settings = settings
        self.retrieval = retrieval
        self.llm = llm
        self._use_llm_eval = settings.model_provider == "openai"

    def get_dataset(self, dataset_name: str) -> list[GoldenCase]:
        if dataset_name in ("sample-golden", "enterprise-golden-v0"):
            from rag_platform.evaluation.sample_dataset import SAMPLE_DATASET

            return SAMPLE_DATASET
        if dataset_name == "golden-v1":
            return golden_dataset.load_dataset()  # type: ignore[return-value]
        raise ValueError(f"Unknown dataset: {dataset_name}")

    async def run_evaluation(
        self,
        session: AsyncSession,
        auth: AuthContext,
        corpus_id: UUID,
        dataset_name: str = "sample-golden",
    ) -> RunSummary:
        dataset = self.get_dataset(dataset_name)
        run_id = uuid4()
        run = EvaluationRun(
            run_id=run_id,
            tenant_id=auth.tenant_id,
            dataset_name=dataset_name,
            corpus_id=corpus_id,
            case_count=len(dataset),
            status="running",
        )
        session.add(run)
        await session.commit()

        case_metrics: list[PerCaseMetrics] = []
        for case in dataset:
            metrics = await self._evaluate_case(session, auth, corpus_id, case)
            case_metrics.append(metrics)

            session.add(
                EvaluationMetric(
                    run_id=run_id,
                    tenant_id=auth.tenant_id,
                    question_hash=metrics.question_hash,
                    metrics={
                        "context_precision": metrics.context_precision,
                        "context_recall": metrics.context_recall,
                        "faithfulness": metrics.faithfulness,
                        "answer_relevancy": metrics.answer_relevancy,
                        "answer_correctness": metrics.answer_correctness,
                        "retrieved_count": metrics.retrieved_count,
                        "expected_count": metrics.expected_count,
                        "latency_ms": metrics.latency_ms,
                    },
                )
            )

        run.status = "completed"
        run.completed_at = func.now()
        await session.commit()

        result = EvaluationResult(
            run_id=run_id,
            dataset_name=dataset_name,
            corpus_id=corpus_id,
            case_count=len(dataset),
            status="completed",
            created_at=run.created_at.isoformat() if run.created_at else "",
            completed_at=(run.completed_at.isoformat() if run.completed_at else None),
            case_metrics=case_metrics,
        )

        trends = await self._build_trends(session, auth, result)
        notes: list[str] = []
        if not self._use_llm_eval:
            notes.append("requires_openai")
        return RunSummary(
            run_id=result.run_id,
            dataset_name=result.dataset_name,
            corpus_id=result.corpus_id,
            case_count=result.case_count,
            status=result.status,
            recall_at_20=result.recall_at_20,
            faithfulness=result.faithfulness,
            citation_precision=result.citation_precision,
            answer_relevancy=result.answer_relevancy,
            answer_correctness=result.answer_correctness or 0.0,
            p95_latency_s=result.p95_latency_s,
            created_at=result.created_at,
            completed_at=result.completed_at,
            trends=trends,
            release_gates=self._build_release_gates(result),
            notes=notes,
        )

    async def get_latest_results(
        self,
        session: AsyncSession,
        auth: AuthContext,
    ) -> RunSummary | None:
        await set_tenant_context(session, auth.tenant_id)

        latest_run_id = await session.scalar(
            select(EvaluationRun.run_id)
            .where(
                EvaluationRun.tenant_id == auth.tenant_id,
                EvaluationRun.status == "completed",
            )
            .order_by(desc(EvaluationRun.created_at))
            .limit(1)
        )
        if latest_run_id is None:
            return None

        return await self._build_summary(session, auth, latest_run_id)

    async def _evaluate_case(
        self,
        session: AsyncSession,
        auth: AuthContext,
        corpus_id: UUID,
        case: GoldenCase,
    ) -> PerCaseMetrics:
        started = time.perf_counter()
        try:
            search_response = await self.retrieval.search(
                session,
                auth,
                SearchRequest(
                    query=case.question,
                    filters=SearchFilters(corpus_ids=[corpus_id]),
                    top_k=20,
                ),
            )
            results = search_response.results
        except Exception:
            logger.exception("retrieval_failed_during_evaluation")
            results = []
        latency_ms = round((time.perf_counter() - started) * 1000, 2)

        retrieved_texts = [r.content for r in results]
        if hasattr(case, "relevant_chunks"):
            expected_texts = case.relevant_chunks
        elif hasattr(case, "expected_facts"):
            expected_texts = list(case.expected_facts)
        else:
            expected_texts = []

        precision, recall = _token_overlap(retrieved_texts, expected_texts)

        generated_answer = results[0].compressed_content if results else ""
        if not generated_answer:
            for r in results:
                if r.content:
                    generated_answer = r.content
                    break

        if not generated_answer and not retrieved_texts:
            faith_score = -1.0
            relevancy_score = -1.0
            correctness_score = -1.0
        elif self._use_llm_eval:
            faith_score = await self._llm_faithfulness(
                case.question, retrieved_texts, generated_answer
            )
            relevancy_score = await self._llm_relevancy(case.question, generated_answer)
            correctness_score = await self._llm_correctness(
                case.question, generated_answer, case.reference_answer
            )
        else:
            faith_score = -1.0
            relevancy_score = -1.0
            correctness_score = -1.0

        return PerCaseMetrics(
            question_hash=_question_hash(case.question),
            context_precision=precision,
            context_recall=recall,
            faithfulness=faith_score,
            answer_relevancy=relevancy_score,
            answer_correctness=correctness_score,
            retrieved_count=len(results),
            expected_count=len(expected_texts),
            latency_ms=latency_ms,
        )

    async def _llm_faithfulness(self, question: str, context: list[str], answer: str) -> float:
        if not context or not answer:
            return 0.0
        prompt = EVALUATION_FAITHFULNESS_PROMPT.format(
            question=question,
            context="\n\n".join(f"- {c}" for c in context[:10]),
            answer=answer,
        )
        result = await self.llm.generate(prompt, [])
        return self._parse_score(result.text)

    async def _llm_relevancy(self, question: str, answer: str) -> float:
        if not answer:
            return 0.0
        prompt = EVALUATION_RELEVANCY_PROMPT.format(question=question, answer=answer)
        result = await self.llm.generate(prompt, [])
        return self._parse_score(result.text)

    async def _llm_correctness(self, question: str, answer: str, reference: str) -> float:
        if not answer or not reference:
            return 0.0
        prompt = EVALUATION_CORRECTNESS_PROMPT.format(
            question=question, answer=answer, reference=reference
        )
        result = await self.llm.generate(prompt, [])
        return self._parse_score(result.text)

    @staticmethod
    def _parse_score(text: str) -> float:
        import re

        match = re.search(r"\b(\d{1,2})\b", text)
        if match:
            score = int(match.group(1))
            return round(min(max(score, 0), 10) / 10, 4)
        return 0.5

    async def _build_summary(
        self,
        session: AsyncSession,
        auth: AuthContext,
        run_id: UUID,
    ) -> RunSummary:
        run = await session.get(EvaluationRun, run_id)
        if run is None:
            raise ValueError(f"Evaluation run {run_id} not found")

        metrics_rows = (
            await session.scalars(
                select(EvaluationMetric).where(
                    EvaluationMetric.run_id == run_id,
                    EvaluationMetric.tenant_id == auth.tenant_id,
                )
            )
        ).all()

        case_metrics = [
            PerCaseMetrics(
                question_hash=m.question_hash,
                context_precision=float(m.metrics.get("context_precision", 0)),
                context_recall=float(m.metrics.get("context_recall", 0)),
                faithfulness=float(m.metrics.get("faithfulness", 0)),
                answer_relevancy=float(m.metrics.get("answer_relevancy", 0)),
                answer_correctness=float(m.metrics.get("answer_correctness", 0)),
                retrieved_count=int(m.metrics.get("retrieved_count", 0)),
                expected_count=int(m.metrics.get("expected_count", 0)),
                latency_ms=float(m.metrics.get("latency_ms", 0)),
            )
            for m in metrics_rows
        ]

        result = EvaluationResult(
            run_id=run.run_id,
            dataset_name=run.dataset_name,
            corpus_id=run.corpus_id,
            case_count=run.case_count,
            status=run.status,
            created_at=run.created_at.isoformat() if run.created_at else "",
            completed_at=(run.completed_at.isoformat() if run.completed_at else None),
            case_metrics=case_metrics,
        )

        trends = await self._build_trends(session, auth, result)
        return RunSummary(
            run_id=result.run_id,
            dataset_name=result.dataset_name,
            corpus_id=result.corpus_id,
            case_count=result.case_count,
            status=result.status,
            recall_at_20=result.recall_at_20,
            faithfulness=result.faithfulness,
            citation_precision=result.citation_precision,
            answer_relevancy=result.answer_relevancy,
            answer_correctness=result.answer_correctness or 0.0,
            p95_latency_s=result.p95_latency_s,
            created_at=result.created_at,
            completed_at=result.completed_at,
            trends=trends,
            release_gates=self._build_release_gates(result),
            notes=[],
        )

    async def _build_trends(
        self,
        session: AsyncSession,
        auth: AuthContext,
        current: EvaluationResult,
    ) -> list[dict[str, object]]:
        rows = (
            await session.execute(
                select(
                    EvaluationRun.run_id,
                    EvaluationRun.dataset_name,
                    EvaluationRun.created_at,
                )
                .where(
                    EvaluationRun.tenant_id == auth.tenant_id,
                    EvaluationRun.status == "completed",
                )
                .order_by(EvaluationRun.created_at.asc())
                .limit(12)
            )
        ).all()

        trend_data: list[dict[str, object]] = []
        for row in rows:
            run_id = row[0]
            label = str(row[1])[:24]
            if not label:
                label = run_id.hex[:8]
            agg = (
                await session.execute(
                    select(
                        func.avg(
                            EvaluationMetric.metrics["context_recall"].cast(Float).label("avg")
                        )
                    ).where(
                        EvaluationMetric.run_id == run_id,
                        EvaluationMetric.tenant_id == auth.tenant_id,
                    )
                )
            ).scalar()
            recall = round(float(agg) * 100, 1) if agg else 0.0
            trend_data.append({"release": label, "recall": recall})

        latest_label = current.dataset_name[:24] if current.dataset_name else current.run_id.hex[:8]
        trend_data.append({"release": latest_label, "recall": round(current.recall_at_20 * 100, 1)})
        return trend_data

    def _build_release_gates(self, result: EvaluationResult) -> list[dict[str, str]]:
        gates = [
            {
                "name": "Recall@20",
                "actual": f"{result.recall_at_20 * 100:.1f}%",
                "threshold": ">= 85.0%",
                "margin": (
                    f"+{abs(result.recall_at_20 * 100 - 85.0):.1f} pp"
                    if result.recall_at_20 * 100 >= 85.0
                    else f"-{abs(result.recall_at_20 * 100 - 85.0):.1f} pp"
                ),
            },
            {
                "name": "Faithfulness",
                "actual": f"{result.faithfulness * 100:.1f}%",
                "threshold": ">= 90.0%",
                "margin": (
                    f"+{abs(result.faithfulness * 100 - 90.0):.1f} pp"
                    if result.faithfulness * 100 >= 90.0
                    else f"-{abs(result.faithfulness * 100 - 90.0):.1f} pp"
                ),
            },
            {
                "name": "Citation precision",
                "actual": f"{result.citation_precision * 100:.1f}%",
                "threshold": ">= 90.0%",
                "margin": (
                    f"+{abs(result.citation_precision * 100 - 90.0):.1f} pp"
                    if result.citation_precision * 100 >= 90.0
                    else f"-{abs(result.citation_precision * 100 - 90.0):.1f} pp"
                ),
            },
            {
                "name": "Answer relevancy",
                "actual": f"{result.answer_relevancy * 100:.1f}%",
                "threshold": ">= 85.0%",
                "margin": (
                    f"+{abs(result.answer_relevancy * 100 - 85.0):.1f} pp"
                    if result.answer_relevancy * 100 >= 85.0
                    else f"-{abs(result.answer_relevancy * 100 - 85.0):.1f} pp"
                ),
            },
            {
                "name": "P95 response latency",
                "actual": f"{result.p95_latency_s:.2f} s",
                "threshold": "<= 2.20 s",
                "margin": f"-{abs(result.p95_latency_s - 2.20):.2f} s"
                if result.p95_latency_s <= 2.20
                else f"+{abs(result.p95_latency_s - 2.20):.2f} s",
            },
        ]
        return gates
