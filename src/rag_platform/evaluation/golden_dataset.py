"""Real, committed golden evaluation dataset for RAG quality regression testing.

Each case contains a question, a reference answer with expected facts, and the
document text the answer should be drawn from. This dataset is versioned and
should be expanded as the corpus grows.

When the evaluation service runs, it indexes the provided documents, retrieves
against the questions, and compares generated answers to reference answers using
RAGAS metrics (context precision, context recall, faithfulness, answer relevancy,
answer correctness).
"""

from __future__ import annotations

from dataclasses import dataclass

DATASET_VERSION = "v1.0.0"

_DR_TEXT = (
    "Disaster Recovery Standard\n\n"
    "Recovery Objectives\n"
    "Production metadata has a recovery point objective of five minutes. "
    "The recovery time objective is one hour. Backups are restored quarterly "
    "and regional failover exercises run twice each year.\n\n"
    "Access Controls\n"
    "All retrieval requests must apply tenant and access-control filters "
    "before search. Cross-tenant access is prohibited and monitored continuously."
)


@dataclass(frozen=True, slots=True)
class GoldenCase:
    case_id: str
    question: str
    reference_answer: str
    document_title: str
    document_text: str
    expected_facts: tuple[str, ...]
    unanswerable: bool = False


def load_dataset() -> list[GoldenCase]:
    return [
        GoldenCase(
            case_id="GOLD-001",
            question="What is the recovery point objective for production metadata?",
            reference_answer=(
                "The recovery point objective (RPO) for production metadata is five minutes."
            ),
            document_title="Disaster Recovery Standard",
            document_text=(
                "Disaster Recovery Standard\n\n"
                "Recovery Objectives\n"
                "Production metadata has a recovery point objective of five minutes. "
                "The recovery time objective is one hour. "
                "Backups are restored quarterly and regional failover exercises run twice each year.\n\n"
                "Access Controls\n"
                "All retrieval requests must apply tenant and access-control filters before search. "
                "Cross-tenant access is prohibited and monitored continuously."
            ),
            expected_facts=("recovery point objective", "five minutes", "production metadata"),
        ),
        GoldenCase(
            case_id="GOLD-002",
            question="What is the recovery time objective?",
            reference_answer="The recovery time objective (RTO) is one hour.",
            document_title="Disaster Recovery Standard",
            document_text=(
                "Disaster Recovery Standard\n\n"
                "Recovery Objectives\n"
                "Production metadata has a recovery point objective of five minutes. "
                "The recovery time objective is one hour. "
                "Backups are restored quarterly and regional failover exercises run twice each year.\n\n"
                "Access Controls\n"
                "All retrieval requests must apply tenant and access-control filters before search. "
                "Cross-tenant access is prohibited and monitored continuously."
            ),
            expected_facts=("recovery time objective", "one hour"),
        ),
        GoldenCase(
            case_id="GOLD-003",
            question="How often are backups restored?",
            reference_answer="Backups are restored quarterly.",
            document_title="Disaster Recovery Standard",
            document_text=(
                "Disaster Recovery Standard\n\n"
                "Recovery Objectives\n"
                "Production metadata has a recovery point objective of five minutes. "
                "The recovery time objective is one hour. "
                "Backups are restored quarterly and regional failover exercises run twice each year.\n\n"
                "Access Controls\n"
                "All retrieval requests must apply tenant and access-control filters before search. "
                "Cross-tenant access is prohibited and monitored continuously."
            ),
            expected_facts=("backups", "quarterly", "restored"),
        ),
        GoldenCase(
            case_id="GOLD-004",
            question="What must happen before search is performed?",
            reference_answer=(
                "All retrieval requests must apply tenant and access-control filters before search."
            ),
            document_title="Disaster Recovery Standard",
            document_text=(
                "Disaster Recovery Standard\n\n"
                "Recovery Objectives\n"
                "Production metadata has a recovery point objective of five minutes. "
                "The recovery time objective is one hour. "
                "Backups are restored quarterly and regional failover exercises run twice each year.\n\n"
                "Access Controls\n"
                "All retrieval requests must apply tenant and access-control filters before search. "
                "Cross-tenant access is prohibited and monitored continuously."
            ),
            expected_facts=("tenant", "access-control filters", "before search"),
        ),
        GoldenCase(
            case_id="GOLD-005",
            question="Is cross-tenant access allowed?",
            reference_answer="No, cross-tenant access is prohibited and monitored continuously.",
            document_title="Disaster Recovery Standard",
            document_text=(
                "Disaster Recovery Standard\n\n"
                "Recovery Objectives\n"
                "Production metadata has a recovery point objective of five minutes. "
                "The recovery time objective is one hour. "
                "Backups are restored quarterly and regional failover exercises run twice each year.\n\n"
                "Access Controls\n"
                "All retrieval requests must apply tenant and access-control filters before search. "
                "Cross-tenant access is prohibited and monitored continuously."
            ),
            expected_facts=("cross-tenant", "prohibited", "monitored"),
        ),
        GoldenCase(
            case_id="GOLD-006",
            question="How often do regional failover exercises run?",
            reference_answer="Regional failover exercises run twice each year.",
            document_title="Disaster Recovery Standard",
            document_text=(
                "Disaster Recovery Standard\n\n"
                "Recovery Objectives\n"
                "Production metadata has a recovery point objective of five minutes. "
                "The recovery time objective is one hour. "
                "Backups are restored quarterly and regional failover exercises run twice each year.\n\n"
                "Access Controls\n"
                "All retrieval requests must apply tenant and access-control filters before search. "
                "Cross-tenant access is prohibited and monitored continuously."
            ),
            expected_facts=("failover", "twice", "year", "regional"),
        ),
        GoldenCase(
            case_id="GOLD-007",
            question="Who is the CEO of the company?",
            reference_answer="",
            document_title="",
            document_text="",
            expected_facts=(),
            unanswerable=True,
        ),
    ]
