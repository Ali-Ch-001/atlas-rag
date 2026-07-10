from __future__ import annotations

from functools import lru_cache
from typing import Literal
from uuid import UUID

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    environment: Literal["local", "dev", "stage", "prod", "test"] = "local"
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"  # noqa: S104
    api_port: int = 8000
    cors_origins: str = "http://localhost:3000"

    auth_disabled: bool = True
    dev_tenant_id: UUID = UUID("00000000-0000-0000-0000-000000000001")
    dev_subject_id: str = "local-developer"
    auth0_domain: str = ""
    auth0_audience: str = ""
    auth0_algorithms: str = "RS256"

    database_url: str = "postgresql+asyncpg://rag:rag-local-password@localhost:5432/rag"
    database_pool_size: int = 10
    database_max_overflow: int = 20

    s3_endpoint_url: str | None = None
    s3_region: str = "us-east-1"
    s3_access_key_id: str | None = None
    s3_secret_access_key: str | None = None
    s3_quarantine_bucket: str = "rag-quarantine"
    s3_clean_bucket: str = "rag-clean"
    s3_derived_bucket: str = "rag-derived"
    s3_force_path_style: bool = True

    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_security_protocol: str = "PLAINTEXT"
    kafka_sasl_mechanism: str | None = None
    kafka_sasl_username: str | None = None
    kafka_sasl_password: str | None = None
    kafka_consumer_group: str = "rag-ingestion-v1"
    kafka_aws_msk_iam: bool = False
    aws_region: str = "us-east-1"

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection_prefix: str = "rag_chunks"
    qdrant_vector_size: int = 1536
    qdrant_replication_factor: int = 1

    redis_url: str = "redis://localhost:6379/0"
    redis_iam_enabled: bool = False
    redis_iam_user: str | None = None
    redis_iam_cache_name: str | None = None
    cache_volatile_ttl_seconds: int = 3600
    cache_stable_ttl_seconds: int = 86400
    semantic_cache_threshold: float = 0.93
    citation_hmac_secret: str = "local-citation-secret-change-in-production"  # noqa: S105

    model_provider: Literal["deterministic", "openai"] = "deterministic"
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    openai_embedding_model: str = "text-embedding-3-small"
    openai_chat_model: str = "gpt-4.1-mini"
    openai_timeout_seconds: float = 30.0

    reranker_provider: Literal["lexical", "cross_encoder"] = "lexical"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_top_n: int = 5
    reranker_term_weight: float = 0.7
    reranker_title_weight: float = 0.2
    reranker_rrf_weight: float = 0.1
    reranker_rrf_scale: float = 100.0

    tavily_api_key: str | None = None
    tavily_base_url: str = "https://api.tavily.com"
    web_search_enabled: bool = False

    max_upload_bytes: int = 100 * 1024 * 1024
    max_pdf_pages: int = 5000
    chunk_target_tokens: int = 450
    chunk_max_tokens: int = 600
    chunk_overlap_tokens: int = 60
    chunker_document_max_tokens: int = 50000
    ocr_enabled: bool = False
    ocr_dpi: int = 300
    ocr_language: str = "eng"
    clamav_enabled: bool = False
    clamav_host: str = "localhost"
    clamav_port: int = 3310

    sparse_candidates: int = 50
    dense_candidates: int = 50
    fusion_candidates: int = 20
    final_context_chunks: int = 5
    rrf_k: int = 60
    mmr_lambda: float = 0.7
    context_evidence_ratio: float = 0.65

    neo4j_enabled: bool = False
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str = "neo4j-local-password"  # noqa: S105

    otel_enabled: bool = False
    otel_exporter_otlp_endpoint: str = "http://localhost:4317"
    otel_service_name: str = "rag-api"

    @property
    def qdrant_collection(self) -> str:
        model_slug = self.openai_embedding_model.replace("-", "_").replace(".", "_")
        return f"{self.qdrant_collection_prefix}_{model_slug}_{self.qdrant_vector_size}"

    @property
    def cors_origin_list(self) -> list[str]:
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]

    @model_validator(mode="after")
    def validate_production_safety(self) -> Settings:
        if self.environment in {"stage", "prod"} and self.auth_disabled:
            raise ValueError("AUTH_DISABLED must be false in stage and prod")
        if self.environment in {"stage", "prod"} and self.citation_hmac_secret.startswith("local-"):
            raise ValueError("CITATION_HMAC_SECRET must be replaced in stage and prod")
        if not self.auth_disabled and (not self.auth0_domain or not self.auth0_audience):
            raise ValueError("AUTH0_DOMAIN and AUTH0_AUDIENCE are required when auth is enabled")
        if self.model_provider == "openai" and not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when MODEL_PROVIDER=openai")
        if self.web_search_enabled and not self.tavily_api_key:
            raise ValueError("TAVILY_API_KEY is required when WEB_SEARCH_ENABLED=true")
        if self.redis_iam_enabled and (not self.redis_iam_user or not self.redis_iam_cache_name):
            raise ValueError(
                "REDIS_IAM_USER and REDIS_IAM_CACHE_NAME are required when Redis IAM is enabled"
            )
        if not 0.0 < self.context_evidence_ratio <= 0.70:
            raise ValueError("CONTEXT_EVIDENCE_RATIO must be in (0, 0.70]")
        if not 0.0 <= self.mmr_lambda <= 1.0:
            raise ValueError("MMR_LAMBDA must be in [0, 1]")
        reranker_sum = (
            self.reranker_term_weight + self.reranker_title_weight + self.reranker_rrf_weight
        )
        if abs(reranker_sum - 1.0) > 0.01:
            raise ValueError(
                "RERANKER_TERM_WEIGHT + RERANKER_TITLE_WEIGHT + RERANKER_RRF_WEIGHT "
                f"must sum to ~1.0 (got {reranker_sum})"
            )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
