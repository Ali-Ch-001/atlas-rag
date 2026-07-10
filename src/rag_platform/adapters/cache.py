from __future__ import annotations

import asyncio
import hashlib
import json
from array import array
from dataclasses import dataclass
from time import monotonic
from urllib.parse import urlencode
from uuid import UUID, uuid4

import botocore.session  # type: ignore[import-untyped]
import redis.asyncio as redis
import structlog
from botocore.model import ServiceId  # type: ignore[import-untyped]
from botocore.signers import RequestSigner  # type: ignore[import-untyped]
from redis.credentials import CredentialProvider
from redis.exceptions import AuthenticationError, ConnectionError, ResponseError, TimeoutError

from rag_platform.config import Settings

logger = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class SemanticCacheHit:
    payload: str
    similarity: float


class ElastiCacheIamCredentialProvider(CredentialProvider):
    """IAM credential provider with retry and expiry-aware renewal.

    AWS ElastiCache IAM tokens expire after 15 minutes. This provider
    generates a new token 60 seconds before expiry (at 840s into the 900s
    lifetime) and retries once on transient credential fetch failures.
    """

    def __init__(self, user_id: str, cache_name: str, region: str) -> None:
        self.user_id = user_id
        self.cache_name = cache_name.lower()
        self.region = region
        self._session = botocore.session.get_session()
        self._credentials: tuple[str, str] | None = None
        self._expires_at = 0.0
        self._lock = asyncio.Lock()

    def get_credentials(self) -> tuple[str, str]:
        if self._credentials and monotonic() < self._expires_at:
            return self._credentials
        for attempt in range(3):
            try:
                credentials = self._session.get_credentials()
                if credentials is None:
                    raise RuntimeError("AWS workload credentials unavailable for Redis IAM")
                signer = RequestSigner(
                    ServiceId("elasticache"),
                    self.region,
                    "elasticache",
                    "v4",
                    credentials,
                    self._session.get_component("event_emitter"),
                )
                query = urlencode({"Action": "connect", "User": self.user_id})
                signed_url = signer.generate_presigned_url(
                    {
                        "method": "GET",
                        "url": f"http://{self.cache_name}/?{query}",
                        "body": {},
                        "headers": {},
                        "context": {},
                    },
                    operation_name="connect",
                    expires_in=900,
                    region_name=self.region,
                )
                self._credentials = (self.user_id, signed_url.removeprefix("http://"))
                self._expires_at = monotonic() + 840
                return self._credentials
            except Exception:
                if attempt == 2:
                    raise
                logger.warning(
                    "redis_iam_credential_retry",
                    attempt=attempt + 1,
                    user_id=self.user_id,
                    region=self.region,
                )
        raise RuntimeError("Failed to obtain ElastiCache IAM credentials after retries")

    async def get_credentials_async(self) -> tuple[str, str]:
        async with self._lock:
            return await asyncio.to_thread(self.get_credentials)


class CacheStore:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        client_options: dict[str, object] = {
            "decode_responses": False,
            "retry_on_timeout": True,
            "health_check_interval": 30,
        }
        if settings.redis_iam_enabled:
            client_options["credential_provider"] = ElastiCacheIamCredentialProvider(
                settings.redis_iam_user or "",
                settings.redis_iam_cache_name or "",
                settings.aws_region,
            )
        self.client = redis.from_url(  # type: ignore[no-untyped-call]
            settings.redis_url, **client_options
        )
        self.semantic_index = f"rag_semantic_cache_{settings.qdrant_vector_size}"

    async def ping(self) -> bool:
        try:
            return bool(await self.client.ping())
        except (ConnectionError, TimeoutError, AuthenticationError):
            return False

    async def close(self) -> None:
        await self.client.aclose()

    async def ensure_semantic_index(self) -> None:
        try:
            await self.client.execute_command(
                "FT.CREATE",
                self.semantic_index,
                "ON",
                "HASH",
                "PREFIX",
                "1",
                "rag:semantic:",
                "SCHEMA",
                "tenant",
                "TAG",
                "acl",
                "TAG",
                "scope",
                "TAG",
                "query_vector",
                "VECTOR",
                "HNSW",
                "6",
                "TYPE",
                "FLOAT32",
                "DIM",
                str(self.settings.qdrant_vector_size),
                "DISTANCE_METRIC",
                "COSINE",
            )
        except ResponseError as exc:
            if "Index already exists" not in str(exc):
                raise

    async def corpus_epoch(self, tenant_id: UUID, corpus_ids: list[UUID]) -> str:
        keys = [f"rag:epoch:{tenant_id}:{corpus_id}" for corpus_id in sorted(corpus_ids)]
        values = await self.client.mget(keys)
        return ":".join(
            f"{corpus_id}={int(value) if value is not None else 0}"
            for corpus_id, value in zip(sorted(corpus_ids), values, strict=True)
        )

    async def bump_corpus_epoch(self, tenant_id: UUID, corpus_id: UUID) -> int:
        return int(await self.client.incr(f"rag:epoch:{tenant_id}:{corpus_id}"))

    async def get_exact(self, key: str) -> str | None:
        value = await self.client.get(f"rag:exact:{key}")
        return value.decode() if value else None

    async def set_exact(self, key: str, payload: str, *, stable: bool = False) -> None:
        ttl = (
            self.settings.cache_stable_ttl_seconds
            if stable
            else self.settings.cache_volatile_ttl_seconds
        )
        await self.client.set(f"rag:exact:{key}", payload, ex=ttl)

    @staticmethod
    def vector_bytes(vector: list[float]) -> bytes:
        values = array("f", vector)
        if values.itemsize != 4:
            raise RuntimeError("Semantic cache requires 32-bit floats")
        return values.tobytes()

    async def get_semantic(
        self,
        *,
        tenant_id: UUID,
        acl_fingerprint: str,
        scope_hash: str,
        query_vector: list[float],
    ) -> SemanticCacheHit | None:
        query = (
            f"(@tenant:{{{tenant_id}}} @acl:{{{acl_fingerprint}}} "
            f"@scope:{{{scope_hash}}})=>[KNN 1 @query_vector $vector AS distance]"
        )
        try:
            result = await self.client.execute_command(
                "FT.SEARCH",
                self.semantic_index,
                query,
                "PARAMS",
                "2",
                "vector",
                self.vector_bytes(query_vector),
                "SORTBY",
                "distance",
                "RETURN",
                "2",
                "payload",
                "distance",
                "DIALECT",
                "2",
            )
        except ResponseError:
            return None
        if not result or int(result[0]) == 0:
            return None
        fields = result[2]
        decoded = {
            fields[index].decode(): fields[index + 1].decode() for index in range(0, len(fields), 2)
        }
        similarity = 1.0 - float(decoded["distance"])
        if similarity < self.settings.semantic_cache_threshold:
            return None
        return SemanticCacheHit(payload=decoded["payload"], similarity=similarity)

    async def set_semantic(
        self,
        *,
        tenant_id: UUID,
        acl_fingerprint: str,
        scope_hash: str,
        query_vector: list[float],
        payload: str,
        stable: bool = False,
    ) -> None:
        ttl = (
            self.settings.cache_stable_ttl_seconds
            if stable
            else self.settings.cache_volatile_ttl_seconds
        )
        key = f"rag:semantic:{uuid4()}"
        await self.client.hset(
            key,
            mapping={
                "tenant": str(tenant_id),
                "acl": acl_fingerprint,
                "scope": scope_hash,
                "query_vector": self.vector_bytes(query_vector),
                "payload": payload,
            },
        )
        await self.client.expire(key, ttl)

    async def get_embedding(self, model: str, text_hash: str) -> list[float] | None:
        model_hash = hashlib.sha256(model.encode()).hexdigest()[:16]
        value = await self.client.get(f"rag:embedding:{model_hash}:{text_hash}")
        return json.loads(value) if value else None

    async def set_embedding(
        self, model: str, text_hash: str, vector: list[float], ttl: int = 86400
    ) -> None:
        model_hash = hashlib.sha256(model.encode()).hexdigest()[:16]
        await self.client.set(
            f"rag:embedding:{model_hash}:{text_hash}",
            json.dumps(vector, separators=(",", ":")),
            ex=ttl,
        )

    async def cancel_response(self, response_id: UUID, ttl: int = 300) -> None:
        await self.client.set(f"rag:cancel:{response_id}", "1", ex=ttl)

    async def is_cancelled(self, response_id: UUID) -> bool:
        return bool(await self.client.exists(f"rag:cancel:{response_id}"))
