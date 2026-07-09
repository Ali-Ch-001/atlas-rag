from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

import orjson
import structlog
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer  # type: ignore[import-untyped]
from aws_msk_iam_sasl_signer import MSKAuthTokenProvider  # type: ignore[import-untyped]
from sqlalchemy import select

from rag_platform.config import Settings
from rag_platform.db.models import OutboxEvent
from rag_platform.db.session import SessionFactory

logger = structlog.get_logger(__name__)


class MskIamTokenProvider:
    def __init__(self, region: str) -> None:
        self.region = region

    async def token(self) -> str:
        token, _ = await asyncio.to_thread(MSKAuthTokenProvider.generate_auth_token, self.region)
        return str(token)


def _kafka_auth(settings: Settings) -> dict[str, Any]:
    if settings.kafka_aws_msk_iam:
        return {
            "security_protocol": "SASL_SSL",
            "sasl_mechanism": "OAUTHBEARER",
            "sasl_oauth_token_provider": MskIamTokenProvider(settings.aws_region),
        }
    return {
        "security_protocol": settings.kafka_security_protocol,
        "sasl_mechanism": settings.kafka_sasl_mechanism,
        "sasl_plain_username": settings.kafka_sasl_username,
        "sasl_plain_password": settings.kafka_sasl_password,
    }


class EventProducer:
    def __init__(self, settings: Settings) -> None:
        self._producer = AIOKafkaProducer(
            bootstrap_servers=settings.kafka_bootstrap_servers,
            value_serializer=orjson.dumps,
            key_serializer=lambda value: value.encode(),
            acks="all",
            enable_idempotence=True,
            **_kafka_auth(settings),
        )

    async def start(self) -> None:
        await self._producer.start()

    async def stop(self) -> None:
        await self._producer.stop()

    async def send(self, topic: str, key: str, payload: dict[str, object]) -> None:
        await self._producer.send_and_wait(topic, key=key, value=payload)


class OutboxPublisher:
    def __init__(self, producer: EventProducer, poll_interval: float = 0.5) -> None:
        self.producer = producer
        self.poll_interval = poll_interval
        self._stopping = asyncio.Event()

    async def run(self) -> None:
        while not self._stopping.is_set():
            published = await self.publish_batch()
            if published == 0:
                try:
                    await asyncio.wait_for(self._stopping.wait(), timeout=self.poll_interval)
                except TimeoutError:
                    pass

    def stop(self) -> None:
        self._stopping.set()

    async def publish_batch(self, limit: int = 100) -> int:
        async with SessionFactory() as session, session.begin():
            events = list(
                (
                    await session.scalars(
                        select(OutboxEvent)
                        .where(OutboxEvent.published_at.is_(None))
                        .order_by(OutboxEvent.created_at)
                        .limit(limit)
                        .with_for_update(skip_locked=True)
                    )
                ).all()
            )
            for event in events:
                await self.producer.send(event.topic, event.event_key, event.payload)
                event.published_at = datetime.now(UTC)
            return len(events)


class EventConsumer:
    def __init__(self, settings: Settings, topic: str, group_id: str) -> None:
        self._consumer = AIOKafkaConsumer(
            topic,
            bootstrap_servers=settings.kafka_bootstrap_servers,
            group_id=group_id,
            value_deserializer=orjson.loads,
            enable_auto_commit=False,
            auto_offset_reset="earliest",
            **_kafka_auth(settings),
        )

    async def __aenter__(self) -> EventConsumer:
        await self._consumer.start()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self._consumer.stop()

    async def events(self) -> AsyncIterator[dict[str, object]]:
        async for message in self._consumer:
            yield message.value

    async def commit(self) -> None:
        await self._consumer.commit()
