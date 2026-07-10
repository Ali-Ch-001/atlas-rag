from __future__ import annotations

import asyncio
import signal

import structlog

from rag_platform.adapters.cache import CacheStore
from rag_platform.adapters.embeddings import create_embedding_provider
from rag_platform.adapters.events import EventConsumer, EventProducer
from rag_platform.adapters.graph_store import GraphStore
from rag_platform.adapters.object_store import ObjectStore
from rag_platform.adapters.vector_store import VectorStore
from rag_platform.config import get_settings
from rag_platform.domain.models import IngestionEvent
from rag_platform.ingestion.service import IngestionService
from rag_platform.logging import configure_logging
from rag_platform.telemetry import DEAD_LETTER_MESSAGES


async def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    logger = structlog.get_logger(__name__)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)

    object_store = ObjectStore(settings)
    vector_store = VectorStore(settings)
    cache = CacheStore(settings)
    await object_store.ensure_buckets()
    await vector_store.ensure_collection()
    service = IngestionService(
        settings,
        object_store,
        create_embedding_provider(settings),
        vector_store,
        cache,
        GraphStore(settings) if settings.neo4j_enabled else None,
    )
    retry_producer = EventProducer(settings)
    await retry_producer.start()
    async with EventConsumer(
        settings, "document.accepted.v1", settings.kafka_consumer_group
    ) as consumer:
        async for payload in consumer.events():
            if stop.is_set():
                break
            event = IngestionEvent.model_validate(payload)
            try:
                await service.process(event)
            except Exception as exc:
                logger.exception("ingestion_event_failed", event_id=str(event.event_id))
                if event.attempt < 3:
                    await asyncio.sleep(min(2**event.attempt, 30))
                    retry = event.model_copy(update={"attempt": event.attempt + 1})
                    await retry_producer.send(
                        event.event_type,
                        str(event.version_id),
                        retry.model_dump(mode="json"),
                    )
                else:
                    await retry_producer.send(
                        "document.failed.v1",
                        str(event.version_id),
                        {
                            **event.model_dump(mode="json"),
                            "event_type": "document.failed.v1",
                            "error_code": type(exc).__name__,
                        },
                    )
                    await retry_producer.send(
                        "document.dead_letter.v1",
                        str(event.version_id),
                        {
                            **event.model_dump(mode="json"),
                            "event_type": "document.dead_letter.v1",
                            "error_code": type(exc).__name__,
                            "attempt": event.attempt,
                        },
                    )
                    DEAD_LETTER_MESSAGES.inc()
            finally:
                await consumer.commit()
    await retry_producer.stop()
    await cache.close()


if __name__ == "__main__":
    asyncio.run(main())
