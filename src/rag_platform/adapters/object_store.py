from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from io import BytesIO
from typing import Any, BinaryIO, cast

import aioboto3  # type: ignore[import-untyped]
from botocore.config import Config  # type: ignore[import-untyped]
from botocore.exceptions import ClientError  # type: ignore[import-untyped]

from rag_platform.config import Settings


class ObjectStore:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._session = aioboto3.Session()

    def _client_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "service_name": "s3",
            "region_name": self.settings.s3_region,
            "endpoint_url": self.settings.s3_endpoint_url,
            "config": Config(
                s3={"addressing_style": "path" if self.settings.s3_force_path_style else "auto"},
                retries={"max_attempts": 5, "mode": "adaptive"},
            ),
        }
        if self.settings.s3_access_key_id:
            kwargs["aws_access_key_id"] = self.settings.s3_access_key_id
        if self.settings.s3_secret_access_key:
            kwargs["aws_secret_access_key"] = self.settings.s3_secret_access_key
        return kwargs

    @asynccontextmanager
    async def client(self) -> AsyncIterator[Any]:
        async with self._session.client(**self._client_kwargs()) as client:
            yield client

    async def ensure_buckets(self) -> None:
        async with self.client() as client:
            for bucket in (
                self.settings.s3_quarantine_bucket,
                self.settings.s3_clean_bucket,
                self.settings.s3_derived_bucket,
            ):
                try:
                    await client.head_bucket(Bucket=bucket)
                except ClientError as exc:
                    code = exc.response.get("Error", {}).get("Code")
                    if code not in {"404", "NoSuchBucket", "NotFound"}:
                        raise
                    args: dict[str, Any] = {"Bucket": bucket}
                    if self.settings.s3_region != "us-east-1" and not self.settings.s3_endpoint_url:
                        args["CreateBucketConfiguration"] = {
                            "LocationConstraint": self.settings.s3_region
                        }
                    await client.create_bucket(**args)

    async def upload_fileobj(
        self,
        bucket: str,
        key: str,
        fileobj: BinaryIO,
        *,
        content_type: str = "application/octet-stream",
        metadata: dict[str, str] | None = None,
    ) -> None:
        fileobj.seek(0)
        async with self.client() as client:
            await client.upload_fileobj(
                fileobj,
                bucket,
                key,
                ExtraArgs={"ContentType": content_type, "Metadata": metadata or {}},
            )

    async def upload_bytes(
        self,
        bucket: str,
        key: str,
        content: bytes,
        *,
        content_type: str = "application/octet-stream",
        metadata: dict[str, str] | None = None,
    ) -> None:
        await self.upload_fileobj(
            bucket,
            key,
            BytesIO(content),
            content_type=content_type,
            metadata=metadata,
        )

    async def download_bytes(self, bucket: str, key: str) -> bytes:
        async with self.client() as client:
            response = await client.get_object(Bucket=bucket, Key=key)
            async with response["Body"] as stream:
                return cast(bytes, await stream.read())

    async def health_check(self) -> bool:
        try:
            async with self.client() as client:
                await client.head_bucket(Bucket=self.settings.s3_clean_bucket)
            return True
        except Exception:
            return False

    async def copy(
        self,
        source_bucket: str,
        source_key: str,
        target_bucket: str,
        target_key: str,
    ) -> None:
        async with self.client() as client:
            await client.copy_object(
                Bucket=target_bucket,
                Key=target_key,
                CopySource={"Bucket": source_bucket, "Key": source_key},
                MetadataDirective="COPY",
            )

    async def delete(self, bucket: str, key: str) -> None:
        async with self.client() as client:
            await client.delete_object(Bucket=bucket, Key=key)

    async def delete_prefix(self, bucket: str, prefix: str) -> int:
        deleted = 0
        continuation: str | None = None
        async with self.client() as client:
            while True:
                arguments: dict[str, Any] = {"Bucket": bucket, "Prefix": prefix}
                if continuation:
                    arguments["ContinuationToken"] = continuation
                response = await client.list_objects_v2(**arguments)
                objects = [{"Key": item["Key"]} for item in response.get("Contents", [])]
                if objects:
                    await client.delete_objects(
                        Bucket=bucket,
                        Delete={"Objects": objects, "Quiet": True},
                    )
                    deleted += len(objects)
                if not response.get("IsTruncated"):
                    break
                continuation = response.get("NextContinuationToken")
        return deleted
