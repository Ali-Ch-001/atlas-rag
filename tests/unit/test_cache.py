from __future__ import annotations

from array import array
from unittest.mock import MagicMock, patch

from rag_platform.adapters.cache import (
    CacheStore,
    ElastiCacheIamCredentialProvider,
)


class TestVectorBytes:
    def test_vector_bytes_converts_float_list_to_bytes(self) -> None:
        floats = [1.0, 2.0, 3.0, 4.0]
        result = CacheStore.vector_bytes(floats)
        expected = array("f", floats).tobytes()
        assert isinstance(result, bytes)
        assert len(result) == len(floats) * 4
        assert result == expected

    def test_vector_bytes_empty_list(self) -> None:
        result = CacheStore.vector_bytes([])
        assert isinstance(result, bytes)
        assert len(result) == 0


class TestElastiCacheIamCredentialProvider:
    def test_caches_credentials_within_expiry_window(self) -> None:
        with (
            patch(
                "rag_platform.adapters.cache.botocore.session"
            ) as mock_session_module,
            patch(
                "rag_platform.adapters.cache.RequestSigner"
            ) as mock_signer_cls,
        ):
            mock_creds = MagicMock()
            mock_creds.access_key = "AKIATEST"
            mock_creds.secret_key = "testsecret"  # noqa: S105
            mock_creds.token = "testtoken"  # noqa: S105
            mock_sess = MagicMock()
            mock_sess.get_credentials.return_value = mock_creds
            mock_sess.get_component.return_value = MagicMock()
            mock_session_module.get_session.return_value = mock_sess

            mock_signer = MagicMock()
            mock_signer.generate_presigned_url.return_value = (
                "http://test-cache/?Action=connect&User=test-user&"
                "X-Amz-Signature=abc123"
            )
            mock_signer_cls.return_value = mock_signer

            provider = ElastiCacheIamCredentialProvider(
                user_id="test-user",
                cache_name="test-cache",
                region="us-east-1",
            )

            first = provider.get_credentials()
            assert first[0] == "test-user"
            assert "test-cache" in first[1]
            assert first is provider._credentials

            second = provider.get_credentials()
            assert second is first
