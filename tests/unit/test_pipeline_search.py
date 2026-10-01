"""Unit tests for the pipeline's search backend selection and health checks."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import SecretStr

from exceptions import SearchBackendError
from pipeline import DisinformationDetectionPipeline
from retrieval.web_search import OllamaSearchBackend, SearxngBackend

_KEY = "sk-test-pipeline-0123456789"


def _init(settings: MagicMock) -> tuple[DisinformationDetectionPipeline, dict[str, MagicMock]]:
    pipeline = DisinformationDetectionPipeline(settings=settings)
    with patch(
        "retrieval.web_search.SearxngBackend.health_check", return_value=True
    ) as searx_hc, patch(
        "retrieval.web_search.OllamaSearchBackend.health_check", return_value=True
    ) as ollama_hc, patch(
        "verification.nli_verifier.NLIVerifier._load_model"
    ), patch(
        "verification.rag_verifier.OllamaClient.health_check", return_value=True
    ), patch("pipeline.setup_logging"):
        pipeline.initialize()
    return pipeline, {"searxng": searx_hc, "ollama": ollama_hc}


class TestBackendSelection:
    def test_default_builds_searxng(self, settings: MagicMock) -> None:
        pipeline, checks = _init(settings)
        assert isinstance(pipeline._retriever._backend, SearxngBackend)
        assert pipeline._retriever._fetcher.mode == "direct"
        checks["searxng"].assert_called_once()
        checks["ollama"].assert_not_called()

    def test_ollama_backend(self, settings: MagicMock) -> None:
        settings.search.backend = "ollama"
        settings.search.ollama_api_key = SecretStr(_KEY)
        pipeline, checks = _init(settings)
        assert isinstance(pipeline._retriever._backend, OllamaSearchBackend)
        assert pipeline._retriever.backend_name == "ollama"
        checks["ollama"].assert_called_once()
        checks["searxng"].assert_not_called()

    def test_mixed_searxng_with_ollama_fetcher(self, settings: MagicMock) -> None:
        settings.search.page_fetcher = "ollama"
        settings.search.ollama_api_key = SecretStr(_KEY)
        pipeline, checks = _init(settings)
        assert isinstance(pipeline._retriever._backend, SearxngBackend)
        assert pipeline._retriever._fetcher.mode == "ollama"
        checks["searxng"].assert_called_once()
        checks["ollama"].assert_called_once()


class TestMissingKey:
    @pytest.mark.parametrize(
        ("backend", "fetcher", "key"),
        [
            ("ollama", "direct", None),
            ("searxng", "ollama", None),
            ("ollama", "ollama", SecretStr("   ")),
        ],
    )
    def test_fails_startup_naming_setting(
        self, settings: MagicMock, backend: str, fetcher: str, key: SecretStr | None
    ) -> None:
        settings.search.backend = backend
        settings.search.page_fetcher = fetcher
        settings.search.ollama_api_key = key
        with pytest.raises(SearchBackendError, match="SEARCH_OLLAMA_API_KEY"):
            _init(settings)

    def test_read_only_skips_checks_and_key(
        self, settings: MagicMock, tmp_path: Path
    ) -> None:
        settings.retrieval.cache_mode = "read_only"
        settings.retrieval.cache_dir = str(tmp_path)
        settings.search.backend = "ollama"
        settings.search.page_fetcher = "ollama"
        settings.search.ollama_api_key = None
        pipeline, checks = _init(settings)
        assert pipeline._initialized is True
        checks["ollama"].assert_not_called()
        checks["searxng"].assert_not_called()


class TestStartupReachability:
    def test_ollama_unreachable_names_url(self, settings: MagicMock) -> None:
        import httpx

        settings.search.backend = "ollama"
        settings.search.ollama_api_key = SecretStr(_KEY)
        settings.search.ollama_web_url = "https://ollama.invalid"

        def refuse(self: httpx.Client, url: str, **kwargs: object) -> httpx.Response:
            raise httpx.ConnectError("refused")

        pipeline = DisinformationDetectionPipeline(settings=settings)
        with patch.object(httpx.Client, "get", refuse), patch("pipeline.setup_logging"):
            with pytest.raises(SearchBackendError, match="https://ollama.invalid") as info:
                pipeline.initialize()
        assert _KEY not in str(info.value)
        assert pipeline._initialized is False


class TestHealthCheck:
    def test_default_reports_searxng(self, settings: MagicMock) -> None:
        pipeline = DisinformationDetectionPipeline(settings=settings)
        with patch(
            "retrieval.web_search.SearxngBackend.health_check", return_value=True
        ), patch("verification.rag_verifier.OllamaClient.health_check", return_value=True):
            status = pipeline.health_check()
        assert status == {"search": True, "searxng": True, "ollama": True}

    def test_ollama_backend_has_no_searxng_key(self, settings: MagicMock) -> None:
        settings.search.backend = "ollama"
        settings.search.ollama_api_key = SecretStr(_KEY)
        pipeline = DisinformationDetectionPipeline(settings=settings)
        with patch(
            "retrieval.web_search.OllamaSearchBackend.health_check", return_value=True
        ), patch("verification.rag_verifier.OllamaClient.health_check", return_value=True):
            status = pipeline.health_check()
        assert status == {"search": True, "ollama": True}
