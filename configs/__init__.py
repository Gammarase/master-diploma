"""
Configuration management for the Disinformation Detection System.

Settings are loaded from configs/config.yaml and can be overridden via
environment variables (e.g. SEARCH_BASE_URL overrides search.base_url).
"""

from __future__ import annotations

import os
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = [
    "OllamaSettings",
    "RetrievalSettings",
    "SearchSettings",
    "SourcePolicySettings",
    "PreprocessingSettings",
    "ClaimExtractionSettings",
    "VerificationThresholds",
    "VerificationSettings",
    "LoggingSettings",
    "Settings",
    "get_settings",
]

_CONFIG_PATH = Path(__file__).parent / "config.yaml"


def _load_yaml_defaults() -> dict[str, Any]:
    """Load default values from config.yaml."""
    if _CONFIG_PATH.exists():
        with open(_CONFIG_PATH, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


_YAML = _load_yaml_defaults()


class OllamaSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OLLAMA_")

    base_url: str = Field(
        default=_YAML.get("ollama", {}).get("base_url", "http://localhost:11434")
    )
    model: str = Field(default=_YAML.get("ollama", {}).get("model", "qwen3:14b"))
    temperature: float = Field(
        default=_YAML.get("ollama", {}).get("temperature", 0.0)
    )
    timeout_seconds: int = Field(
        default=_YAML.get("ollama", {}).get("timeout_seconds", 120)
    )
    num_ctx: int = Field(default=_YAML.get("ollama", {}).get("num_ctx", 8192))
    seed: int = Field(default=_YAML.get("ollama", {}).get("seed", 42))
    think: bool = Field(default=_YAML.get("ollama", {}).get("think", False))
    evidence_mode: Literal["web", "evidence_only"] = Field(
        default=_YAML.get("ollama", {}).get("evidence_mode", "web")
    )


class RetrievalSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RETRIEVAL_")

    top_k: int = Field(default=_YAML.get("retrieval", {}).get("top_k", 5))
    candidate_k: int = Field(
        default=_YAML.get("retrieval", {}).get("candidate_k", 40)
    )
    reranker_model: str | None = Field(
        default=_YAML.get("retrieval", {}).get(
            "reranker_model", "BAAI/bge-reranker-v2-m3"
        )
    )
    device: str = Field(default=_YAML.get("retrieval", {}).get("device", "cpu"))
    min_relevance: float = Field(
        default=_YAML.get("retrieval", {}).get("min_relevance", 0.2)
    )
    max_passages_per_source: int = Field(
        default=_YAML.get("retrieval", {}).get("max_passages_per_source", 2)
    )
    max_passage_chars: int = Field(
        default=_YAML.get("retrieval", {}).get("max_passage_chars", 800)
    )
    query_generation: bool = Field(
        default=_YAML.get("retrieval", {}).get("query_generation", True)
    )
    max_search_requests: int = Field(
        default=_YAML.get("retrieval", {}).get("max_search_requests", 8)
    )
    max_pages_per_claim: int = Field(
        default=_YAML.get("retrieval", {}).get("max_pages_per_claim", 8)
    )
    cache_dir: str = Field(
        default=_YAML.get("retrieval", {}).get("cache_dir", "cache")
    )
    cache_mode: Literal["read_write", "read_only", "off"] = Field(
        default=_YAML.get("retrieval", {}).get("cache_mode", "read_write")
    )
    cache_max_age_days: float | None = Field(
        default=_YAML.get("retrieval", {}).get("cache_max_age_days", None)
    )


class SearchSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SEARCH_")

    backend: Literal["searxng"] = Field(
        default=_YAML.get("search", {}).get("backend", "searxng")
    )
    base_url: str = Field(
        default=_YAML.get("search", {}).get("base_url", "http://localhost:8080")
    )
    timeout_seconds: float = Field(
        default=_YAML.get("search", {}).get("timeout_seconds", 20)
    )
    min_interval_seconds: float = Field(
        default=_YAML.get("search", {}).get("min_interval_seconds", 1.0)
    )
    results_per_query: int = Field(
        default=_YAML.get("search", {}).get("results_per_query", 10)
    )
    site_filter: Literal["grouped", "per_domain", "none"] = Field(
        default=_YAML.get("search", {}).get("site_filter", "grouped")
    )
    site_group_size: int = Field(
        default=_YAML.get("search", {}).get("site_group_size", 10), ge=1
    )
    fetch_timeout_seconds: float = Field(
        default=_YAML.get("search", {}).get("fetch_timeout_seconds", 10)
    )
    max_page_bytes: int = Field(
        default=_YAML.get("search", {}).get("max_page_bytes", 2_000_000)
    )
    user_agent: str = Field(
        default=_YAML.get("search", {}).get(
            "user_agent", "DisinfoDetection-Research/1.0"
        )
    )
    respect_robots: bool = Field(
        default=_YAML.get("search", {}).get("respect_robots", True)
    )


class SourcePolicySettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SOURCE_POLICY_")

    path: str = Field(
        default=_YAML.get("source_policy", {}).get(
            "path", "configs/source_policy.yaml"
        )
    )
    mode: Literal["strict", "lenient", "off"] = Field(
        default=_YAML.get("source_policy", {}).get("mode", "strict")
    )
    unknown_trust: float = Field(
        default=_YAML.get("source_policy", {}).get("unknown_trust", 0.3),
        ge=0.0,
        le=1.0,
    )


class PreprocessingSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PREPROCESSING_")

    supported_languages: list[str] = Field(
        default=_YAML.get("preprocessing", {}).get(
            "supported_languages", ["en", "uk", "ru", "zh"]
        )
    )
    spacy_models: dict[str, str] = Field(
        default=_YAML.get("preprocessing", {}).get(
            "spacy_models",
            {
                "en": "en_core_web_sm",
                "uk": "uk_core_news_sm",
                "ru": "ru_core_news_sm",
                "zh": "zh_core_web_sm",
            },
        )
    )


class ClaimExtractionSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CLAIM_EXTRACTION_")

    min_claim_length: int = Field(
        default=_YAML.get("claim_extraction", {}).get("min_claim_length", 10)
    )
    max_claim_length: int = Field(
        default=_YAML.get("claim_extraction", {}).get("max_claim_length", 512)
    )
    checkworthy_threshold: float = Field(
        default=_YAML.get("claim_extraction", {}).get("checkworthy_threshold", 0.1)
    )
    classifier_model: str = Field(
        default=_YAML.get("claim_extraction", {}).get(
            "classifier_model",
            "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
        )
    )
    ner_weight: float = Field(
        default=_YAML.get("claim_extraction", {}).get("ner_weight", 0.3)
    )
    classifier_weight: float = Field(
        default=_YAML.get("claim_extraction", {}).get("classifier_weight", 0.7)
    )


class VerificationThresholds(BaseSettings):
    disinformation: float = Field(
        default=_YAML.get("verification", {})
        .get("thresholds", {})
        .get("disinformation", 0.35)
    )
    confirmed: float = Field(
        default=_YAML.get("verification", {})
        .get("thresholds", {})
        .get("confirmed", 0.65)
    )


_LEGACY_MARGIN_KEY = "disagreement_delta"
_MARGIN_KEY = "decisiveness_margin"


def _warn_legacy_margin() -> None:
    warnings.warn(
        f"verification.{_LEGACY_MARGIN_KEY} is deprecated; "
        f"use verification.{_MARGIN_KEY} instead.",
        DeprecationWarning,
        stacklevel=3,
    )


def _default_decisiveness_margin() -> float:
    """Read the margin from config.yaml, accepting the legacy key."""
    section = _YAML.get("verification", {})
    if _MARGIN_KEY in section:
        return float(section[_MARGIN_KEY])
    if _LEGACY_MARGIN_KEY in section:
        _warn_legacy_margin()
        return float(section[_LEGACY_MARGIN_KEY])
    return 0.15


class VerificationSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VERIFICATION_")

    nli_model: str = Field(
        default=_YAML.get("verification", {}).get(
            "nli_model", "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
        )
    )
    nli_weight: float = Field(
        default=_YAML.get("verification", {}).get("nli_weight", 0.3)
    )
    rag_weight: float = Field(
        default=_YAML.get("verification", {}).get("rag_weight", 0.7)
    )
    thresholds: VerificationThresholds = Field(
        default_factory=VerificationThresholds
    )
    decisiveness_margin: float = Field(
        default_factory=_default_decisiveness_margin
    )

    @model_validator(mode="before")
    @classmethod
    def _accept_legacy_margin(cls, data: Any) -> Any:
        """Map the deprecated ``disagreement_delta`` to ``decisiveness_margin``."""
        if isinstance(data, dict) and _LEGACY_MARGIN_KEY in data:
            data = dict(data)
            legacy = data.pop(_LEGACY_MARGIN_KEY)
            _warn_legacy_margin()
            data.setdefault(_MARGIN_KEY, legacy)
        return data


class LoggingSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LOGGING_")

    level: str = Field(
        default=_YAML.get("logging", {}).get("level", "INFO")
    )
    log_file: str = Field(
        default=_YAML.get("logging", {}).get("log_file", "logs/app.log")
    )
    max_bytes: int = Field(
        default=_YAML.get("logging", {}).get("max_bytes", 10_485_760)
    )
    backup_count: int = Field(
        default=_YAML.get("logging", {}).get("backup_count", 5)
    )


class Settings(BaseSettings):
    """Top-level settings aggregating all subsystem configurations.

    Environment variable overrides use double-underscore as separator:
      SEARCH__BASE_URL=http://host:8080  overrides  search.base_url
    """

    model_config = SettingsConfigDict(
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    search: SearchSettings = Field(default_factory=SearchSettings)
    source_policy: SourcePolicySettings = Field(
        default_factory=SourcePolicySettings
    )
    preprocessing: PreprocessingSettings = Field(
        default_factory=PreprocessingSettings
    )
    claim_extraction: ClaimExtractionSettings = Field(
        default_factory=ClaimExtractionSettings
    )
    verification: VerificationSettings = Field(
        default_factory=VerificationSettings
    )
    logging: LoggingSettings = Field(default_factory=LoggingSettings)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton Settings instance (cached after first call)."""
    return Settings()
