"""
Configuration management for the Disinformation Detection System.

Settings are loaded from configs/config.yaml and can be overridden via
environment variables (e.g. PINECONE_API_KEY overrides pinecone.api_key).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = [
    "PineconeSettings",
    "OllamaSettings",
    "EmbeddingSettings",
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


class PineconeSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PINECONE_")

    api_key: str = Field(default=_YAML.get("pinecone", {}).get("api_key", ""))
    index_name: str = Field(
        default=_YAML.get("pinecone", {}).get("index_name", "ru22fact-evidence")
    )
    cloud: str = Field(default=_YAML.get("pinecone", {}).get("cloud", "aws"))
    region: str = Field(
        default=_YAML.get("pinecone", {}).get("region", "us-east-1")
    )
    top_k: int = Field(default=_YAML.get("pinecone", {}).get("top_k", 5))


class OllamaSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OLLAMA_")

    base_url: str = Field(
        default=_YAML.get("ollama", {}).get("base_url", "http://localhost:11434")
    )
    model: str = Field(default=_YAML.get("ollama", {}).get("model", "llama3"))
    temperature: float = Field(
        default=_YAML.get("ollama", {}).get("temperature", 0.0)
    )
    timeout_seconds: int = Field(
        default=_YAML.get("ollama", {}).get("timeout_seconds", 60)
    )


class EmbeddingSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EMBEDDINGS_")

    model_name: str = Field(
        default=_YAML.get("embeddings", {}).get(
            "model_name",
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        )
    )
    batch_size: int = Field(
        default=_YAML.get("embeddings", {}).get("batch_size", 32)
    )
    device: str = Field(
        default=_YAML.get("embeddings", {}).get("device", "cpu")
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
            "spacy_models", {"en": "en_core_web_sm", "uk": "uk_core_news_sm"}
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
        default=_YAML.get("claim_extraction", {}).get("checkworthy_threshold", 0.5)
    )
    classifier_model: str = Field(
        default=_YAML.get("claim_extraction", {}).get(
            "classifier_model", "typeform/distilbert-base-uncased-mnli"
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
        .get("disinformation", 0.3)
    )
    confirmed: float = Field(
        default=_YAML.get("verification", {})
        .get("thresholds", {})
        .get("confirmed", 0.7)
    )


class VerificationSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VERIFICATION_")

    nli_model: str = Field(
        default=_YAML.get("verification", {}).get(
            "nli_model", "cross-encoder/nli-deberta-v3-base"
        )
    )
    nli_weight: float = Field(
        default=_YAML.get("verification", {}).get("nli_weight", 0.4)
    )
    rag_weight: float = Field(
        default=_YAML.get("verification", {}).get("rag_weight", 0.6)
    )
    thresholds: VerificationThresholds = Field(
        default_factory=VerificationThresholds
    )
    disagreement_delta: float = Field(
        default=_YAML.get("verification", {}).get("disagreement_delta", 0.3)
    )


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
      PINECONE__API_KEY=xxx  overrides  pinecone.api_key
    """

    model_config = SettingsConfigDict(
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    pinecone: PineconeSettings = Field(default_factory=PineconeSettings)
    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    embeddings: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
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
