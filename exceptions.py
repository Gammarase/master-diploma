"""
Custom exception hierarchy for the Disinformation Detection System.

All exceptions inherit from DisinfoDetectionError, allowing callers to
catch the entire family or specific subtypes.
"""

from __future__ import annotations

__all__ = [
    "DisinfoDetectionError",
    "PreprocessingError",
    "ClaimExtractionError",
    "EmbeddingError",
    "VectorStoreError",
    "PineconeConnectionError",
    "PineconeUpsertError",
    "VerificationError",
    "NLIError",
    "RAGVerifierError",
    "OllamaConnectionError",
    "ExplainabilityError",
]


class DisinfoDetectionError(Exception):
    """Base exception for all disinformation detection errors."""

    __slots__ = ("message", "original_error")

    def __init__(
        self,
        message: str,
        original_error: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.original_error = original_error

    def __str__(self) -> str:
        if self.original_error:
            return f"{self.message} (caused by: {self.original_error})"
        return self.message


class PreprocessingError(DisinfoDetectionError):
    """Raised when text preprocessing fails."""


class ClaimExtractionError(DisinfoDetectionError):
    """Raised when claim extraction fails."""


class EmbeddingError(DisinfoDetectionError):
    """Raised when embedding generation fails."""


class VectorStoreError(DisinfoDetectionError):
    """Raised for general vector store failures."""


class PineconeConnectionError(VectorStoreError):
    """Raised when connection to Pinecone cannot be established."""


class PineconeUpsertError(VectorStoreError):
    """Raised when upserting vectors to Pinecone fails."""


class VerificationError(DisinfoDetectionError):
    """Raised for general verification failures."""


class NLIError(VerificationError):
    """Raised when NLI inference fails."""


class RAGVerifierError(VerificationError):
    """Raised when RAG-based verification fails."""


class OllamaConnectionError(RAGVerifierError):
    """Raised when Ollama is unreachable or returns an unexpected error."""


class ExplainabilityError(DisinfoDetectionError):
    """Raised when explanation generation fails."""
