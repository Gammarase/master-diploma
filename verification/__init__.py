"""
Verification module for the Disinformation Detection System.

Exports NLIVerifier, StanceDetector, RAGVerifier, OllamaClient,
ResultAggregator, and related dataclasses.
"""

from verification.aggregator import ResultAggregator, VerificationResult
from verification.nli_verifier import NLIResult, NLIVerifier
from verification.rag_verifier import OllamaClient, RAGVerdict, RAGVerifier
from verification.stance_detector import StanceDetector, StanceResult

__all__ = [
    "NLIVerifier",
    "NLIResult",
    "StanceDetector",
    "StanceResult",
    "OllamaClient",
    "RAGVerifier",
    "RAGVerdict",
    "ResultAggregator",
    "VerificationResult",
]
