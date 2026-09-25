"""
Retrieval module for the Disinformation Detection System.

Exports the web evidence retriever, the source policy, the reranker, the
passage chunker and the evidence dataclasses.
"""

from retrieval.chunking import chunk_evidence
from retrieval.evidence import RetrievalResult, RetrievedEvidence
from retrieval.reranker import Reranker
from retrieval.source_policy import SourcePolicy
from retrieval.web_retriever import WebEvidenceRetriever

__all__ = [
    "RetrievedEvidence",
    "RetrievalResult",
    "WebEvidenceRetriever",
    "SourcePolicy",
    "Reranker",
    "chunk_evidence",
]
