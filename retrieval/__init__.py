"""
Retrieval module for the Disinformation Detection System.

Exports EmbeddingModel, Reranker, PineconeVectorStore, and related dataclasses.
"""

from retrieval.embeddings import EmbeddingModel
from retrieval.reranker import Reranker
from retrieval.vector_store import EvidenceDocument, PineconeVectorStore, RetrievedEvidence

__all__ = [
    "EmbeddingModel",
    "Reranker",
    "PineconeVectorStore",
    "EvidenceDocument",
    "RetrievedEvidence",
]
