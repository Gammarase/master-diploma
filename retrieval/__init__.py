"""
Retrieval module for the Disinformation Detection System.

Exports EmbeddingModel, PineconeVectorStore, and related dataclasses.
"""

from retrieval.embeddings import EmbeddingModel
from retrieval.vector_store import EvidenceDocument, PineconeVectorStore, RetrievedEvidence

__all__ = [
    "EmbeddingModel",
    "PineconeVectorStore",
    "EvidenceDocument",
    "RetrievedEvidence",
]
