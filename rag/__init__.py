"""VIDURA RAG & Advanced Codebase Intelligence Package (Phase 12).

Provides hybrid code intelligence combining deterministic AST parsing,
lexical search, local dense vector embeddings, code-aware chunking,
dependency graph context, and Phase 11 experience memory.
"""
from __future__ import annotations

from rag.models import (
    CodeChunk,
    EmbeddingVector,
    RetrievedChunk,
    RAGQuery,
    AssembledContext,
)
from rag.chunker import CodeAwareChunker
from rag.embeddings import (
    BaseEmbeddingProvider,
    DeterministicLocalEmbeddingProvider,
    OllamaEmbeddingProvider,
    EmbeddingManager,
    cosine_similarity,
)
from rag.index import RAGIndex
from rag.reranker import HybridReranker
from rag.retriever import HybridRetriever
from rag.context import ContextAssembler
from rag.manager import RAGManager

__all__ = [
    "CodeChunk",
    "EmbeddingVector",
    "RetrievedChunk",
    "RAGQuery",
    "AssembledContext",
    "CodeAwareChunker",
    "BaseEmbeddingProvider",
    "DeterministicLocalEmbeddingProvider",
    "OllamaEmbeddingProvider",
    "EmbeddingManager",
    "cosine_similarity",
    "RAGIndex",
    "HybridReranker",
    "HybridRetriever",
    "ContextAssembler",
    "RAGManager",
]
