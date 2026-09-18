"""VIDURA Hybrid Retriever (Phase 12).

Orchestrates deterministic AST lookup, lexical inverted search, dense vector semantic search,
dependency graph context, and Phase 11 experience memory retrieval.
"""
from __future__ import annotations

import logging
from typing import Any

from rag.models import CodeChunk, RetrievedChunk, RAGQuery
from rag.index import RAGIndex
from rag.embeddings import EmbeddingManager
from rag.reranker import HybridReranker

logger = logging.getLogger("VIDURA.rag.retriever")


class HybridRetriever:
    """Coordinates hybrid retrieval combining AST, lexical, semantic, dependency, and experience signals."""

    def __init__(
        self,
        index: RAGIndex,
        embedding_manager: EmbeddingManager,
        reranker: HybridReranker | None = None,
        codebase_manager: Any = None,
        experience_manager: Any = None,
    ) -> None:
        self.index = index
        self.embedding_manager = embedding_manager
        self.reranker = reranker or HybridReranker()
        self.codebase_manager = codebase_manager
        self.experience_manager = experience_manager

    def retrieve(self, query: RAGQuery) -> list[RetrievedChunk]:
        """Executes multi-signal hybrid retrieval and returns reranked, filtered code chunks."""
        candidate_limit = max(30, query.top_k * 4)

        # 1. Lexical Search
        lexical_matches: list[tuple[CodeChunk, float]] = []
        if query.query and query.query.strip():
            lexical_matches = self.index.search_lexical(query.query, limit=candidate_limit)

        # 2. Semantic Search
        semantic_matches: list[tuple[CodeChunk, float]] = []
        if query.query and query.query.strip():
            try:
                q_vec = self.embedding_manager.embed_text(query.query)
                semantic_matches = self.index.search_semantic(
                    query_vector=q_vec,
                    limit=candidate_limit,
                    min_similarity=0.05,
                )
            except Exception as emb_err:
                logger.warning(f"Semantic search embedding failed: {emb_err}")

        # 3. Deterministic AST / Symbol & File Injections
        # If target files or symbols are specified in query, ensure their chunks are candidates
        extra_chunks: list[tuple[CodeChunk, float]] = []
        for tf in query.target_files:
            file_chunks = self.index.list_chunks(relative_path=tf, limit=10)
            for fc in file_chunks:
                extra_chunks.append((fc, 0.8))

        for sym in query.target_symbols:
            sym_chunks = self.index.list_chunks(symbol_name=sym, limit=5)
            for sc in sym_chunks:
                extra_chunks.append((sc, 0.9))

        if extra_chunks:
            lexical_matches.extend(extra_chunks)

        # 4. Dependency Graph Context Expansion
        dependencies: set[str] = set()
        if self.codebase_manager and query.include_dependencies:
            for tf in query.target_files:
                mod_name = tf.replace(".py", "").replace("/", ".")
                try:
                    for dep in self.codebase_manager.find_dependencies(mod_name):
                        dependencies.add(dep)
                    for imp in self.codebase_manager.find_importers(mod_name):
                        dependencies.add(imp)
                except Exception:
                    pass

        # 5. Hybrid Reranking
        return self.reranker.rerank(
            lexical_results=lexical_matches,
            semantic_results=semantic_matches,
            query=query,
            dependencies=dependencies,
        )

    def retrieve_relevant_experiences(
        self,
        query_text: str,
        target_files: list[str] | None = None,
        target_symbols: list[str] | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Retrieves Phase 11 experiences relevant to current query."""
        if not self.experience_manager:
            return []
        try:
            from experience.models import ExperienceQuery
            eq = ExperienceQuery(
                query=query_text,
                file_path=target_files[0] if target_files else None,
                symbol=target_symbols[0] if target_symbols else None,
                limit=limit,
                min_score=0.15,
            )
            scored = self.experience_manager.retrieve_experiences(eq)
            return [s.to_dict() for s in scored]
        except Exception as err:
            logger.warning(f"Failed to retrieve past experiences in RAG retriever: {err}")
            return []
