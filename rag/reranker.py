"""VIDURA Hybrid Reranker (Phase 12).

Combines lexical search, semantic vector similarity, and deterministic AST/codebase
signals using Reciprocal Rank Fusion (RRF) and explainable multi-signal scoring.
"""
from __future__ import annotations

import logging
from typing import Any

from rag.models import CodeChunk, RetrievedChunk, RAGQuery

logger = logging.getLogger("VIDURA.rag.reranker")


class HybridReranker:
    """Reranks candidate chunks by fusing lexical, semantic, symbol, and dependency signals."""

    RRF_K = 60
    WEIGHT_LEXICAL_RRF = 1.0
    WEIGHT_SEMANTIC_RRF = 1.0

    BONUS_SYMBOL_MATCH = 0.25
    BONUS_FILE_MATCH = 0.20
    BONUS_DEPENDENCY_MATCH = 0.15

    def rerank(
        self,
        lexical_results: list[tuple[CodeChunk, float]],
        semantic_results: list[tuple[CodeChunk, float]],
        query: RAGQuery,
        dependencies: set[str] | None = None,
    ) -> list[RetrievedChunk]:
        """Fuses ranked candidate lists into a single ranked list of RetrievedChunk objects."""
        deps = dependencies or set()
        chunk_map: dict[str, CodeChunk] = {}
        lex_ranks: dict[str, int] = {}
        sem_ranks: dict[str, int] = {}
        lex_scores: dict[str, float] = {}
        sem_scores: dict[str, float] = {}

        # 1. Index lexical ranks
        for rank, (chunk, score) in enumerate(lexical_results, start=1):
            chunk_map[chunk.chunk_id] = chunk
            lex_ranks[chunk.chunk_id] = rank
            lex_scores[chunk.chunk_id] = score

        # 2. Index semantic ranks
        for rank, (chunk, score) in enumerate(semantic_results, start=1):
            chunk_map[chunk.chunk_id] = chunk
            sem_ranks[chunk.chunk_id] = rank
            sem_scores[chunk.chunk_id] = score

        target_files_norm = {f.strip().lstrip("./").lower() for f in query.target_files}
        if query.file_filter:
            target_files_norm.add(query.file_filter.strip().lstrip("./").lower())

        target_symbols_lower = {s.strip().lower() for s in query.target_symbols}
        if query.symbol_filter:
            target_symbols_lower.add(query.symbol_filter.strip().lower())

        reranked: list[RetrievedChunk] = []

        # 3. Compute fused RRF score + deterministic bonuses
        for chunk_id, chunk in chunk_map.items():
            breakdown: dict[str, float] = {}

            # RRF Component
            rrf_lex = 0.0
            if chunk_id in lex_ranks:
                rrf_lex = self.WEIGHT_LEXICAL_RRF / (self.RRF_K + lex_ranks[chunk_id])

            rrf_sem = 0.0
            if chunk_id in sem_ranks:
                rrf_sem = self.WEIGHT_SEMANTIC_RRF / (self.RRF_K + sem_ranks[chunk_id])

            rrf_score = rrf_lex + rrf_sem
            # Scale RRF up into roughly [0.0, 0.40] for fusion
            scaled_rrf = min(0.40, rrf_score * 12.0)
            breakdown["rrf_score"] = scaled_rrf

            l_score = lex_scores.get(chunk_id, 0.0)
            s_score = sem_scores.get(chunk_id, 0.0)
            breakdown["lexical"] = l_score * 0.20
            breakdown["semantic"] = s_score * 0.20

            # Deterministic File Match Bonus
            file_bonus = 0.0
            rf_norm = chunk.relative_path.strip().lstrip("./").lower()
            if target_files_norm:
                if any(tf == rf_norm or rf_norm.endswith(tf) or tf.endswith(rf_norm) for tf in target_files_norm):
                    file_bonus = self.BONUS_FILE_MATCH
            breakdown["file_bonus"] = file_bonus

            # Deterministic Symbol Match Bonus
            symbol_bonus = 0.0
            if target_symbols_lower and chunk.symbol_name:
                sym_lower = chunk.symbol_name.lower()
                if any(ts in sym_lower for ts in target_symbols_lower):
                    symbol_bonus = self.BONUS_SYMBOL_MATCH
            breakdown["symbol_bonus"] = symbol_bonus

            # Deterministic Dependency Bonus
            dep_bonus = 0.0
            if deps:
                for dep in deps:
                    d_norm = dep.strip().lstrip("./").lower()
                    if d_norm in rf_norm:
                        dep_bonus = self.BONUS_DEPENDENCY_MATCH
                        break
            breakdown["dependency_bonus"] = dep_bonus

            total_score = sum(breakdown.values())
            bounded_score = min(1.0, max(0.0, total_score))

            if bounded_score >= query.min_score:
                reranked.append(
                    RetrievedChunk(
                        chunk=chunk,
                        score=bounded_score,
                        lexical_score=l_score,
                        semantic_score=s_score,
                        symbol_score=symbol_bonus,
                        dependency_score=dep_bonus,
                        rrf_score=rrf_score,
                        score_breakdown=breakdown,
                    )
                )

        # Sort by total score DESC, then start_line ASC
        reranked.sort(key=lambda r: (r.score, -r.chunk.start_line), reverse=True)
        return reranked[: query.top_k]
