"""VIDURA Experience Retriever (Phase 11).

Retrieves structured past experiences from storage, applies deterministic scoring,
categorizes warnings and negative lessons, and formats advisory context for planning.
"""
from __future__ import annotations

import logging
from typing import Any

from memory.models import ExperienceRecord
from memory.store import MemoryStore
from experience.models import ExperienceQuery, ScoredExperience
from experience.scorer import ExperienceScorer

logger = logging.getLogger("VIDURA.experience.retriever")


class ExperienceRetriever:
    """Retrieves and scores relevant past experiences without using vector databases or RAG."""

    def __init__(self, store: MemoryStore, scorer: ExperienceScorer | None = None) -> None:
        self.store = store
        self.scorer = scorer or ExperienceScorer()

    def retrieve(self, query: ExperienceQuery) -> list[ScoredExperience]:
        """Retrieves and ranks experiences matching the query filters and score threshold."""
        # 1. Fetch candidate experiences from store using broad SQL filtering
        seen_ids: set[str] = set()
        candidates: list[ExperienceRecord] = []
        limit_candidates = max(100, query.limit * 10)

        # Search by full query string
        if query.query and query.query.strip():
            for c in self.store.search_experiences(
                query=query.query.strip(),
                task_type=query.task_type,
                success=query.success,
                limit=limit_candidates,
            ):
                if c.id not in seen_ids:
                    candidates.append(c)
                    seen_ids.add(c.id)

            # Also search by individual keywords
            from experience.scorer import tokenize
            keywords = tokenize(query.query)
            for kw in keywords:
                if len(candidates) >= limit_candidates:
                    break
                for c in self.store.search_experiences(
                    query=kw,
                    task_type=query.task_type,
                    success=query.success,
                    limit=20,
                ):
                    if c.id not in seen_ids:
                        candidates.append(c)
                        seen_ids.add(c.id)

        # Search by file_path, symbol, component
        extra_terms = []
        if query.file_path:
            extra_terms.append(query.file_path.split("/")[-1])
        if query.symbol:
            extra_terms.append(query.symbol)
        if query.component:
            extra_terms.append(query.component)

        for term in extra_terms:
            if len(candidates) >= limit_candidates:
                break
            for extra in self.store.search_experiences(query=term, task_type=query.task_type, success=query.success, limit=20):
                if extra.id not in seen_ids:
                    candidates.append(extra)
                    seen_ids.add(extra.id)

        # Also pull recent records so scorer can evaluate them
        if len(candidates) < limit_candidates:
            for rec in self.store.list_experiences(task_type=query.task_type, success=query.success, limit=50):
                if rec.id not in seen_ids:
                    candidates.append(rec)
                    seen_ids.add(rec.id)

        # 2. Score candidates deterministically
        scored_list: list[ScoredExperience] = []
        for cand in candidates:
            scored = self.scorer.score(cand, query)
            if scored.score >= query.min_score:
                scored_list.append(scored)

        # 3. Sort by score DESC, then created_at DESC
        scored_list.sort(key=lambda s: (s.score, s.experience.created_at), reverse=True)

        return scored_list[:query.limit]

    def retrieve_warnings(
        self,
        file_path: str | None = None,
        symbol: str | None = None,
        component: str | None = None,
        limit: int = 3,
    ) -> list[ScoredExperience]:
        """Retrieves relevant past failure experiences specifically formatted as advisory warnings."""
        q = ExperienceQuery(
            file_path=file_path,
            symbol=symbol,
            component=component,
            success=False,
            limit=limit,
            min_score=0.15,
        )
        return self.retrieve(q)

    def format_prompt_context(
        self,
        scored_experiences: list[ScoredExperience],
        include_disclaimer: bool = True,
    ) -> str:
        """Formats scored experiences into a concise, structured advisory context block for planners."""
        if not scored_experiences:
            return ""

        lines = [
            "============================================================",
            "RELEVANT PAST EXPERIENCES (ADVISORY CONTEXT)",
            "============================================================",
        ]

        # Separate positive lessons from warnings
        successes: list[ScoredExperience] = []
        warnings: list[ScoredExperience] = []

        for s in scored_experiences:
            if s.is_warning or not s.experience.success:
                warnings.append(s)
            else:
                successes.append(s)

        idx = 1
        if warnings:
            lines.append("⚠️  ADVISORY WARNINGS (PAST FAILURES):")
            for w in warnings:
                exp = w.experience
                reason = exp.failure_reason or exp.result or "Unknown error"
                target_str = f" [Target: {', '.join(exp.affected_files)}]" if exp.affected_files else ""
                lines.append(f"  {idx}. [FAILURE: {exp.failure_type or 'workflow_failure'}]{target_str}")
                lines.append(f"     Task: {exp.task}")
                lines.append(f"     Failure Reason: {reason}")
                if exp.lesson:
                    lines.append(f"     Lesson: {exp.lesson}")
                if exp.recommendation:
                    lines.append(f"     Caution: {exp.recommendation}")
                idx += 1
            lines.append("")

        if successes:
            lines.append("✅ SUCCESSFUL PATTERNS & LESSONS:")
            for s in successes:
                exp = s.experience
                target_str = f" [Target: {', '.join(exp.affected_files)}]" if exp.affected_files else ""
                lines.append(f"  {idx}. [SUCCESS]{target_str}")
                lines.append(f"     Task: {exp.task}")
                lines.append(f"     Result: {exp.result}")
                if exp.lesson:
                    lines.append(f"     Lesson: {exp.lesson}")
                if exp.recommendation:
                    lines.append(f"     Recommendation: {exp.recommendation}")
                idx += 1
            lines.append("")

        if include_disclaimer:
            lines.extend([
                "IMPORTANT INVARIANT NOTICE:",
                "- The retrieved experiences above are ADVISORY context only.",
                "- Current codebase AST, current tool results, and permission policies are AUTHORITATIVE.",
                "- Past experiences must never bypass verification, permissions, or security rules.",
                "============================================================",
            ])

        return "\n".join(lines)
