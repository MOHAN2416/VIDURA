"""VIDURA High-Level Experience Manager (Phase 11).

Orchestrates storage, deterministic scoring, retrieval, deduplication,
and evidence-backed lesson derivation for VIDURA.
"""
from __future__ import annotations

import logging
from typing import Any

from memory.models import ExperienceRecord
from memory.store import MemoryStore
from experience.models import ExperienceQuery, ScoredExperience, LessonRecord
from experience.extractor import ExperienceExtractor
from experience.scorer import ExperienceScorer
from experience.retriever import ExperienceRetriever
from experience.learning import LearningLayer

logger = logging.getLogger("VIDURA.experience.manager")


class ExperienceManager:
    """High-level API for Experience Memory and Controlled Learning."""

    def __init__(
        self,
        store: MemoryStore,
        scorer: ExperienceScorer | None = None,
        retriever: ExperienceRetriever | None = None,
        learning_layer: LearningLayer | None = None,
    ) -> None:
        self.store = store
        self.scorer = scorer or ExperienceScorer()
        self.retriever = retriever or ExperienceRetriever(store=self.store, scorer=self.scorer)
        self.learning_layer = learning_layer or LearningLayer(store=self.store)
        self.extractor = ExperienceExtractor()

    def record_experience(self, record: ExperienceRecord) -> str:
        """Stores an experience record with deterministic deduplication and historical preservation.

        Failure Safety: If storage fails, catches error and returns empty string without crashing caller.
        """
        try:
            return self.deduplicate_or_record(record)
        except Exception as err:
            logger.error(f"Experience storage failed: {err}", exc_info=True)
            return ""

    def deduplicate_or_record(self, record: ExperienceRecord) -> str:
        """Checks for existing identical experiences. If duplicate, aggregates evidence; otherwise inserts."""
        # Find candidates by primary file or task keyword
        candidates: list[ExperienceRecord] = []
        if record.affected_files:
            candidates = self.store.search_experiences(query=record.affected_files[0], limit=10)
        elif record.task:
            candidates = self.store.search_experiences(query=record.task[:40], limit=10)

        primary_file = record.affected_files[0] if record.affected_files else ""

        for cand in candidates:
            cand_primary = cand.affected_files[0] if cand.affected_files else ""
            # Check for near-identical event
            same_target = (primary_file and primary_file == cand_primary)
            same_task = (record.task.strip().lower() == cand.task.strip().lower())
            same_outcome = (record.success == cand.success and record.failure_type == cand.failure_type)

            if same_target and same_task and same_outcome:
                # Aggregate duplicate evidence rather than spamming duplicate records
                cand.duplicate_count += 1
                cand.attempt += 1
                cand.evidence_count += record.evidence_count
                cand.confidence = min(1.0, cand.confidence + 0.05)
                self.store.update_experience(cand)
                logger.info(f"Aggregated duplicate experience ID '{cand.id}' (count={cand.duplicate_count}).")
                return cand.id

            elif same_target and (not cand.success) and record.success:
                # Milestone: Previously failed, now succeeded!
                # Preserve historical failure and link in related_experience_ids
                if cand.id not in record.related_experience_ids:
                    record.related_experience_ids.append(cand.id)
                prev_reason = cand.failure_reason or cand.failure_type or "unknown"
                record.lesson = (
                    f"Resolved previous failure ({prev_reason}): {record.lesson or 'Change applied and verified.'}"
                )
                logger.info(f"Linked previous failure '{cand.id}' to new successful experience '{record.id}'.")

        # Standard insert
        return self.store.add_experience(record)

    def retrieve_experiences(self, query: ExperienceQuery) -> list[ScoredExperience]:
        """Retrieves and ranks experiences matching the query filters."""
        return self.retriever.retrieve(query)

    def get_experience(self, experience_id: str) -> ExperienceRecord | None:
        """Retrieves a single ExperienceRecord by ID."""
        return self.store.get_experience(experience_id)

    def list_experiences(
        self,
        task_type: str | None = None,
        success: bool | None = None,
        limit: int = 50,
    ) -> list[ExperienceRecord]:
        """Lists experience records with optional filtering."""
        return self.store.list_experiences(task_type=task_type, success=success, limit=limit)

    def delete_experience(self, experience_id: str) -> bool:
        """Permanently deletes an ExperienceRecord by ID."""
        return self.store.delete_experience(experience_id)

    def derive_lessons(self, component: str | None = None, limit: int = 50) -> list[LessonRecord]:
        """Derives evidence-backed lessons from stored experiences."""
        return self.learning_layer.derive_lessons(component=component, limit=limit)

    def get_relevant_lessons(
        self, component: str | None = None, task_type: str | None = None
    ) -> list[LessonRecord]:
        """Returns lessons relevant to a component or task type."""
        lessons = self.learning_layer.derive_lessons(component=component)
        if task_type:
            tt_lower = task_type.lower()
            lessons = [l for l in lessons if tt_lower in l.task_type.lower()]
        return lessons

    def validate_lesson(self, lesson: LessonRecord) -> dict[str, Any]:
        """Validates lesson claims against actual empirical stored records."""
        return self.learning_layer.validate_lesson(lesson)

    def get_context_for_developer_planning(
        self,
        goal: str,
        files: list[str] | None = None,
        symbols: list[str] | None = None,
        task_type: str = "developer",
        limit: int = 5,
    ) -> str:
        """Retrieves relevant past experiences and formats advisory prompt context for developer planning."""
        target_file = files[0] if files else None
        target_symbol = symbols[0] if symbols else None

        query = ExperienceQuery(
            query=goal,
            task_type=task_type,
            file_path=target_file,
            symbol=target_symbol,
            limit=limit,
            min_score=0.15,
        )
        scored = self.retrieve_experiences(query)
        return self.retriever.format_prompt_context(scored, include_disclaimer=True)

    def get_context_for_self_development(
        self,
        goal_description: str,
        scope: list[str] | None = None,
        limit: int = 5,
    ) -> str:
        """Retrieves relevant past self-development experiences and lessons."""
        target_file = scope[0] if scope else None
        query = ExperienceQuery(
            query=goal_description,
            task_type="self_development",
            file_path=target_file,
            limit=limit,
            min_score=0.15,
        )
        scored = self.retrieve_experiences(query)
        return self.retriever.format_prompt_context(scored, include_disclaimer=True)
