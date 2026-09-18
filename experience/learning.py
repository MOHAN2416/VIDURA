"""VIDURA Controlled Learning Layer (Phase 11).

Derives validated, empirical lessons and identifies recurring patterns from past experiences.
STRICT INVARIANTS:
1. NO model weight modifications or fine-tuning.
2. NO autonomous modification of system instructions, permissions, or security boundaries.
3. Lessons are empirical and evidence-weighted; high confidence requires repeated verified evidence.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from memory.models import ExperienceRecord
from memory.store import MemoryStore
from experience.models import LessonRecord

logger = logging.getLogger("VIDURA.experience.learning")


class LearningLayer:
    """Controlled, deterministic learning layer analyzing empirical experiences."""

    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def derive_lessons(self, component: str | None = None, limit: int = 50) -> list[LessonRecord]:
        """Derives evidence-backed lessons grouped by affected components and failure/success patterns."""
        exps = self.store.list_experiences(limit=limit)
        if component:
            comp_norm = component.strip().lower()
            exps = [
                e for e in exps
                if any(comp_norm in f.lower() for f in e.affected_files)
                or any(comp_norm in c.lower() for c in e.affected_components)
            ]

        # Group experiences by primary component or task type
        groups: dict[str, list[ExperienceRecord]] = defaultdict(list)
        for e in exps:
            key = e.affected_files[0] if e.affected_files else (e.task_type or "general")
            groups[key].append(e)

        derived: list[LessonRecord] = []
        for topic, records in groups.items():
            successes = [r for r in records if r.success]
            failures = [r for r in records if not r.success]

            success_count = len(successes)
            failure_count = len(failures)
            total_count = success_count + failure_count

            if total_count == 0:
                continue

            # Compute empirical confidence:
            # 1 record -> 0.35 (low)
            # 2-3 records -> 0.65 (medium)
            # 4+ records -> 0.85 - 0.95 (high)
            if total_count == 1:
                confidence = 0.40
            elif total_count <= 3:
                confidence = 0.65
            elif total_count <= 6:
                confidence = 0.85
            else:
                confidence = 0.95

            # Synthesize representative lesson & recommendation
            if success_count > failure_count:
                rep = successes[0]
                lesson_text = rep.lesson or f"Modifications to '{topic}' verified successfully with test coverage."
                rec_text = rep.recommendation or f"Follow verified implementation patterns when modifying '{topic}'."
            else:
                rep = failures[0]
                reason = rep.failure_reason or rep.failure_type or "verification or test failure"
                lesson_text = rep.lesson or f"Modifications to '{topic}' frequently encountered {reason}."
                rec_text = rep.recommendation or f"Exercise caution with '{topic}'; run full regression tests and inspect callers."

            source_ids = [r.id for r in records]

            derived.append(
                LessonRecord(
                    topic=topic,
                    lesson=lesson_text,
                    task_type=records[0].task_type or "developer",
                    component=topic if "/" in topic or "." in topic else None,
                    source_experience_ids=source_ids,
                    confidence=confidence,
                    success_count=success_count,
                    failure_count=failure_count,
                    recommendation=rec_text,
                )
            )

        # Sort by confidence DESC, total evidence DESC
        derived.sort(key=lambda l: (l.confidence, l.success_count + l.failure_count), reverse=True)
        return derived

    def get_common_failures(self, limit: int = 10) -> list[dict[str, Any]]:
        """Identifies recurring failure patterns across all stored experiences."""
        failures = self.store.list_experiences(success=False, limit=100)
        pattern_counts: dict[str, list[ExperienceRecord]] = defaultdict(list)

        for f in failures:
            ft = f.failure_type or "general_failure"
            pattern_counts[ft].append(f)

        summary: list[dict[str, Any]] = []
        for f_type, recs in pattern_counts.items():
            affected = list(dict.fromkeys([file for r in recs for file in r.affected_files]))
            sample_lesson = recs[0].lesson if recs else ""
            summary.append({
                "failure_type": f_type,
                "occurrence_count": len(recs),
                "affected_files": affected[:5],
                "sample_lesson": sample_lesson,
                "recommendation": recs[0].recommendation if recs else "",
            })

        summary.sort(key=lambda s: s["occurrence_count"], reverse=True)
        return summary[:limit]

    def get_component_cautions(self, component_name: str) -> list[str]:
        """Returns specific cautions and negative lessons for a named component."""
        if not component_name:
            return []
        c_lower = component_name.lower()
        failures = self.store.list_experiences(success=False, limit=50)

        cautions: list[str] = []
        for f in failures:
            if any(c_lower in aff.lower() for aff in f.affected_files) or any(c_lower in comp.lower() for comp in f.affected_components):
                msg = f.recommendation or f.lesson or f.failure_reason
                if msg and msg not in cautions:
                    cautions.append(msg)

        return cautions

    def validate_lesson(self, lesson: LessonRecord) -> dict[str, Any]:
        """Validates a derived lesson against actual stored empirical evidence."""
        source_exps: list[ExperienceRecord] = []
        for eid in lesson.source_experience_ids:
            exp = self.store.get_experience(eid)
            if exp:
                source_exps.append(exp)

        actual_success = sum(1 for e in source_exps if e.success)
        actual_failure = sum(1 for e in source_exps if not e.success)
        is_supported = len(source_exps) > 0

        computed_conf = 0.3
        if len(source_exps) >= 5:
            computed_conf = 0.9
        elif len(source_exps) >= 2:
            computed_conf = 0.7
        elif len(source_exps) == 1:
            computed_conf = 0.45

        return {
            "lesson_id": lesson.lesson_id,
            "topic": lesson.topic,
            "is_supported_by_evidence": is_supported,
            "evidence_records_found": len(source_exps),
            "expected_successes": lesson.success_count,
            "actual_successes": actual_success,
            "expected_failures": lesson.failure_count,
            "actual_failures": actual_failure,
            "validated_confidence": computed_conf,
            "is_authoritative_rule": False,  # Lessons are NEVER authoritative rules
        }
