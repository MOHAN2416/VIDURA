"""VIDURA Deterministic Experience Scorer (Phase 11).

Calculates explainable, deterministic relevance scores using structured signals.
Strictly avoids semantic embeddings, vector databases, and RAG.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any

from memory.models import ExperienceRecord
from experience.models import ExperienceQuery, ScoredExperience

logger = logging.getLogger("VIDURA.experience.scorer")

STOP_WORDS = {
    "a", "an", "the", "in", "on", "of", "to", "for", "with", "at", "by", "from",
    "up", "about", "into", "over", "after", "is", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "and", "or",
    "but", "if", "then", "else", "when", "where", "why", "how", "all", "any",
    "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very", "can", "will",
    "just", "should", "now", "vidura", "test", "task",
}


def tokenize(text: str) -> set[str]:
    """Tokenizes text into normalized alphanumeric keywords excluding stop words."""
    if not text:
        return set()
    words = re.findall(r"[a-zA-Z0-9_\-\.]{3,}", text.lower())
    return {w.strip(".") for w in words if w not in STOP_WORDS and len(w) > 2}


class ExperienceScorer:
    """Computes deterministic, explainable relevance scores for past experiences."""

    WEIGHT_FILE_MATCH = 0.30
    WEIGHT_SYMBOL_MATCH = 0.25
    WEIGHT_TASK_MATCH = 0.20
    WEIGHT_FAILURE_MATCH = 0.20
    WEIGHT_KEYWORD_MATCH = 0.20
    WEIGHT_COMPONENT_MATCH = 0.15
    WEIGHT_OUTCOME_MATCH = 0.10
    MAX_RECENCY_BONUS = 0.05

    def score(self, record: ExperienceRecord, query: ExperienceQuery) -> ScoredExperience:
        """Calculates relevance score and breakdown for an experience against a query."""
        breakdown: dict[str, float] = {}

        # 1. File Path Match (0.30)
        file_score = 0.0
        if query.file_path and record.affected_files:
            q_file_norm = query.file_path.strip().lstrip("./").lower()
            for rf in record.affected_files:
                rf_norm = rf.strip().lstrip("./").lower()
                if q_file_norm == rf_norm:
                    file_score = self.WEIGHT_FILE_MATCH
                    break
                elif q_file_norm.endswith(rf_norm) or rf_norm.endswith(q_file_norm):
                    file_score = max(file_score, self.WEIGHT_FILE_MATCH * 0.8)
                elif q_file_norm.split("/")[-1] == rf_norm.split("/")[-1]:
                    file_score = max(file_score, self.WEIGHT_FILE_MATCH * 0.6)
        breakdown["file_match"] = file_score

        # 2. Symbol Match (0.25)
        symbol_score = 0.0
        if query.symbol and record.relevant_symbols:
            q_sym = query.symbol.strip().lower()
            for sym in record.relevant_symbols:
                s_lower = sym.strip().lower()
                if q_sym == s_lower:
                    symbol_score = self.WEIGHT_SYMBOL_MATCH
                    break
                elif q_sym in s_lower or s_lower in q_sym:
                    symbol_score = max(symbol_score, self.WEIGHT_SYMBOL_MATCH * 0.7)
        breakdown["symbol_match"] = symbol_score

        # 3. Task Type Match (0.20)
        task_score = 0.0
        if query.task_type:
            q_tt = query.task_type.strip().lower()
            r_tt = (record.task_type or "").strip().lower()
            if q_tt == r_tt:
                task_score = self.WEIGHT_TASK_MATCH
            elif q_tt in r_tt or r_tt in q_tt:
                task_score = self.WEIGHT_TASK_MATCH * 0.7
        breakdown["task_match"] = task_score

        # 4. Failure Type Match (0.20)
        failure_score = 0.0
        if query.failure_type and record.failure_type:
            q_ft = query.failure_type.strip().lower()
            r_ft = record.failure_type.strip().lower()
            if q_ft == r_ft:
                failure_score = self.WEIGHT_FAILURE_MATCH
        breakdown["failure_match"] = failure_score

        # 5. Component / Module Match (0.15)
        component_score = 0.0
        target_comp = query.component or query.module
        if target_comp and (record.affected_components or record.affected_files):
            q_comp = target_comp.strip().lower().lstrip("./")
            for c in record.affected_components:
                c_norm = c.strip().lower().lstrip("./")
                if q_comp == c_norm or q_comp in c_norm or c_norm in q_comp:
                    component_score = self.WEIGHT_COMPONENT_MATCH
                    break
            if not component_score:
                for f in record.affected_files:
                    f_norm = f.strip().lower().lstrip("./")
                    if q_comp in f_norm:
                        component_score = self.WEIGHT_COMPONENT_MATCH * 0.8
                        break
        breakdown["component_match"] = component_score

        # 6. Keyword Overlap Match (0.20)
        keyword_score = 0.0
        if query.query and query.query.strip():
            q_tokens = tokenize(query.query)
            if q_tokens:
                rec_text = f"{record.task} {record.goal} {record.lesson} {record.recommendation} {record.failure_reason or ''}"
                r_tokens = tokenize(rec_text)
                overlap = q_tokens.intersection(r_tokens)
                if overlap:
                    ratio = len(overlap) / len(q_tokens)
                    keyword_score = min(self.WEIGHT_KEYWORD_MATCH, self.WEIGHT_KEYWORD_MATCH * ratio)
        breakdown["keyword_match"] = keyword_score

        # 7. Outcome Match (0.10)
        outcome_score = 0.0
        if query.success is not None:
            if query.success == record.success:
                outcome_score = self.WEIGHT_OUTCOME_MATCH
        breakdown["outcome_match"] = outcome_score

        # 8. Recency Bonus (Max 0.05, bounded)
        recency_bonus = 0.0
        try:
            created = datetime.fromisoformat(record.created_at)
            now = datetime.now(timezone.utc)
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            delta_hours = max(0.0, (now - created).total_seconds() / 3600.0)
            if delta_hours < 1.0:
                recency_bonus = self.MAX_RECENCY_BONUS
            elif delta_hours < 24.0:
                recency_bonus = self.MAX_RECENCY_BONUS * 0.6
            elif delta_hours < 168.0:
                recency_bonus = self.MAX_RECENCY_BONUS * 0.3
            else:
                recency_bonus = self.MAX_RECENCY_BONUS * 0.1
        except Exception:
            recency_bonus = 0.0
        breakdown["recency_bonus"] = recency_bonus

        total_score = sum(breakdown.values())
        # Bound score between 0.0 and 1.0
        final_score = min(1.0, max(0.0, total_score))

        is_warning = (not record.success) or (record.failure_type is not None)

        return ScoredExperience(
            experience=record,
            score=final_score,
            score_breakdown=breakdown,
            is_warning=is_warning,
        )
