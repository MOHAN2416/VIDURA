"""VIDURA Context Assembler (Phase 12).

Formats structured, budget-controlled context packets combining retrieved code chunks,
AST symbols, dependency graphs, and Phase 11 experience memory with authoritative disclaimers.
"""
from __future__ import annotations

import logging
from typing import Any

from rag.models import RetrievedChunk, AssembledContext

logger = logging.getLogger("VIDURA.rag.context")


class ContextAssembler:
    """Assembles retrieved codebase chunks, dependencies, and experiences into a structured prompt packet."""

    DISCLAIMER = (
        "============================================================\n"
        "AUTHORITATIVE SYSTEM INVARIANT NOTICE:\n"
        "- The codebase snippets and experiences below are RETRIEVED ADVISORY CONTEXT.\n"
        "- RAG is a retrieval system, NOT the source of truth.\n"
        "- The actual filesystem, AST parser, tool results, permissions, and\n"
        "  post-change verification system remain strictly AUTHORITATIVE.\n"
        "============================================================"
    )

    def assemble(
        self,
        retrieved_chunks: list[RetrievedChunk] | None = None,
        target_files: list[str] | None = None,
        target_symbols: list[str] | None = None,
        dependencies: list[str] | None = None,
        relevant_experiences: list[dict[str, Any]] | None = None,
        max_context_tokens: int = 4000,
        chunks: list[RetrievedChunk] | None = None,
        max_tokens: int | None = None,
    ) -> AssembledContext:
        """Assembles a budget-constrained, strongly-typed AssembledContext."""
        actual_chunks = chunks if chunks is not None else (retrieved_chunks or [])
        budget_tokens = max_tokens if max_tokens is not None else max_context_tokens
        t_files = list(dict.fromkeys(target_files or []))
        t_symbols = list(dict.fromkeys(target_symbols or []))
        deps = list(dict.fromkeys(dependencies or []))
        exps = relevant_experiences or []

        if not actual_chunks and not t_files and not t_symbols and not deps and not exps:
            return AssembledContext(
                context_text="",
                retrieved_chunks=[],
                target_files=[],
                target_symbols=[],
                dependencies=[],
                relevant_experiences=[],
                token_estimate=0,
                is_authoritative=False,
            )

        lines: list[str] = [
            "============================================================",
            "CODEBASE RETRIEVAL CONTEXT (RAG HYBRID INTELLIGENCE)",
            "============================================================",
        ]

        # 1. Target files and AST symbols
        if t_files or t_symbols:
            lines.append("📌 PRIMARY TARGETS (AUTHORITATIVE AST):")
            if t_files:
                lines.append(f"  Target Files: {', '.join(t_files)}")
            if t_symbols:
                lines.append(f"  Target Symbols: {', '.join(t_symbols)}")
            lines.append("")

        # 2. Dependency Context
        if deps:
            lines.append(f"🔗 RELATED DEPENDENCIES: {', '.join(deps[:10])}")
            lines.append("")

        # 3. Relevant Code Chunks (Budget Bounded)
        # Approximate 1 token per 4 characters
        char_budget = budget_tokens * 4
        current_chars = sum(len(l) for l in lines)
        included_chunks: list[RetrievedChunk] = []

        if actual_chunks:
            lines.append("📄 RELEVANT CODE SNIPPETS:")
            for idx, item in enumerate(actual_chunks, start=1):
                chunk = item.chunk
                header = (
                    f"--- Snippet {idx} [Score: {item.score:.2f}] "
                    f"{chunk.relative_path}:{chunk.start_line}-{chunk.end_line} "
                    f"({chunk.chunk_type}: {chunk.symbol_name or 'code'}) ---"
                )
                chunk_block = f"{header}\n{chunk.content}\n"

                if current_chars + len(chunk_block) > char_budget:
                    lines.append("  [Additional lower-ranked snippets omitted due to token budget limit]")
                    break

                lines.append(chunk_block)
                current_chars += len(chunk_block)
                included_chunks.append(item)

        # 4. Relevant Past Experiences (Phase 11)
        if exps:
            lines.append("🧠 PAST EXPERIENCES & LESSONS:")
            for e in exps[:3]:
                exp_data = e.get("experience", e)
                task_str = exp_data.get("task", "")
                succ = exp_data.get("success", True)
                status_str = "✅ SUCCESS" if succ else f"⚠️ FAILURE ({exp_data.get('failure_type', 'unknown')})"
                lesson_str = exp_data.get("lesson") or exp_data.get("result", "")
                exp_line = f"  • [{status_str}] {task_str}: {lesson_str}"
                if current_chars + len(exp_line) <= char_budget:
                    lines.append(exp_line)
                    current_chars += len(exp_line)
            lines.append("")

        # 5. Append Mandatory Invariant Notice
        lines.append(self.DISCLAIMER)

        full_text = "\n".join(lines)
        token_estimate = max(1, len(full_text) // 4)

        return AssembledContext(
            context_text=full_text,
            retrieved_chunks=included_chunks,
            target_files=t_files,
            target_symbols=t_symbols,
            dependencies=deps,
            relevant_experiences=exps,
            token_estimate=token_estimate,
            is_authoritative=False,
        )
