"""VIDURA RAG Tools (Phase 12).

Provides read-only tools for advanced semantic/hybrid codebase retrieval
and budget-bounded context assembly.
"""
from __future__ import annotations

import logging
from typing import Any
from tools.base import BaseTool, success_result, error_result
from rag.models import RAGQuery

logger = logging.getLogger("VIDURA.tools.rag")


class SearchCodebaseSemanticTool(BaseTool):
    """Tool to perform hybrid semantic and lexical retrieval across codebase chunks."""

    def __init__(self, rag_manager: Any) -> None:
        self.rag_manager = rag_manager

    @property
    def name(self) -> str:
        return "search_codebase_semantic"

    @property
    def description(self) -> str:
        return (
            "Performs advanced hybrid retrieval (semantic vector similarity, lexical keyword matching, "
            "and AST symbol matching) across indexed codebase chunks. Read-only."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language or code query describing what functionality or concept to find.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Maximum number of top results to return (bounded default: 5).",
                    "default": 5,
                },
                "target_file": {
                    "type": "string",
                    "description": "Optional specific file to restrict search to.",
                },
                "target_symbol": {
                    "type": "string",
                    "description": "Optional specific symbol to search for or boost.",
                },
            },
            "required": ["query"],
        }

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        query_text = kwargs.get("query", "").strip()
        if not query_text:
            return error_result(self.name, "Query parameter cannot be empty.")

        top_k = min(20, max(1, int(kwargs.get("top_k", 5))))
        target_file = kwargs.get("target_file")
        target_symbol = kwargs.get("target_symbol")

        try:
            rag_query = RAGQuery(
                query=query_text,
                top_k=top_k,
                target_files=[target_file] if target_file else [],
                target_symbols=[target_symbol] if target_symbol else [],
            )
            retrieved = self.rag_manager.retrieve(rag_query)
            results = []
            for r in retrieved:
                c = r.chunk
                results.append({
                    "file": c.relative_path,
                    "symbol": c.symbol_name,
                    "type": c.chunk_type,
                    "start_line": c.start_line,
                    "end_line": c.end_line,
                    "score": round(r.score, 4),
                    "matched_by": r.match_reasons if r.match_reasons else [
                        "semantic" if r.semantic_score > 0.3 else "lexical"
                    ],
                    "content_preview": c.content[:300].strip(),
                })

            return success_result(
                self.name,
                data={
                    "count": len(results),
                    "query": query_text,
                    "results": results,
                },
            )
        except Exception as err:
            logger.warning(f"Error in search_codebase_semantic tool: {err}")
            return error_result(self.name, f"Search failed: {err}")


class GetRelevantCodeContextTool(BaseTool):
    """Tool to assemble structured, budget-bounded context combining code chunks, dependencies, and lessons."""

    def __init__(self, rag_manager: Any) -> None:
        self.rag_manager = rag_manager

    @property
    def name(self) -> str:
        return "get_relevant_code_context"

    @property
    def description(self) -> str:
        return (
            "Assembles structured, token-bounded codebase context, related dependencies, "
            "and relevant past experience lessons for a given task. Read-only."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task_description": {
                    "type": "string",
                    "description": "Description of the development or debugging task.",
                },
                "target_files": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of target files related to the task.",
                },
                "target_symbols": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of target symbols related to the task.",
                },
            },
            "required": ["task_description"],
        }

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        task_desc = kwargs.get("task_description", "").strip()
        if not task_desc:
            return error_result(self.name, "task_description parameter cannot be empty.")

        target_files = kwargs.get("target_files") or []
        target_symbols = kwargs.get("target_symbols") or []

        try:
            rag_query = RAGQuery(
                query=task_desc,
                target_files=list(target_files),
                target_symbols=list(target_symbols),
                top_k=5,
            )
            ctx = self.rag_manager.assemble_context(rag_query)
            return success_result(self.name, data=ctx.to_dict())
        except Exception as err:
            logger.warning(f"Error in get_relevant_code_context tool: {err}")
            return error_result(self.name, f"Context assembly failed: {err}")
