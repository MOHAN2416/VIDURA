"""VIDURA RAG & Codebase Intelligence Models (Phase 12).

Defines strongly-typed representations of code chunks, embedding vectors,
hybrid retrieval queries, scored results, and assembled context packets.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CodeChunk:
    """Represents a code-aware structural piece of source code or documentation."""
    file_path: str
    relative_path: str = ""
    start_line: int = 1
    end_line: int = 1
    content: str = ""
    chunk_type: str = "block"  # "module_header", "class", "function", "method", "block"
    chunk_id: str = field(default_factory=lambda: f"chk_{uuid.uuid4().hex[:10]}")
    symbol_name: str | None = None
    name: str | None = None
    symbols: list[str] = field(default_factory=list)
    docstring: str | None = None
    imports: list[str] = field(default_factory=list)
    token_estimate: int = 0
    file_hash: str = ""
    content_hash: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.relative_path:
            self.relative_path = self.file_path
        if self.name and not self.symbol_name:
            self.symbol_name = self.name
        elif self.symbol_name and not self.name:
            self.name = self.symbol_name
        if self.content_hash and not self.file_hash:
            self.file_hash = self.content_hash
        elif self.file_hash and not self.content_hash:
            self.content_hash = self.file_hash
        if not self.symbols and self.symbol_name:
            self.symbols = [self.symbol_name]
        if not self.token_estimate and self.content:
            # Quick whitespace/word approximation: ~1 token per 4 chars
            self.token_estimate = max(1, len(self.content) // 4)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "file_path": self.file_path,
            "relative_path": self.relative_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "chunk_type": self.chunk_type,
            "symbol_name": self.symbol_name,
            "name": self.name,
            "symbols": list(self.symbols),
            "content": self.content,
            "docstring": self.docstring,
            "imports": list(self.imports),
            "token_estimate": self.token_estimate,
            "file_hash": self.file_hash,
            "content_hash": self.content_hash,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CodeChunk":
        return cls(
            chunk_id=str(data.get("chunk_id") or f"chk_{uuid.uuid4().hex[:10]}"),
            file_path=str(data.get("file_path", "")),
            relative_path=str(data.get("relative_path", "")),
            start_line=int(data.get("start_line", 1)),
            end_line=int(data.get("end_line", 1)),
            content=str(data.get("content", "")),
            chunk_type=str(data.get("chunk_type", "block")),
            symbol_name=data.get("symbol_name") or data.get("name"),
            name=data.get("name") or data.get("symbol_name"),
            symbols=list(data.get("symbols", [])),
            docstring=data.get("docstring"),
            imports=list(data.get("imports", [])),
            token_estimate=int(data.get("token_estimate", 0)),
            file_hash=str(data.get("file_hash") or data.get("content_hash") or ""),
            content_hash=str(data.get("content_hash") or data.get("file_hash") or ""),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class EmbeddingVector:
    """Represents a dense semantic vector embedding."""
    vector: list[float]
    dimension: int
    model: str
    provider: str = "local"
    chunk_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "vector": list(self.vector),
            "dimension": self.dimension,
            "model": self.model,
            "provider": self.provider,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EmbeddingVector":
        return cls(
            vector=[float(x) for x in data.get("vector", [])],
            dimension=int(data.get("dimension", 0)),
            model=str(data.get("model", "")),
            provider=str(data.get("provider", "local")),
            chunk_id=data.get("chunk_id"),
        )


@dataclass
class RetrievedChunk:
    """A CodeChunk paired with its hybrid retrieval scores and rank."""
    chunk: CodeChunk
    score: float
    lexical_score: float = 0.0
    semantic_score: float = 0.0
    similarity_score: float = 0.0
    symbol_score: float = 0.0
    dependency_score: float = 0.0
    rrf_score: float = 0.0
    match_reasons: list[str] = field(default_factory=list)
    score_breakdown: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.similarity_score and not self.semantic_score:
            self.semantic_score = self.similarity_score
        elif self.semantic_score and not self.similarity_score:
            self.similarity_score = self.semantic_score

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk": self.chunk.to_dict(),
            "score": round(self.score, 4),
            "lexical_score": round(self.lexical_score, 4),
            "semantic_score": round(self.semantic_score, 4),
            "similarity_score": round(self.similarity_score or self.semantic_score, 4),
            "symbol_score": round(self.symbol_score, 4),
            "dependency_score": round(self.dependency_score, 4),
            "rrf_score": round(self.rrf_score, 4),
            "match_reasons": list(self.match_reasons),
            "score_breakdown": {k: round(v, 4) for k, v in self.score_breakdown.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RetrievedChunk":
        chunk_data = data.get("chunk", {})
        chunk = CodeChunk.from_dict(chunk_data) if isinstance(chunk_data, dict) else chunk_data
        sem_score = float(data.get("semantic_score", data.get("similarity_score", 0.0)))
        return cls(
            chunk=chunk,
            score=float(data.get("score", 0.0)),
            lexical_score=float(data.get("lexical_score", 0.0)),
            semantic_score=sem_score,
            similarity_score=sem_score,
            symbol_score=float(data.get("symbol_score", 0.0)),
            dependency_score=float(data.get("dependency_score", 0.0)),
            rrf_score=float(data.get("rrf_score", 0.0)),
            match_reasons=list(data.get("match_reasons", [])),
            score_breakdown=dict(data.get("score_breakdown", {})),
        )


@dataclass
class RAGQuery:
    """Query parameters for hybrid codebase retrieval."""
    query: str
    target_files: list[str] = field(default_factory=list)
    target_symbols: list[str] = field(default_factory=list)
    file_filter: str | None = None
    symbol_filter: str | None = None
    top_k: int = 5
    min_score: float = 0.15
    include_dependencies: bool = True
    include_experiences: bool = True
    max_context_tokens: int = 4000

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "target_files": list(self.target_files),
            "target_symbols": list(self.target_symbols),
            "file_filter": self.file_filter,
            "symbol_filter": self.symbol_filter,
            "top_k": self.top_k,
            "min_score": self.min_score,
            "include_dependencies": self.include_dependencies,
            "include_experiences": self.include_experiences,
            "max_context_tokens": self.max_context_tokens,
        }


@dataclass
class AssembledContext:
    """Represents a budget-controlled, structured context packet for model prompting."""
    context_text: str
    retrieved_chunks: list[RetrievedChunk] = field(default_factory=list)
    target_files: list[str] = field(default_factory=list)
    target_symbols: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    relevant_experiences: list[dict[str, Any]] = field(default_factory=list)
    token_estimate: int = 0
    total_tokens: int = 0
    total_chunks: int = 0
    is_authoritative: bool = False  # CRITICAL: RAG is strictly advisory, never source of truth

    def __post_init__(self) -> None:
        if self.total_tokens and not self.token_estimate:
            self.token_estimate = self.total_tokens
        elif self.token_estimate and not self.total_tokens:
            self.total_tokens = self.token_estimate
        if not self.total_chunks:
            self.total_chunks = len(self.retrieved_chunks)

    @property
    def prompt_text(self) -> str:
        return self.context_text

    def to_dict(self) -> dict[str, Any]:
        return {
            "context_text": self.context_text,
            "prompt_text": self.prompt_text,
            "retrieved_chunks": [c.to_dict() for c in self.retrieved_chunks],
            "target_files": list(self.target_files),
            "target_symbols": list(self.target_symbols),
            "dependencies": list(self.dependencies),
            "relevant_experiences": list(self.relevant_experiences),
            "token_estimate": self.token_estimate,
            "total_tokens": self.total_tokens,
            "total_chunks": self.total_chunks,
            "is_authoritative": self.is_authoritative,
        }
