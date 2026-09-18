"""VIDURA High-Level RAG Manager (Phase 12).

Unifies code-aware chunking, dense vector embeddings, persistent indexing,
hybrid multi-signal retrieval, and structured context assembly.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

from rag.models import CodeChunk, RetrievedChunk, RAGQuery, AssembledContext
from rag.chunker import CodeAwareChunker
from rag.embeddings import EmbeddingManager
from rag.index import RAGIndex
from rag.retriever import HybridRetriever
from rag.context import ContextAssembler
from models.security import is_protected_file_target, redact_secrets

logger = logging.getLogger("VIDURA.rag.manager")


class RAGManager:
    """High-level API for RAG & Advanced Codebase Intelligence."""

    def __init__(
        self,
        index: RAGIndex | None = None,
        embedding_manager: EmbeddingManager | None = None,
        chunker: CodeAwareChunker | None = None,
        retriever: HybridRetriever | None = None,
        context_assembler: ContextAssembler | None = None,
        codebase_manager: Any = None,
        experience_manager: Any = None,
        workspace_root: str | Path | None = None,
        db_path: str | Path | None = None,
        enabled: bool = True,
    ) -> None:
        self.enabled = enabled
        self.workspace_root = Path(workspace_root).resolve() if workspace_root else (
            getattr(codebase_manager, "workspace_root", None) or Path.cwd().resolve()
        )
        self.index = index or RAGIndex(db_path=db_path)
        self.embedding_manager = embedding_manager or EmbeddingManager()
        self.chunker = chunker or CodeAwareChunker()
        self.context_assembler = context_assembler or ContextAssembler()
        self.codebase_manager = codebase_manager
        if self.codebase_manager is None and self.workspace_root:
            from codebase.manager import CodebaseManager
            self.codebase_manager = CodebaseManager(workspace_root=self.workspace_root)
        self.experience_manager = experience_manager

        self.retriever = retriever or HybridRetriever(
            index=self.index,
            embedding_manager=self.embedding_manager,
            codebase_manager=self.codebase_manager,
            experience_manager=self.experience_manager,
        )

    def index_file(
        self,
        file_path: str | Path,
        relative_path: str | None = None,
        force: bool = False,
    ) -> list[CodeChunk]:
        """Chunks and indexes a single file if it has changed, or if force is True."""
        if not self.enabled:
            return []
        p = Path(file_path).resolve()
        rel_p = relative_path or p.name

        # Security check: Skip protected files (.git, .env, private keys)
        if is_protected_file_target(str(p)) or is_protected_file_target(rel_p):
            logger.info(f"Skipping indexing for protected sensitive file: '{rel_p}'")
            return []

        try:
            chunks = self.chunker.chunk_file(file_path=p, relative_path=rel_p)
            if not chunks:
                return []

            file_hash = chunks[0].file_hash

            # Incremental check: if file unchanged, skip re-indexing
            if not force and self.index.is_file_up_to_date(rel_p, file_hash):
                logger.debug(f"File '{rel_p}' is already up-to-date in RAG index.")
                return self.index.list_chunks(relative_path=rel_p)

            # Remove previous chunks for this file if updating
            self.index.remove_file(rel_p)

            # Generate embeddings
            texts = [f"{c.symbol_name or ''}\n{c.content}\n{c.docstring or ''}" for c in chunks]
            vectors = [self.embedding_manager.embed_text(t) for t in texts]

            # Save chunks and embeddings
            self.index.save_chunks_batch(
                chunks=chunks,
                vectors=vectors,
                model_name=self.embedding_manager.active_provider.model_name,
            )
            logger.info(f"Indexed file '{rel_p}': {len(chunks)} chunks.")
            return chunks

        except Exception as err:
            logger.warning(f"Failed to index file '{rel_p}' (failure safety engaged): {err}")
            return []

    def index_codebase(
        self,
        codebase_manager: Any = None,
        force: bool = False,
     ) -> dict[str, Any]:
        """Scans workspace files and incrementally updates the RAG index."""
        if not self.enabled:
            return {
                "indexed_files": 0,
                "skipped_files": 0,
                "total_chunks": 0,
                "errors": [],
            }
        cb_mgr = codebase_manager or self.codebase_manager
        if not cb_mgr:
            logger.warning("No CodebaseManager available to index codebase.")
            return {"indexed_files": 0, "skipped_files": 0, "total_chunks": self.index.count_chunks()}

        if hasattr(cb_mgr, "scan"):
            cb_mgr.scan()
        elif hasattr(cb_mgr, "ensure_scanned"):
            cb_mgr.ensure_scanned()
        files = cb_mgr.list_files()

        indexed_count = 0
        skipped_count = 0

        for cf in files:
            # Skip sensitive or protected files
            if is_protected_file_target(cf.path) or is_protected_file_target(cf.relative_path):
                skipped_count += 1
                continue

            # Only index source files (Python and text/markdown/config)
            ext = Path(cf.relative_path).suffix.lower()
            if ext not in (".py", ".md", ".txt", ".toml", ".yaml", ".yml", ".json"):
                skipped_count += 1
                continue

            try:
                p = Path(cf.path)
                content = p.read_text(encoding="utf-8", errors="replace")
                sanitized = redact_secrets(content)
                file_hash = hashlib.sha256(sanitized.encode("utf-8")).hexdigest()
                if not force and self.index.is_file_up_to_date(cf.relative_path, file_hash):
                    skipped_count += 1
                    continue
            except Exception:
                pass

            chunks = self.index_file(file_path=cf.path, relative_path=cf.relative_path, force=force)
            if chunks:
                indexed_count += 1
            else:
                skipped_count += 1

        total = self.index.count_chunks()
        logger.info(f"Codebase indexing completed: {indexed_count} files indexed, {skipped_count} skipped, {total} total chunks.")
        return {
            "indexed_files": indexed_count,
            "skipped_files": skipped_count,
            "total_chunks": total,
            "errors": [],
        }

    def retrieve(self, query: RAGQuery | str) -> list[RetrievedChunk]:
        """Executes hybrid retrieval matching the given query with failure safety."""
        if not self.enabled:
            return []
        try:
            if isinstance(query, str):
                query_obj = RAGQuery(query=query)
            else:
                query_obj = query

            return self.retriever.retrieve(query_obj)
        except Exception as err:
            logger.warning(f"RAG retrieval failed (failure safety engaged): {err}")
            return []

    def assemble_context(self, query: RAGQuery | str) -> AssembledContext:
        """Runs hybrid retrieval and formats structured prompt context."""
        if not self.enabled:
            return AssembledContext(
                context_text="",
                retrieved_chunks=[],
                is_authoritative=False,
            )
        try:
            if isinstance(query, str):
                query_obj = RAGQuery(query=query)
            else:
                query_obj = query

            chunks = self.retrieve(query_obj)

            # Retrieve past experiences if enabled
            exps: list[dict[str, Any]] = []
            if query_obj.include_experiences:
                exps = self.retriever.retrieve_relevant_experiences(
                    query_text=query_obj.query,
                    target_files=query_obj.target_files,
                    target_symbols=query_obj.target_symbols,
                )

            # Collect dependencies if enabled
            deps: list[str] = []
            if self.codebase_manager and query_obj.include_dependencies:
                for tf in query_obj.target_files:
                    mod = tf.replace(".py", "").replace("/", ".")
                    deps.extend(self.codebase_manager.find_dependencies(mod))

            return self.context_assembler.assemble(
                retrieved_chunks=chunks,
                target_files=query_obj.target_files,
                target_symbols=query_obj.target_symbols,
                dependencies=deps,
                relevant_experiences=exps,
                max_context_tokens=query_obj.max_context_tokens,
            )
        except Exception as err:
            logger.warning(f"RAG context assembly failed (failure safety engaged): {err}")
            return AssembledContext(
                context_text="",
                retrieved_chunks=[],
                is_authoritative=False,
            )

    def get_status(self) -> dict[str, Any]:
        """Returns diagnostic status of the RAG index and embedding models."""
        try:
            total_chunks = self.index.count_chunks()
        except Exception:
            total_chunks = 0
        try:
            total_emb = self.index.count_embeddings()
        except Exception:
            total_emb = 0
        idx_status = {}
        if hasattr(self.index, "get_status"):
            try:
                idx_status = self.index.get_status()
            except Exception:
                pass

        return {
            "enabled": self.enabled,
            "chunk_count": total_chunks,
            "total_chunks": total_chunks,
            "embedding_count": total_emb,
            "indexed_files": idx_status.get("indexed_files", 0),
            "embedding_model": self.embedding_manager.active_provider.model_name,
            "embedding_provider": self.embedding_manager.active_provider.provider_name,
            "embedding_dimension": self.embedding_manager.active_provider.dimension,
            "dimensions": self.embedding_manager.active_provider.dimension,
            "storage_type": "sqlite",
            "database_path": str(getattr(self.index, "db_path", "")),
            "db_path": str(getattr(self.index, "db_path", "")),
        }

    def clear(self) -> None:
        """Clears all chunks and embeddings."""
        self.index.clear()

    def close(self) -> None:
        """Closes underlying index connection."""
        if hasattr(self.index, "close"):
            self.index.close()
