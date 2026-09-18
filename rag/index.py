"""VIDURA RAG Index (Phase 12).

Provides SQLite-backed persistence for code chunks, lexical inverted index,
dense vector embeddings, and incremental hash-based change detection.
"""
from __future__ import annotations

import json
import logging
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rag.models import CodeChunk, EmbeddingVector
from rag.embeddings import cosine_similarity
from config import load_config

logger = logging.getLogger("VIDURA.rag.index")

TOKEN_STOP_WORDS = {
    "a", "an", "the", "in", "on", "of", "to", "for", "with", "at", "by", "from",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "and", "or", "but", "if", "then", "else", "when",
    "this", "that", "these", "those", "self", "cls", "def", "class", "return",
    "import", "from", "as", "pass", "none", "true", "false",
}


def extract_search_tokens(text: str) -> set[str]:
    """Extracts normalized alphanumeric keywords from text excluding stop words."""
    if not text:
        return set()
    words = re.findall(r"[a-zA-Z0-9_\-\.]{2,}", text.lower())
    return {w.strip(".") for w in words if w not in TOKEN_STOP_WORDS and len(w) > 1}


RAG_INDEX_VERSION = "1.0"


class RAGIndex:
    """SQLite-backed index managing chunks, lexical matching, and vector similarity search."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            cfg = load_config()
            self.db_path = cfg.db_path
        else:
            self.db_path = Path(db_path).resolve()

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _init_db(self) -> None:
        """Initializes RAG database tables and indexes with failure safety."""
        try:
            conn = self._get_connection()
            with conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS rag_chunks (
                        chunk_id TEXT PRIMARY KEY,
                        file_path TEXT NOT NULL,
                        relative_path TEXT NOT NULL,
                        start_line INTEGER NOT NULL,
                        end_line INTEGER NOT NULL,
                        chunk_type TEXT NOT NULL,
                        symbol_name TEXT,
                        content TEXT NOT NULL,
                        docstring TEXT,
                        imports TEXT,
                        file_hash TEXT NOT NULL,
                        token_estimate INTEGER DEFAULT 0,
                        metadata TEXT
                    );
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_rag_chunks_rel_path ON rag_chunks(relative_path);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_rag_chunks_symbol ON rag_chunks(symbol_name);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_rag_chunks_file_hash ON rag_chunks(file_hash);")

                conn.execute("""
                    CREATE TABLE IF NOT EXISTS rag_embeddings (
                        chunk_id TEXT PRIMARY KEY,
                        vector_json TEXT NOT NULL,
                        dimension INTEGER NOT NULL,
                        model_name TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        FOREIGN KEY(chunk_id) REFERENCES rag_chunks(chunk_id) ON DELETE CASCADE
                    );
                """)

                conn.execute("""
                    CREATE TABLE IF NOT EXISTS rag_metadata (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL
                    );
                """)
        except Exception as err:
            logger.warning(f"Failed to initialize RAG database at '{self.db_path}' (failure safety engaged): {err}")

    def count_chunks(self) -> int:
        """Returns total number of indexed chunks."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM rag_chunks;")
            return cursor.fetchone()[0]
        except Exception:
            return 0

    def count_embeddings(self) -> int:
        """Returns total number of vector embeddings."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM rag_embeddings;")
            return cursor.fetchone()[0]
        except Exception:
            return 0

    def is_file_up_to_date(self, relative_path: str, current_file_hash: str) -> bool:
        """Checks if all chunks for this file already exist and match current_file_hash."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT DISTINCT file_hash FROM rag_chunks WHERE relative_path = ?;",
                (relative_path,),
            )
            rows = cursor.fetchall()
            if not rows:
                return False
            return len(rows) == 1 and rows[0]["file_hash"] == current_file_hash
        except Exception:
            return False

    def remove_file(self, relative_path: str) -> int:
        """Deletes all chunks and associated embeddings for a given relative path."""
        conn = self._get_connection()
        with conn:
            # Delete embeddings first
            cursor = conn.cursor()
            cursor.execute(
                "SELECT chunk_id FROM rag_chunks WHERE relative_path = ?;",
                (relative_path,),
            )
            chunk_ids = [r["chunk_id"] for r in cursor.fetchall()]
            if chunk_ids:
                placeholders = ",".join("?" for _ in chunk_ids)
                conn.execute(f"DELETE FROM rag_embeddings WHERE chunk_id IN ({placeholders});", chunk_ids)
            conn.execute("DELETE FROM rag_chunks WHERE relative_path = ?;", (relative_path,))
            return len(chunk_ids)

    def delete_file_chunks(self, relative_path: str) -> int:
        """Alias for remove_file."""
        return self.remove_file(relative_path)

    def get_chunks_by_file(self, relative_path: str) -> list[CodeChunk]:
        """Returns all chunks belonging to a relative file path."""
        return self.list_chunks(relative_path=relative_path)

    def get_status(self) -> dict[str, Any]:
        """Returns summary metrics about the stored index."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(DISTINCT relative_path) FROM rag_chunks;")
        indexed_files = cursor.fetchone()[0]
        return {
            "total_chunks": self.count_chunks(),
            "total_embeddings": self.count_embeddings(),
            "indexed_files": indexed_files,
            "db_path": str(self.db_path),
        }

    def upsert_chunk(
        self,
        chunk: CodeChunk,
        vector: Any = None,
        model_name: str = "local-deterministic",
    ) -> None:
        """Upserts a chunk and optionally its embedding vector."""
        if vector is None:
            conn = self._get_connection()
            with conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO rag_chunks (
                        chunk_id, file_path, relative_path, start_line, end_line,
                        chunk_type, symbol_name, content, docstring, imports,
                        file_hash, token_estimate, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        chunk.chunk_id,
                        chunk.file_path,
                        chunk.relative_path,
                        chunk.start_line,
                        chunk.end_line,
                        chunk.chunk_type,
                        chunk.symbol_name,
                        chunk.content,
                        chunk.docstring,
                        json.dumps(chunk.imports),
                        chunk.file_hash,
                        chunk.token_estimate,
                        json.dumps(chunk.metadata),
                    ),
                )
        else:
            if hasattr(vector, "vector"):
                vec_list = vector.vector
                m_name = getattr(vector, "model", model_name) or model_name
            else:
                vec_list = list(vector)
                m_name = model_name
            self.save_chunk_with_embedding(chunk, vec_list, m_name)

    def search_vector(
        self,
        query_vector: list[float],
        top_k: int = 20,
        min_similarity: float = 0.0,
    ) -> list[tuple[CodeChunk, float]]:
        """Alias for search_semantic."""
        return self.search_semantic(query_vector, limit=top_k, min_similarity=min_similarity)

    def save_chunk_with_embedding(
        self,
        chunk: CodeChunk,
        vector: list[float],
        model_name: str,
    ) -> None:
        """Saves a single chunk and its embedding vector atomically."""
        conn = self._get_connection()
        now_str = datetime.now(timezone.utc).isoformat()
        with conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO rag_chunks (
                    chunk_id, file_path, relative_path, start_line, end_line,
                    chunk_type, symbol_name, content, docstring, imports,
                    file_hash, token_estimate, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    chunk.chunk_id,
                    chunk.file_path,
                    chunk.relative_path,
                    chunk.start_line,
                    chunk.end_line,
                    chunk.chunk_type,
                    chunk.symbol_name,
                    chunk.content,
                    chunk.docstring,
                    json.dumps(chunk.imports),
                    chunk.file_hash,
                    chunk.token_estimate,
                    json.dumps(chunk.metadata),
                ),
            )
            conn.execute(
                """
                INSERT OR REPLACE INTO rag_embeddings (
                    chunk_id, vector_json, dimension, model_name, updated_at
                ) VALUES (?, ?, ?, ?, ?);
                """,
                (
                    chunk.chunk_id,
                    json.dumps(vector),
                    len(vector),
                    model_name,
                    now_str,
                ),
            )

    def save_chunks_batch(
        self,
        chunks: list[CodeChunk],
        vectors: list[list[float]],
        model_name: str,
    ) -> None:
        """Batch saves chunks and corresponding embedding vectors atomically."""
        if not chunks:
            return
        conn = self._get_connection()
        now_str = datetime.now(timezone.utc).isoformat()
        with conn:
            for chunk, vec in zip(chunks, vectors):
                conn.execute(
                    """
                    INSERT OR REPLACE INTO rag_chunks (
                        chunk_id, file_path, relative_path, start_line, end_line,
                        chunk_type, symbol_name, content, docstring, imports,
                        file_hash, token_estimate, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        chunk.chunk_id,
                        chunk.file_path,
                        chunk.relative_path,
                        chunk.start_line,
                        chunk.end_line,
                        chunk.chunk_type,
                        chunk.symbol_name,
                        chunk.content,
                        chunk.docstring,
                        json.dumps(chunk.imports),
                        chunk.file_hash,
                        chunk.token_estimate,
                        json.dumps(chunk.metadata),
                    ),
                )
                conn.execute(
                    """
                    INSERT OR REPLACE INTO rag_embeddings (
                        chunk_id, vector_json, dimension, model_name, updated_at
                    ) VALUES (?, ?, ?, ?, ?);
                    """,
                    (
                        chunk.chunk_id,
                        json.dumps(vec),
                        len(vec),
                        model_name,
                        now_str,
                    ),
                )

    def _row_to_chunk(self, row: sqlite3.Row) -> CodeChunk:
        """Converts an SQLite row to a strongly-typed CodeChunk."""
        try:
            imports = json.loads(row["imports"]) if row["imports"] else []
        except Exception:
            imports = []
        try:
            meta = json.loads(row["metadata"]) if row["metadata"] else {}
        except Exception:
            meta = {}

        return CodeChunk(
            chunk_id=row["chunk_id"],
            file_path=row["file_path"],
            relative_path=row["relative_path"],
            start_line=row["start_line"],
            end_line=row["end_line"],
            chunk_type=row["chunk_type"],
            symbol_name=row["symbol_name"],
            content=row["content"],
            docstring=row["docstring"],
            imports=imports,
            file_hash=row["file_hash"],
            token_estimate=row["token_estimate"],
            metadata=meta,
        )

    def get_chunk(self, chunk_id: str) -> CodeChunk | None:
        """Retrieves a single CodeChunk by its ID."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM rag_chunks WHERE chunk_id = ?;", (chunk_id,))
        row = cursor.fetchone()
        return self._row_to_chunk(row) if row else None

    def list_chunks(
        self,
        relative_path: str | None = None,
        symbol_name: str | None = None,
        limit: int = 100,
    ) -> list[CodeChunk]:
        """Lists stored chunks with optional file and symbol filtering."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            sql = "SELECT * FROM rag_chunks"
            params: list[Any] = []
            conditions: list[str] = []

            if relative_path:
                conditions.append("relative_path = ?")
                params.append(relative_path)
            if symbol_name:
                conditions.append("symbol_name = ?")
                params.append(symbol_name)

            if conditions:
                sql += " WHERE " + " AND ".join(conditions)
            sql += " ORDER BY relative_path ASC, start_line ASC LIMIT ?;"
            params.append(limit)

            cursor.execute(sql, tuple(params))
            return [self._row_to_chunk(r) for r in cursor.fetchall()]
        except Exception as err:
            logger.warning(f"Failed to list chunks: {err}")
            return []

    def search_lexical(self, query: str, limit: int = 20) -> list[tuple[CodeChunk, float]]:
        """Performs lexical search using keyword token matching and inverted scoring."""
        try:
            query_tokens = extract_search_tokens(query)
            if not query_tokens:
                return []

            conn = self._get_connection()
            cursor = conn.cursor()

            # Broad SQL search on tokens
            sql_clauses = []
            sql_params = []
            for token in query_tokens:
                sql_clauses.append("(content LIKE ? OR symbol_name LIKE ? OR docstring LIKE ?)")
                pat = f"%{token}%"
                sql_params.extend([pat, pat, pat])

            where_sql = " OR ".join(sql_clauses)
            cursor.execute(f"SELECT * FROM rag_chunks WHERE {where_sql} LIMIT 100;", tuple(sql_params))
            candidates = [self._row_to_chunk(r) for r in cursor.fetchall()]

            # Score candidates by token overlap and exact keyword presence
            scored: list[tuple[CodeChunk, float]] = []
            for chunk in candidates:
                chunk_tokens = extract_search_tokens(f"{chunk.symbol_name or ''} {chunk.content} {chunk.docstring or ''}")
                overlap = query_tokens.intersection(chunk_tokens)
                if overlap:
                    score = len(overlap) / len(query_tokens)
                    # Boost if query appears verbatim in symbol_name or content
                    q_lower = query.lower().strip()
                    if chunk.symbol_name and q_lower in chunk.symbol_name.lower():
                        score += 0.4
                    if q_lower in chunk.content.lower():
                        score += 0.2
                    scored.append((chunk, min(1.0, score)))

            scored.sort(key=lambda item: item[1], reverse=True)
            return scored[:limit]
        except Exception as err:
            logger.warning(f"Lexical search failed: {err}")
            return []

    def search_semantic(
        self,
        query_vector: list[float],
        limit: int = 20,
        min_similarity: float = 0.1,
    ) -> list[tuple[CodeChunk, float]]:
        """Performs vector semantic search by computing cosine similarity against indexed embeddings."""
        if not query_vector:
            return []

        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT c.*, e.vector_json
                FROM rag_chunks c
                JOIN rag_embeddings e ON c.chunk_id = e.chunk_id;
                """
            )

            matches: list[tuple[CodeChunk, float]] = []
            for row in cursor.fetchall():
                try:
                    vec = json.loads(row["vector_json"])
                    sim = cosine_similarity(query_vector, vec)
                    if sim >= min_similarity:
                        chunk = self._row_to_chunk(row)
                        matches.append((chunk, sim))
                except Exception:
                    continue

            matches.sort(key=lambda item: item[1], reverse=True)
            return matches[:limit]
        except Exception as err:
            logger.warning(f"Semantic search failed: {err}")
            return []

    def get_metadata(self, key: str) -> str | None:
        """Retrieves an index metadata value by key."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT value FROM rag_metadata WHERE key = ?;", (key,))
            row = cursor.fetchone()
            return row["value"] if row else None
        except Exception:
            return None

    def set_metadata(self, key: str, value: str) -> None:
        """Sets an index metadata value by key."""
        try:
            conn = self._get_connection()
            with conn:
                conn.execute(
                    "INSERT OR REPLACE INTO rag_metadata (key, value) VALUES (?, ?);",
                    (key, str(value)),
                )
        except Exception as err:
            logger.warning(f"Failed to set metadata {key}: {err}")

    def check_compatibility(self, model_name: str, dimension: int) -> tuple[bool, str]:
        """Checks if stored index matches current version, model name, and vector dimension."""
        try:
            stored_version = self.get_metadata("index_version")
            stored_model = self.get_metadata("embedding_model")
            stored_dim = self.get_metadata("dimension")

            # First run: initialize metadata
            if not stored_version and not stored_model:
                self.set_metadata("index_version", RAG_INDEX_VERSION)
                self.set_metadata("embedding_model", model_name)
                self.set_metadata("dimension", str(dimension))
                return True, "Initialized new RAG index metadata."

            if stored_version != RAG_INDEX_VERSION:
                return False, f"Index version mismatch: stored {stored_version} vs required {RAG_INDEX_VERSION}"

            if stored_model != model_name:
                return False, f"Embedding model mismatch: stored '{stored_model}' vs current '{model_name}'. Re-indexing required."

            if stored_dim and str(stored_dim) != str(dimension):
                return False, f"Dimension mismatch: stored '{stored_dim}' vs current '{dimension}'."

            return True, "Index compatible."
        except Exception as err:
            return False, f"Compatibility check error: {err}"

    def clear(self) -> None:
        """Clears all stored chunks, vector embeddings, and metadata."""
        try:
            conn = self._get_connection()
            with conn:
                conn.execute("DELETE FROM rag_embeddings;")
                conn.execute("DELETE FROM rag_chunks;")
                conn.execute("DELETE FROM rag_metadata;")
        except Exception as err:
            logger.warning(f"Failed to clear RAG index: {err}")

    def rebuild(self) -> None:
        """Clears all stored data and re-initializes metadata."""
        self.clear()

    def close(self) -> None:
        """Closes the underlying database connection."""
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

