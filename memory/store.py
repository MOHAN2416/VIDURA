import json
import sqlite3
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from memory.models import Memory, ExperienceRecord

logger = logging.getLogger("VIDURA.memory.store")


class MemoryStore:
    """SQLite Storage Engine for VIDURA Memory Management."""

    def __init__(self, db_path: Path | str) -> None:
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
        """Initializes database schema and indices."""
        conn = self._get_connection()
        with conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    metadata TEXT,
                    importance REAL DEFAULT 0.5,
                    source_session TEXT
                );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(type);")
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experience_records (
                    id TEXT PRIMARY KEY,
                    task TEXT NOT NULL,
                    attempt INTEGER DEFAULT 1,
                    action_summary TEXT,
                    result TEXT,
                    success INTEGER DEFAULT 1,
                    lesson TEXT,
                    created_at TEXT NOT NULL
                );
            """)

    def _row_to_memory(self, row: sqlite3.Row) -> Memory:
        metadata = {}
        if row["metadata"]:
            try:
                metadata = json.loads(row["metadata"])
            except Exception:
                pass
        return Memory(
            id=row["id"],
            type=row["type"],
            content=row["content"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            metadata=metadata,
            importance=row["importance"],
            source_session=row["source_session"],
        )

    def _row_to_experience(self, row: sqlite3.Row) -> ExperienceRecord:
        return ExperienceRecord(
            id=row["id"],
            task=row["task"],
            attempt=row["attempt"],
            action_summary=row["action_summary"] or "",
            result=row["result"] or "",
            success=bool(row["success"]),
            lesson=row["lesson"] or "",
            created_at=row["created_at"],
        )

    def find_exact_content(self, content: str, memory_type: str) -> Memory | None:
        """Searches for an exact content duplicate within the given memory category."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM memories WHERE type = ? AND LOWER(TRIM(content)) = LOWER(TRIM(?)) LIMIT 1",
            (memory_type, content),
        )
        row = cursor.fetchone()
        return self._row_to_memory(row) if row else None

    def add_memory(self, memory: Memory) -> str:
        """Inserts a new memory record into the store."""
        conn = self._get_connection()
        metadata_str = json.dumps(memory.metadata) if memory.metadata else None
        with conn:
            conn.execute(
                """
                INSERT INTO memories (id, type, content, created_at, updated_at, metadata, importance, source_session)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    memory.id,
                    memory.type,
                    memory.content,
                    memory.created_at,
                    memory.updated_at,
                    metadata_str,
                    memory.importance,
                    memory.source_session,
                ),
            )
        logger.info(f"Saved memory [{memory.type}] ID: '{memory.id}'")
        return memory.id

    def get_memory(self, memory_id: str) -> Memory | None:
        """Retrieves a memory record by unique ID."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM memories WHERE id = ?", (memory_id,))
        row = cursor.fetchone()
        return self._row_to_memory(row) if row else None

    def update_memory(
        self,
        memory_id: str,
        content: str | None = None,
        importance: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Updates fields of an existing memory record."""
        existing = self.get_memory(memory_id)
        if not existing:
            return False

        new_content = content if content is not None else existing.content
        new_importance = importance if importance is not None else existing.importance
        new_metadata = metadata if metadata is not None else existing.metadata
        updated_at = datetime.now(timezone.utc).isoformat()
        metadata_str = json.dumps(new_metadata) if new_metadata else None

        conn = self._get_connection()
        with conn:
            cursor = conn.execute(
                """
                UPDATE memories
                SET content = ?, importance = ?, metadata = ?, updated_at = ?
                WHERE id = ?
                """,
                (new_content, new_importance, metadata_str, updated_at, memory_id),
            )
        return cursor.rowcount > 0

    def delete_memory(self, memory_id: str) -> bool:
        """Deletes a memory record permanently by ID."""
        conn = self._get_connection()
        with conn:
            cursor = conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        success = cursor.rowcount > 0
        if success:
            logger.info(f"Deleted memory ID: '{memory_id}'")
        return success

    def list_memories(self, memory_type: str | None = None, limit: int = 50) -> list[Memory]:
        """Lists memories ordered by creation time descending."""
        conn = self._get_connection()
        cursor = conn.cursor()
        if memory_type:
            cursor.execute(
                "SELECT * FROM memories WHERE type = ? ORDER BY created_at DESC LIMIT ?",
                (memory_type, limit),
            )
        else:
            cursor.execute("SELECT * FROM memories ORDER BY created_at DESC LIMIT ?", (limit,))
        
        return [self._row_to_memory(row) for row in cursor.fetchall()]

    def search_memories(
        self, query: str, memory_type: str | None = None, limit: int = 10
    ) -> list[Memory]:
        """Performs keyword search over stored memories using SQL LIKE query."""
        conn = self._get_connection()
        cursor = conn.cursor()
        search_pattern = f"%{query}%"

        if memory_type:
            cursor.execute(
                """
                SELECT * FROM memories
                WHERE type = ? AND content LIKE ?
                ORDER BY importance DESC, created_at DESC
                LIMIT ?
                """,
                (memory_type, search_pattern, limit),
            )
        else:
            cursor.execute(
                """
                SELECT * FROM memories
                WHERE content LIKE ?
                ORDER BY importance DESC, created_at DESC
                LIMIT ?
                """,
                (search_pattern, limit),
            )

        return [self._row_to_memory(row) for row in cursor.fetchall()]

    def add_experience(self, record: ExperienceRecord) -> str:
        """Inserts an ExperienceRecord into store."""
        conn = self._get_connection()
        with conn:
            conn.execute(
                """
                INSERT INTO experience_records (id, task, attempt, action_summary, result, success, lesson, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.task,
                    record.attempt,
                    record.action_summary,
                    record.result,
                    1 if record.success else 0,
                    record.lesson,
                    record.created_at,
                ),
            )
        return record.id

    def list_experiences(self, limit: int = 50) -> list[ExperienceRecord]:
        """Lists experience records ordered by creation time descending."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM experience_records ORDER BY created_at DESC LIMIT ?", (limit,))
        return [self._row_to_experience(row) for row in cursor.fetchall()]

    def close(self) -> None:
        """Closes the underlying SQLite connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
