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
                    created_at TEXT NOT NULL,
                    task_type TEXT DEFAULT 'developer',
                    goal TEXT,
                    plan_summary TEXT,
                    affected_files TEXT,
                    affected_components TEXT,
                    relevant_symbols TEXT,
                    provider TEXT,
                    model TEXT,
                    is_cloud INTEGER DEFAULT 0,
                    verification_status TEXT,
                    test_status TEXT,
                    tests_passed INTEGER DEFAULT 0,
                    tests_failed INTEGER DEFAULT 0,
                    failure_reason TEXT,
                    failure_type TEXT,
                    recommendation TEXT,
                    confidence REAL DEFAULT 1.0,
                    fallback_used INTEGER DEFAULT 0,
                    duplicate_count INTEGER DEFAULT 1,
                    evidence_count INTEGER DEFAULT 1,
                    related_experience_ids TEXT,
                    cloud_request_id TEXT
                );
            """)
            # Check for existing table schema and migrate any missing columns
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(experience_records)")
            existing_cols = {row["name"] for row in cursor.fetchall()}
            needed_cols = [
                ("task_type", "TEXT DEFAULT 'developer'"),
                ("goal", "TEXT"),
                ("plan_summary", "TEXT"),
                ("affected_files", "TEXT"),
                ("affected_components", "TEXT"),
                ("relevant_symbols", "TEXT"),
                ("provider", "TEXT"),
                ("model", "TEXT"),
                ("is_cloud", "INTEGER DEFAULT 0"),
                ("verification_status", "TEXT"),
                ("test_status", "TEXT"),
                ("tests_passed", "INTEGER DEFAULT 0"),
                ("tests_failed", "INTEGER DEFAULT 0"),
                ("failure_reason", "TEXT"),
                ("failure_type", "TEXT"),
                ("recommendation", "TEXT"),
                ("confidence", "REAL DEFAULT 1.0"),
                ("fallback_used", "INTEGER DEFAULT 0"),
                ("duplicate_count", "INTEGER DEFAULT 1"),
                ("evidence_count", "INTEGER DEFAULT 1"),
                ("related_experience_ids", "TEXT"),
                ("cloud_request_id", "TEXT"),
            ]
            for col_name, col_def in needed_cols:
                if col_name not in existing_cols:
                    try:
                        conn.execute(f"ALTER TABLE experience_records ADD COLUMN {col_name} {col_def};")
                    except Exception as alt_err:
                        logger.debug(f"Column {col_name} alter warning: {alt_err}")

            conn.execute("CREATE INDEX IF NOT EXISTS idx_exp_task_type ON experience_records(task_type);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_exp_success ON experience_records(success);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_exp_created_at ON experience_records(created_at);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_exp_failure_type ON experience_records(failure_type);")

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
        keys = set(row.keys())

        def _get_list(col: str) -> list[str]:
            if col in keys and row[col]:
                try:
                    val = json.loads(row[col])
                    if isinstance(val, list):
                        return [str(v) for v in val]
                except Exception:
                    pass
            return []

        return ExperienceRecord(
            id=row["id"],
            task=row["task"],
            attempt=row["attempt"] if "attempt" in keys and row["attempt"] is not None else 1,
            action_summary=row["action_summary"] or "" if "action_summary" in keys and row["action_summary"] else "",
            result=row["result"] or "" if "result" in keys and row["result"] else "",
            success=bool(row["success"]) if "success" in keys and row["success"] is not None else True,
            lesson=row["lesson"] or "" if "lesson" in keys and row["lesson"] else "",
            created_at=row["created_at"],
            task_type=row["task_type"] if "task_type" in keys and row["task_type"] else "developer",
            goal=row["goal"] if "goal" in keys and row["goal"] else row["task"],
            plan_summary=row["plan_summary"] or "" if "plan_summary" in keys and row["plan_summary"] else "",
            affected_files=_get_list("affected_files"),
            affected_components=_get_list("affected_components"),
            relevant_symbols=_get_list("relevant_symbols"),
            provider=row["provider"] or "" if "provider" in keys and row["provider"] else "",
            model=row["model"] or "" if "model" in keys and row["model"] else "",
            is_cloud=bool(row["is_cloud"]) if "is_cloud" in keys and row["is_cloud"] is not None else False,
            verification_status=row["verification_status"] or "" if "verification_status" in keys and row["verification_status"] else "",
            test_status=row["test_status"] or "" if "test_status" in keys and row["test_status"] else "",
            tests_passed=row["tests_passed"] if "tests_passed" in keys and row["tests_passed"] is not None else 0,
            tests_failed=row["tests_failed"] if "tests_failed" in keys and row["tests_failed"] is not None else 0,
            failure_reason=row["failure_reason"] if "failure_reason" in keys else None,
            failure_type=row["failure_type"] if "failure_type" in keys else None,
            recommendation=row["recommendation"] or "" if "recommendation" in keys and row["recommendation"] else "",
            confidence=float(row["confidence"]) if "confidence" in keys and row["confidence"] is not None else 1.0,
            fallback_used=bool(row["fallback_used"]) if "fallback_used" in keys and row["fallback_used"] is not None else False,
            duplicate_count=int(row["duplicate_count"]) if "duplicate_count" in keys and row["duplicate_count"] is not None else 1,
            evidence_count=int(row["evidence_count"]) if "evidence_count" in keys and row["evidence_count"] is not None else 1,
            related_experience_ids=_get_list("related_experience_ids"),
            cloud_request_id=row["cloud_request_id"] if "cloud_request_id" in keys else None,
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
        affected_files_json = json.dumps(record.affected_files) if record.affected_files else None
        affected_components_json = json.dumps(record.affected_components) if record.affected_components else None
        relevant_symbols_json = json.dumps(record.relevant_symbols) if record.relevant_symbols else None
        related_ids_json = json.dumps(record.related_experience_ids) if record.related_experience_ids else None

        with conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO experience_records (
                    id, task, attempt, action_summary, result, success, lesson, created_at,
                    task_type, goal, plan_summary, affected_files, affected_components,
                    relevant_symbols, provider, model, is_cloud, verification_status,
                    test_status, tests_passed, tests_failed, failure_reason, failure_type,
                    recommendation, confidence, fallback_used, duplicate_count, evidence_count,
                    related_experience_ids, cloud_request_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    record.task_type,
                    record.goal or record.task,
                    record.plan_summary,
                    affected_files_json,
                    affected_components_json,
                    relevant_symbols_json,
                    record.provider,
                    record.model,
                    1 if record.is_cloud else 0,
                    record.verification_status,
                    record.test_status,
                    record.tests_passed,
                    record.tests_failed,
                    record.failure_reason,
                    record.failure_type,
                    record.recommendation,
                    record.confidence,
                    1 if record.fallback_used else 0,
                    record.duplicate_count,
                    record.evidence_count,
                    related_ids_json,
                    record.cloud_request_id,
                ),
            )
        logger.info(f"Saved ExperienceRecord [{record.task_type}] ID: '{record.id}' (success={record.success})")
        return record.id

    def update_experience(self, record: ExperienceRecord) -> bool:
        """Updates an existing ExperienceRecord in store."""
        return bool(self.add_experience(record))

    def get_experience(self, experience_id: str) -> ExperienceRecord | None:
        """Retrieves a single ExperienceRecord by unique ID."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM experience_records WHERE id = ? LIMIT 1", (experience_id,))
        row = cursor.fetchone()
        return self._row_to_experience(row) if row else None

    def delete_experience(self, experience_id: str) -> bool:
        """Deletes an ExperienceRecord by ID."""
        conn = self._get_connection()
        with conn:
            cursor = conn.execute("DELETE FROM experience_records WHERE id = ?", (experience_id,))
        success = cursor.rowcount > 0
        if success:
            logger.info(f"Deleted ExperienceRecord ID: '{experience_id}'")
        return success

    def list_experiences(
        self,
        task_type: str | None = None,
        success: bool | None = None,
        limit: int = 50,
    ) -> list[ExperienceRecord]:
        """Lists experience records with optional task_type and success filtering."""
        conn = self._get_connection()
        cursor = conn.cursor()
        query = "SELECT * FROM experience_records"
        params: list[Any] = []
        conditions: list[str] = []

        if task_type:
            conditions.append("task_type = ?")
            params.append(task_type)
        if success is not None:
            conditions.append("success = ?")
            params.append(1 if success else 0)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, tuple(params))
        return [self._row_to_experience(row) for row in cursor.fetchall()]

    def search_experiences(
        self,
        query: str = "",
        task_type: str | None = None,
        success: bool | None = None,
        limit: int = 50,
    ) -> list[ExperienceRecord]:
        """Performs search over stored experience records using SQL LIKE."""
        conn = self._get_connection()
        cursor = conn.cursor()
        sql = "SELECT * FROM experience_records"
        params: list[Any] = []
        conditions: list[str] = []

        if task_type:
            conditions.append("task_type = ?")
            params.append(task_type)
        if success is not None:
            conditions.append("success = ?")
            params.append(1 if success else 0)
        if query and query.strip():
            pat = f"%{query.strip()}%"
            conditions.append(
                "(task LIKE ? OR goal LIKE ? OR lesson LIKE ? OR action_summary LIKE ? OR "
                "result LIKE ? OR affected_files LIKE ? OR relevant_symbols LIKE ? OR failure_reason LIKE ?)"
            )
            params.extend([pat] * 8)

        if conditions:
            sql += " WHERE " + " AND ".join(conditions)

        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        cursor.execute(sql, tuple(params))
        return [self._row_to_experience(row) for row in cursor.fetchall()]

    def close(self) -> None:
        """Closes the underlying SQLite connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
