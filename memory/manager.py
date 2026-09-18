import logging
from typing import Any
from memory.models import Memory, MemoryCategory, ExperienceRecord
from memory.store import MemoryStore

logger = logging.getLogger("VIDURA.memory.manager")

STOP_WORDS = {"a", "an", "the", "is", "are", "was", "were", "what", "which", "where", "how", "for", "in", "of", "to", "and", "or", "tell", "me", "about", "do", "you", "my"}


class MemoryManager:
    """High-level Memory Manager providing abstract memory operations for VIDURA Agent."""

    def __init__(self, store: MemoryStore) -> None:
        self.store = store
        from experience.manager import ExperienceManager
        self.experience_manager = ExperienceManager(store=self.store)

    def remember(
        self,
        content: str,
        memory_type: str = MemoryCategory.USER,
        importance: float = 0.5,
        metadata: dict[str, Any] | None = None,
        source_session: str | None = None,
    ) -> Memory:
        """Saves a new memory record, preventing duplicate content insertion.

        Args:
            content: Statement/fact to remember.
            memory_type: Memory category ("user", "project", "conversation", "experience").
            importance: Float priority rating (0.0 - 1.0).
            metadata: Optional dictionary of metadata.
            source_session: Optional session identifier.

        Returns:
            The saved or existing Memory object.
        """
        existing = self.store.find_exact_content(content, memory_type)
        if existing:
            logger.info(f"Duplicate memory detected. Returning existing memory ID '{existing.id}'")
            return existing

        memory = Memory(
            type=memory_type,
            content=content.strip(),
            importance=importance,
            metadata=metadata or {},
            source_session=source_session,
        )
        self.store.add_memory(memory)
        return memory

    def recall(
        self, query: str, memory_type: str | None = None, limit: int = 5
    ) -> list[Memory]:
        """Searches for relevant memories matching the query string."""
        if not query or not query.strip():
            return self.store.list_memories(memory_type=memory_type, limit=limit)
        
        # 1. Try exact phrase match
        results = self.store.search_memories(query=query.strip(), memory_type=memory_type, limit=limit)
        if results:
            return results

        # 2. Fallback: Split query into individual keywords if phrase search found nothing
        words = [w.strip("?,.!\"'") for w in query.lower().split()]
        keywords = [w for w in words if len(w) > 2 and w not in STOP_WORDS]
        
        found_map: dict[str, Memory] = {}
        for kw in keywords:
            search_terms = [kw]
            if kw.endswith("ed") and len(kw) > 4:
                stem = kw[:-2]
                search_terms.append(stem)
                if stem.endswith("rr"):
                    search_terms.append(stem[:-1])
            elif kw.endswith("ing") and len(kw) > 5:
                stem = kw[:-3]
                search_terms.append(stem)
                if stem.endswith("rr"):
                    search_terms.append(stem[:-1])
            elif kw.endswith("s") and len(kw) > 3:
                search_terms.append(kw[:-1])

            for st in search_terms:
                for mem in self.store.search_memories(query=st, memory_type=memory_type, limit=limit):
                    found_map[mem.id] = mem

        return list(found_map.values())[:limit]

    def forget(self, memory_id: str) -> bool:
        """Permanently deletes a memory by ID."""
        return self.store.delete_memory(memory_id)

    def list_memories(self, memory_type: str | None = None, limit: int = 50) -> list[Memory]:
        """Lists memories by creation date descending."""
        return self.store.list_memories(memory_type=memory_type, limit=limit)

    def record_experience(self, record: ExperienceRecord) -> str:
        """Stores an ExperienceRecord with deduplication and historical preservation."""
        return self.experience_manager.record_experience(record)

    def retrieve_experiences(self, query: Any) -> list[Any]:
        """Retrieves and ranks experiences matching query filters."""
        return self.experience_manager.retrieve_experiences(query)

    def derive_lessons(self, component: str | None = None, limit: int = 50) -> list[Any]:
        """Derives empirical lessons grouped by component and pattern."""
        return self.experience_manager.derive_lessons(component=component, limit=limit)

    def get_relevant_lessons(self, component: str | None = None, task_type: str | None = None) -> list[Any]:
        """Returns empirical lessons relevant to component or task type."""
        return self.experience_manager.get_relevant_lessons(component=component, task_type=task_type)

    def validate_lesson(self, lesson: Any) -> dict[str, Any]:
        """Validates lesson against stored empirical evidence."""
        return self.experience_manager.validate_lesson(lesson)

    def get_experience(self, experience_id: str) -> ExperienceRecord | None:
        """Retrieves a single ExperienceRecord by ID."""
        return self.store.get_experience(experience_id)

    def delete_experience(self, experience_id: str) -> bool:
        """Deletes an ExperienceRecord by ID."""
        return self.store.delete_experience(experience_id)

    def list_experiences(
        self,
        task_type: str | None = None,
        success: bool | None = None,
        limit: int = 50,
    ) -> list[ExperienceRecord]:
        """Lists experience records with optional filtering."""
        return self.store.list_experiences(task_type=task_type, success=success, limit=limit)

    def search_experiences(
        self,
        query: str = "",
        task_type: str | None = None,
        success: bool | None = None,
        limit: int = 50,
    ) -> list[ExperienceRecord]:
        """Searches stored experience records."""
        return self.store.search_experiences(query=query, task_type=task_type, success=success, limit=limit)

    def get_context_for_prompt(self, user_request: str) -> str:
        """Recalls relevant user/project memories and returns a concise prompt context string."""
        if not user_request:
            return ""

        words = [w.strip("?,.!\"'") for w in user_request.lower().split()]
        keywords = [w for w in words if len(w) > 2 and w not in STOP_WORDS]

        found_memories: dict[str, Memory] = {}

        # 1. Search for keyword matches across user and project memories
        for kw in keywords[:3]:
            for mem in self.store.search_memories(query=kw, limit=3):
                if mem.type in (MemoryCategory.USER, MemoryCategory.PROJECT):
                    found_memories[mem.id] = mem

        # 2. If keywords yielded no matches, include recent high-importance user/project memories
        if not found_memories:
            for mem in self.store.list_memories(memory_type=MemoryCategory.USER, limit=3):
                found_memories[mem.id] = mem
            for mem in self.store.list_memories(memory_type=MemoryCategory.PROJECT, limit=3):
                found_memories[mem.id] = mem

        if not found_memories:
            return ""

        lines = ["[Stored Persistent Memories]:"]
        for mem in list(found_memories.values())[:5]:
            lines.append(f"- [{mem.type}] {mem.content}")

        return "\n".join(lines)
