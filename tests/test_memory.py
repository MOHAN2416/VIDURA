import json
import tempfile
from pathlib import Path
from typing import Any
import pytest
from models.base import BaseLLMProvider
from memory import (
    Memory,
    MemoryCategory,
    ExperienceRecord,
    MemoryStore,
    MemoryManager,
    extract_explicit_memory_request,
)
from agent import Agent


class FakeMemoryModelProvider(BaseLLMProvider):
    """Fake model provider for testing memory injection deterministically."""

    def __init__(self, response_text: str = "Memory context received.") -> None:
        self.response_text = response_text
        self.last_messages: list[dict[str, str]] = []

    @property
    def model_name(self) -> str:
        return "fake-memory-model"

    @property
    def provider_name(self) -> str:
        return "Fake Memory Provider"

    def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        self.last_messages = list(messages)
        return json.dumps({"action": "respond", "content": self.response_text})


# --- Memory Model Tests (1 - 3) ---

def test_valid_memory_creation() -> None:
    """Test 1: Valid memory can be created with required fields."""
    mem = Memory(type=MemoryCategory.USER, content="User prefers Python.")
    assert mem.type == "user"
    assert mem.content == "User prefers Python."
    assert mem.id is not None
    assert mem.importance == 0.5


def test_memory_required_fields_validation() -> None:
    """Test 2: Memory creation with empty content raises ValueError."""
    with pytest.raises(ValueError) as exc_info:
        Memory(type=MemoryCategory.USER, content="   ")
    assert "content cannot be empty" in str(exc_info.value)


def test_memory_category_validation() -> None:
    """Test 3: Invalid memory category type raises ValueError."""
    with pytest.raises(ValueError) as exc_info:
        Memory(type="invalid_category", content="Some text")
    assert "Invalid memory type" in str(exc_info.value)


# --- SQLite Store Tests (4 - 12) ---

def test_store_insert_and_retrieve(tmp_path: Path) -> None:
    """Tests 4 & 5: Memory can be inserted and retrieved from SQLite store."""
    db_file = tmp_path / "test_store.db"
    store = MemoryStore(db_file)
    
    mem = Memory(type=MemoryCategory.USER, content="Target test memory")
    mem_id = store.add_memory(mem)
    
    retrieved = store.get_memory(mem_id)
    assert retrieved is not None
    assert retrieved.id == mem_id
    assert retrieved.content == "Target test memory"
    store.close()


def test_store_update(tmp_path: Path) -> None:
    """Test 6: Memory content and importance can be updated."""
    store = MemoryStore(tmp_path / "test_store.db")
    mem = Memory(type=MemoryCategory.USER, content="Original text", importance=0.4)
    mem_id = store.add_memory(mem)

    updated = store.update_memory(mem_id, content="Updated text", importance=0.9)
    assert updated is True

    retrieved = store.get_memory(mem_id)
    assert retrieved is not None
    assert retrieved.content == "Updated text"
    assert retrieved.importance == 0.9
    store.close()


def test_store_delete(tmp_path: Path) -> None:
    """Test 7: Memory can be deleted from SQLite store."""
    store = MemoryStore(tmp_path / "test_store.db")
    mem = Memory(type=MemoryCategory.USER, content="To be deleted")
    mem_id = store.add_memory(mem)

    deleted = store.delete_memory(mem_id)
    assert deleted is True
    assert store.get_memory(mem_id) is None
    store.close()


def test_store_list_and_type_filtering(tmp_path: Path) -> None:
    """Tests 8 & 10: Listing memories with and without category filtering."""
    store = MemoryStore(tmp_path / "test_store.db")
    store.add_memory(Memory(type=MemoryCategory.USER, content="User fact 1"))
    store.add_memory(Memory(type=MemoryCategory.PROJECT, content="Project fact 1"))

    all_memories = store.list_memories()
    assert len(all_memories) == 2

    user_memories = store.list_memories(memory_type=MemoryCategory.USER)
    assert len(user_memories) == 1
    assert user_memories[0].type == "user"
    store.close()


def test_store_keyword_search(tmp_path: Path) -> None:
    """Test 9: Keyword search finds matching memories."""
    store = MemoryStore(tmp_path / "test_store.db")
    store.add_memory(Memory(type=MemoryCategory.USER, content="Prefers Python for backend"))
    store.add_memory(Memory(type=MemoryCategory.USER, content="Prefers React for frontend"))

    matches = store.search_memories(query="Python")
    assert len(matches) == 1
    assert "Python" in matches[0].content
    store.close()


def test_store_recent_retrieval(tmp_path: Path) -> None:
    """Test 11: Recent memory retrieval returns most recent items first."""
    store = MemoryStore(tmp_path / "test_store.db")
    store.add_memory(Memory(type=MemoryCategory.USER, content="Old memory"))
    store.add_memory(Memory(type=MemoryCategory.USER, content="New memory"))

    recent = store.list_memories(limit=1)
    assert len(recent) == 1
    assert recent[0].content == "New memory"
    store.close()


def test_store_duplicate_handling(tmp_path: Path) -> None:
    """Test 12: Duplicate content check returns existing memory."""
    store = MemoryStore(tmp_path / "test_store.db")
    mem = Memory(type=MemoryCategory.USER, content="Duplicate test fact")
    store.add_memory(mem)

    existing = store.find_exact_content("Duplicate test fact", MemoryCategory.USER)
    assert existing is not None
    assert existing.id == mem.id
    store.close()


# --- Memory Manager Tests (13 - 18) ---

def test_manager_remember_and_duplicate(tmp_path: Path) -> None:
    """Test 13: MemoryManager.remember() stores memory and prevents duplicates."""
    store = MemoryStore(tmp_path / "test_manager.db")
    manager = MemoryManager(store=store)

    mem1 = manager.remember("User fact A", memory_type=MemoryCategory.USER)
    mem2 = manager.remember("User fact A", memory_type=MemoryCategory.USER)

    assert mem1.id == mem2.id
    assert len(manager.list_memories()) == 1
    store.close()


def test_manager_recall(tmp_path: Path) -> None:
    """Test 14: MemoryManager.recall() retrieves relevant memory."""
    store = MemoryStore(tmp_path / "test_manager.db")
    manager = MemoryManager(store=store)
    manager.remember("User prefers FastAPI", memory_type=MemoryCategory.USER)

    recalled = manager.recall("FastAPI")
    assert len(recalled) == 1
    assert "FastAPI" in recalled[0].content
    store.close()


def test_manager_forget(tmp_path: Path) -> None:
    """Test 15 & 18: MemoryManager.forget() removes memory permanently."""
    store = MemoryStore(tmp_path / "test_manager.db")
    manager = MemoryManager(store=store)
    mem = manager.remember("Temporary fact", memory_type=MemoryCategory.USER)

    forgot = manager.forget(mem.id)
    assert forgot is True
    assert len(manager.recall("Temporary")) == 0
    store.close()


def test_manager_category_separation(tmp_path: Path) -> None:
    """Test 16: User and project memories remain distinct."""
    store = MemoryStore(tmp_path / "test_manager.db")
    manager = MemoryManager(store=store)
    manager.remember("User likes dark mode", memory_type=MemoryCategory.USER)
    manager.remember("VIDURA uses local Ollama", memory_type=MemoryCategory.PROJECT)

    user_mems = manager.list_memories(memory_type=MemoryCategory.USER)
    proj_mems = manager.list_memories(memory_type=MemoryCategory.PROJECT)

    assert len(user_mems) == 1
    assert user_mems[0].content == "User likes dark mode"
    assert len(proj_mems) == 1
    assert proj_mems[0].content == "VIDURA uses local Ollama"
    store.close()


def test_manager_experience_records(tmp_path: Path) -> None:
    """Test 17: ExperienceRecords can be stored and retrieved."""
    store = MemoryStore(tmp_path / "test_manager.db")
    manager = MemoryManager(store=store)

    exp = ExperienceRecord(task="Test task", action_summary="Step 1 completed", result="Success", success=True, lesson="All good")
    exp_id = manager.record_experience(exp)

    records = manager.list_experiences()
    assert len(records) == 1
    assert records[0].id == exp_id
    assert records[0].task == "Test task"
    store.close()


# --- Persistence Across Instances Test (19 - 23) ---

def test_persistence_across_store_instances(tmp_path: Path) -> None:
    """Tests 19-23: Create memory, close store, open new store on same file, verify memory persists."""
    db_file = tmp_path / "persistence_test.db"
    
    # Instance 1: Create & save memory
    store1 = MemoryStore(db_file)
    manager1 = MemoryManager(store=store1)
    saved_mem = manager1.remember("Persistent fact across sessions", memory_type=MemoryCategory.USER)
    saved_id = saved_mem.id
    store1.close()

    # Instance 2: Open new store on same SQLite file
    store2 = MemoryStore(db_file)
    manager2 = MemoryManager(store=store2)
    retrieved = manager2.recall("Persistent")
    
    assert len(retrieved) == 1
    assert retrieved[0].id == saved_id
    assert retrieved[0].content == "Persistent fact across sessions"
    store2.close()


# --- Explicit Memory Extractor Tests ---

def test_extract_explicit_memory_request() -> None:
    """Verifies natural language explicit memory request extraction."""
    stmt1, cat1 = extract_explicit_memory_request("Remember that I prefer Python.")
    assert stmt1 == "I prefer Python."
    assert cat1 == MemoryCategory.USER

    stmt2, cat2 = extract_explicit_memory_request("Remember VIDURA works offline.")
    assert stmt2 == "VIDURA works offline."
    assert cat2 == MemoryCategory.PROJECT

    stmt3, cat3 = extract_explicit_memory_request("What is the capital of France?")
    assert stmt3 is None
    assert cat3 == ""


# --- Agent Integration Tests (24 - 28) ---

def test_agent_memory_context_in_prompt(tmp_path: Path) -> None:
    """Tests 24 & 25: Agent accesses relevant memories and includes them in prompt context."""
    store = MemoryStore(tmp_path / "agent_mem.db")
    manager = MemoryManager(store=store)
    manager.remember("User prefers Python language", memory_type=MemoryCategory.USER)

    provider = FakeMemoryModelProvider()
    agent = Agent(model=provider, memory_manager=manager, max_steps=5)

    response = agent.run("What language do I prefer?")

    assert response == "Memory context received."
    # Check that system context containing stored memory was injected into messages
    messages_str = json.dumps(provider.last_messages)
    assert "Stored Persistent Memories" in messages_str
    assert "User prefers Python language" in messages_str
    store.close()


def test_agent_irrelevant_memories_excluded(tmp_path: Path) -> None:
    """Test 26: Irrelevant query does not inject unrelated memories when search yields no match."""
    store = MemoryStore(tmp_path / "agent_mem2.db")
    manager = MemoryManager(store=store)
    # Empty memory store
    provider = FakeMemoryModelProvider()
    agent = Agent(model=provider, memory_manager=manager, max_steps=5)

    agent.run("Explain recursion")
    messages_str = json.dumps(provider.last_messages)
    assert "Stored Persistent Memories" not in messages_str
    store.close()


def test_agent_normal_conversation_without_memories() -> None:
    """Test 27: Normal conversation works smoothly when memory_manager is None."""
    provider = FakeMemoryModelProvider(response_text="Hello user")
    agent = Agent(model=provider, memory_manager=None, max_steps=5)

    response = agent.run("Hello")
    assert response == "Hello user"


def test_agent_memory_failure_resilience(tmp_path: Path) -> None:
    """Test 28: Memory errors do not crash the Agent."""
    store = MemoryStore(tmp_path / "agent_mem3.db")
    manager = MemoryManager(store=store)
    store.close() # Intentionally close connection to force DB error

    provider = FakeMemoryModelProvider(response_text="Resilient answer")
    agent = Agent(model=provider, memory_manager=manager, max_steps=5)

    # Should handle DB error gracefully and complete turn
    response = agent.run("Hello")
    assert response == "Resilient answer"
