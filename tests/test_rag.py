import json
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from rag.models import CodeChunk, EmbeddingVector, RetrievedChunk, RAGQuery, AssembledContext
from rag.chunker import CodeAwareChunker
from rag.embeddings import (
    DeterministicLocalEmbeddingProvider,
    OllamaEmbeddingProvider,
    EmbeddingManager,
    cosine_similarity,
)
from rag.index import RAGIndex, RAG_INDEX_VERSION
from rag.reranker import HybridReranker
from rag.retriever import HybridRetriever
from rag.context import ContextAssembler
from rag.manager import RAGManager
from codebase.manager import CodebaseManager
from planning.planner import DeveloperPlanner
from planning.models import DeveloperPlan
from task_understanding.models import DeveloperTask, TaskType
from developer.generator import CodeChangeGenerator
from tools.rag import SearchCodebaseSemanticTool, GetRelevantCodeContextTool


# =====================================================================
# 1. RAG Models & Data Structures Tests
# =====================================================================

def test_code_chunk_creation_and_serialization():
    chunk = CodeChunk(
        chunk_id="chk_123",
        file_path="foo/bar.py",
        relative_path="foo/bar.py",
        chunk_type="function",
        name="calculate_total",
        start_line=10,
        end_line=25,
        content="def calculate_total(items):\n    return sum(items)\n",
        symbols=["calculate_total", "items"],
        docstring="Calculates total sum.",
        content_hash="abc123hash",
    )

    data = chunk.to_dict()
    assert data["chunk_id"] == "chk_123"
    assert data["file_path"] == "foo/bar.py"
    assert data["chunk_type"] == "function"
    assert data["symbol_name"] == "calculate_total"
    assert data["start_line"] == 10
    assert data["end_line"] == 25
    assert data["symbols"] == ["calculate_total", "items"]
    assert data["docstring"] == "Calculates total sum."

    restored = CodeChunk.from_dict(data)
    assert restored.chunk_id == chunk.chunk_id
    assert restored.content == chunk.content
    assert restored.symbols == chunk.symbols


def test_retrieved_chunk_serialization():
    chunk = CodeChunk(
        chunk_id="chk_1",
        file_path="agent/loop.py",
        relative_path="agent/loop.py",
        chunk_type="class",
        name="AgentLoop",
        start_line=1,
        end_line=50,
        content="class AgentLoop:\n    pass\n",
    )
    rc = RetrievedChunk(
        chunk=chunk,
        score=0.88,
        similarity_score=0.91,
        lexical_score=0.75,
        match_reasons=["vector_similarity", "symbol_match"],
    )

    d = rc.to_dict()
    assert d["score"] == 0.88
    assert d["similarity_score"] == 0.91
    assert "symbol_match" in d["match_reasons"]
    assert d["chunk"]["name"] == "AgentLoop"

    restored = RetrievedChunk.from_dict(d)
    assert restored.score == rc.score
    assert restored.chunk.name == "AgentLoop"


def test_rag_query_defaults():
    q = RAGQuery(query="authentication token handler")
    assert q.top_k == 5
    assert q.min_score == 0.15
    assert q.include_dependencies is True
    assert q.include_experiences is True
    assert q.max_context_tokens == 4000


# =====================================================================
# 2. Code-Aware Chunker Tests
# =====================================================================

def test_chunker_python_ast(tmp_path):
    sample_code = (
        '"""Module docstring for auth service."""\n'
        'import os\n'
        'from pathlib import Path\n\n\n'
        'class TokenManager:\n'
        '    """Manages session tokens."""\n\n'
        '    def __init__(self, secret_key: str):\n'
        '        self.secret_key = secret_key\n\n'
        '    def generate_token(self, user_id: str) -> str:\n'
        '        """Generates a token."""\n'
        '        return f"tok_{user_id}"\n\n\n'
        'def verify_signature(token: str) -> bool:\n'
        '    """Verifies token signature."""\n'
        '    return token.startswith("tok_")\n'
    )
    py_file = tmp_path / "auth.py"
    py_file.write_text(sample_code, encoding="utf-8")

    chunker = CodeAwareChunker(workspace_root=tmp_path)
    chunks = chunker.chunk_file(py_file, relative_path="auth.py", source_code=sample_code)

    assert len(chunks) >= 3
    types = [c.chunk_type for c in chunks]
    assert "module" in types or "module_header" in types
    assert "class" in types
    assert "function" in types

    # Check function chunk
    fn_chunk = next(c for c in chunks if c.name == "verify_signature")
    assert fn_chunk.chunk_type == "function"
    assert "def verify_signature" in fn_chunk.content
    assert fn_chunk.docstring == "Verifies token signature."


def test_chunker_redacts_secrets(tmp_path):
    code_with_secret = (
        '# Configuration file\n'
        'API_KEY = "sk-ant-api03-secretkey1234567890abcdef"\n'
        'def get_key():\n'
        '    return API_KEY\n'
    )
    py_file = tmp_path / "secrets.py"
    py_file.write_text(code_with_secret, encoding="utf-8")

    chunker = CodeAwareChunker(workspace_root=tmp_path)
    chunks = chunker.chunk_file(py_file, relative_path="secrets.py", source_code=code_with_secret)

    assert len(chunks) >= 1
    for chk in chunks:
        assert "sk-ant-api03-secretkey1234567890abcdef" not in chk.content
    all_content = "\n".join(c.content for c in chunks)
    assert "[REDACTED" in all_content or "[REDACTED]" in all_content


def test_chunker_sliding_window_fallback(tmp_path):
    json_content = json.dumps({"config": [f"item_{i}" for i in range(100)]}, indent=2)
    chunker = CodeAwareChunker(workspace_root=tmp_path, max_chunk_lines=30, chunk_overlap_lines=5)
    chunks = chunker.chunk_file("data.json", source_code=json_content)

    assert len(chunks) > 1
    assert all(c.chunk_type == "block" for c in chunks)
    assert chunks[0].start_line == 1


def test_chunker_empty_file(tmp_path):
    chunker = CodeAwareChunker(workspace_root=tmp_path)
    assert chunker.chunk_file("empty.py", source_code="") == []
    assert chunker.chunk_file("spaces.py", source_code="   \n\n  \t  ") == []


# =====================================================================
# 3. Embedding Providers & Similarity Tests
# =====================================================================

def test_deterministic_embedding_provider():
    provider = DeterministicLocalEmbeddingProvider(dimension=128)
    text1 = "def authenticate_user(username, password):"
    text2 = "def authenticate_user(username, password):"
    text3 = "import numpy as np\nmatrix = np.zeros((10, 10))"

    vec1 = provider.embed(text1)
    vec2 = provider.embed(text2)
    vec3 = provider.embed(text3)

    assert vec1 is not None and vec2 is not None and vec3 is not None
    assert len(vec1.vector) == 128
    assert vec1.vector == vec2.vector  # deterministic

    # Test L2 norm is approx 1.0
    norm = sum(x * x for x in vec1.vector) ** 0.5
    assert pytest.approx(norm, abs=1e-4) == 1.0

    # Test cosine similarity
    sim_identical = cosine_similarity(vec1.vector, vec2.vector)
    assert pytest.approx(sim_identical, abs=1e-4) == 1.0

    sim_diff = cosine_similarity(vec1.vector, vec3.vector)
    assert sim_diff < sim_identical


def test_cosine_similarity_edge_cases():
    assert cosine_similarity([], []) == 0.0
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0
    assert cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0]) == 0.0


def test_ollama_embedding_provider_fallback():
    provider = OllamaEmbeddingProvider(model_name="nomic-embed-text")
    # Simulate network error
    with patch("urllib.request.urlopen", side_effect=Exception("Connection refused")):
        res = provider.embed("some text to embed")
        assert res is None


def test_embedding_manager_fallback():
    manager = EmbeddingManager(model_name="nonexistent-model", dimension=64)
    # Even if Ollama fails, EmbeddingManager transparently falls back to DeterministicLocalEmbeddingProvider
    vec = manager.embed("class TestRunner:\n    pass")
    assert vec is not None
    assert len(vec.vector) == 64
    assert "deterministic" in vec.model


# =====================================================================
# 4. RAG Index & SQLite Storage Tests
# =====================================================================

def test_rag_index_crud(tmp_path):
    db_path = tmp_path / "test_rag.db"
    index = RAGIndex(db_path=db_path)

    chunk1 = CodeChunk(
        chunk_id="chk_mod_1",
        file_path="tools/search.py",
        relative_path="tools/search.py",
        chunk_type="function",
        name="search_code",
        start_line=1,
        end_line=20,
        content="def search_code(query: str):\n    return []\n",
        symbols=["search_code", "query"],
        content_hash="hash_111",
    )
    vec1 = EmbeddingVector(chunk_id="chk_mod_1", vector=[0.1] * 32, dimension=32, model="test")

    index.upsert_chunk(chunk1, vec1)

    # Retrieval
    retrieved = index.get_chunk("chk_mod_1")
    assert retrieved is not None
    assert retrieved.name == "search_code"
    assert retrieved.file_path == "tools/search.py"

    # Up to date check
    assert index.is_file_up_to_date("tools/search.py", "hash_111") is True
    assert index.is_file_up_to_date("tools/search.py", "hash_changed") is False
    assert index.is_file_up_to_date("nonexistent.py", "hash_111") is False

    # Status
    status = index.get_status()
    assert status["total_chunks"] == 1
    assert status["indexed_files"] == 1

    # Delete
    index.delete_file_chunks("tools/search.py")
    assert index.get_chunk("chk_mod_1") is None
    assert index.get_status()["total_chunks"] == 0

    index.close()


def test_rag_index_lexical_search(tmp_path):
    db_path = tmp_path / "test_lexical.db"
    index = RAGIndex(db_path=db_path)

    c1 = CodeChunk(
        chunk_id="c1",
        file_path="security/auth.py",
        relative_path="security/auth.py",
        chunk_type="function",
        name="authenticate_token",
        start_line=1,
        end_line=15,
        content="def authenticate_token(token: str) -> bool:\n    return verify_jwt(token)\n",
        symbols=["authenticate_token", "verify_jwt"],
        content_hash="h1",
    )
    c2 = CodeChunk(
        chunk_id="c2",
        file_path="models/router.py",
        relative_path="models/router.py",
        chunk_type="class",
        name="ModelRouter",
        start_line=1,
        end_line=30,
        content="class ModelRouter:\n    def route(self, task): pass\n",
        symbols=["ModelRouter", "route"],
        content_hash="h2",
    )
    index.upsert_chunk(c1)
    index.upsert_chunk(c2)

    matches = index.search_lexical("authenticate token verify", limit=5)
    assert len(matches) >= 1
    assert matches[0][0].chunk_id == "c1"
    assert matches[0][1] > 0.0

    index.close()


def test_rag_index_vector_search(tmp_path):
    db_path = tmp_path / "test_vector.db"
    index = RAGIndex(db_path=db_path)

    provider = DeterministicLocalEmbeddingProvider(dimension=64)
    text1 = "def authenticate_user_credentials(user, password): pass"
    text2 = "def format_markdown_table(data, headers): pass"

    v1 = provider.embed(text1)
    v2 = provider.embed(text2)

    c1 = CodeChunk(
        chunk_id="c1",
        file_path="auth.py",
        relative_path="auth.py",
        start_line=1,
        end_line=10,
        content=text1,
        chunk_type="function",
        symbol_name="auth",
        content_hash="h1",
    )
    c2 = CodeChunk(
        chunk_id="c2",
        file_path="format.py",
        relative_path="format.py",
        start_line=1,
        end_line=10,
        content=text2,
        chunk_type="function",
        symbol_name="format",
        content_hash="h2",
    )

    index.upsert_chunk(c1, v1)
    index.upsert_chunk(c2, v2)

    query_v = provider.embed("user authentication credentials")
    results = index.search_vector(query_v.vector, top_k=5)

    assert len(results) == 2
    # c1 should score higher than c2
    assert results[0][0].chunk_id == "c1"
    assert results[0][1] > results[1][1]

    index.close()


def test_index_versioning_and_model_change(tmp_path):
    db_path = tmp_path / "test_version.db"
    index = RAGIndex(db_path=db_path)

    # First check initializes metadata
    compat, msg = index.check_compatibility("deterministic-128", 128)
    assert compat is True
    assert index.get_metadata("index_version") == RAG_INDEX_VERSION
    assert index.get_metadata("embedding_model") == "deterministic-128"

    # Same model and dimension is compatible
    compat2, _ = index.check_compatibility("deterministic-128", 128)
    assert compat2 is True

    # Changed model is detected as incompatible
    compat3, msg3 = index.check_compatibility("nomic-embed-text", 384)
    assert compat3 is False
    assert "Embedding model mismatch" in msg3

    # Rebuild resets metadata
    index.rebuild()
    compat4, _ = index.check_compatibility("nomic-embed-text", 384)
    assert compat4 is True
    assert index.get_metadata("embedding_model") == "nomic-embed-text"

    index.close()


# =====================================================================
# 5. Hybrid Reranker Tests
# =====================================================================

def test_hybrid_reranker_fusion_and_bonuses():
    reranker = HybridReranker()
    chunk1 = CodeChunk(
        chunk_id="c1",
        file_path="agent/loop.py",
        relative_path="agent/loop.py",
        chunk_type="class",
        name="AgentLoop",
        start_line=10,
        end_line=50,
        content="class AgentLoop:\n    pass\n",
        symbols=["AgentLoop", "run"],
    )
    chunk2 = CodeChunk(
        chunk_id="c2",
        file_path="utils/helper.py",
        relative_path="utils/helper.py",
        chunk_type="function",
        name="format_time",
        start_line=1,
        end_line=10,
        content="def format_time(t): pass",
        symbols=["format_time"],
    )

    vector_results = [(chunk1, 0.85), (chunk2, 0.40)]
    lexical_results = [(chunk1, 0.70), (chunk2, 0.10)]

    # Query targeting AgentLoop symbol and agent/loop.py file
    query = RAGQuery(
        query="AgentLoop execution loop",
        target_symbols=["AgentLoop"],
        target_files=["agent/loop.py"],
    )

    ranked = reranker.rerank(
        lexical_results=lexical_results,
        semantic_results=vector_results,
        query=query,
    )
    assert len(ranked) == 2
    top = ranked[0]
    assert top.chunk.chunk_id == "c1"
    assert top.score > 0.4


# =====================================================================
# 6. Context Assembler & Budget Tests
# =====================================================================

def test_context_assembler_budget_and_disclaimer():
    assembler = ContextAssembler()
    chunks = [
        RetrievedChunk(
            chunk=CodeChunk(
                chunk_id=f"c_{i}",
                file_path=f"mod_{i}.py",
                relative_path=f"mod_{i}.py",
                chunk_type="function",
                name=f"func_{i}",
                start_line=1,
                end_line=10,
                content=f"def func_{i}():\n    return {i}\n",
                docstring=f"Function {i}",
            ),
            score=0.9 - (i * 0.1),
            similarity_score=0.85,
            lexical_score=0.80,
        )
        for i in range(5)
    ]

    context = assembler.assemble(chunks=chunks, max_tokens=100)
    assert "AUTHORITATIVE SYSTEM INVARIANT NOTICE" in context.prompt_text
    assert "RAG is a retrieval system, NOT the source of truth" in context.prompt_text
    assert len(context.retrieved_chunks) < len(chunks)


# =====================================================================
# 7. End-to-End RAG Manager Tests
# =====================================================================

def test_rag_manager_indexing_and_retrieval(tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "service.py").write_text(
        '"""Authentication microservice."""\n\n'
        'class AuthService:\n'
        '    """Handles user authorization."""\n'
        '    def verify(self, token: str) -> bool:\n'
        '        return token == "valid_token"\n\n'
        'def create_session(user_id: str) -> str:\n'
        '    """Creates a user session."""\n'
        '    return f"sess_{user_id}"\n',
        encoding="utf-8",
    )
    (ws / "math_utils.py").write_text(
        'def add_numbers(a: int, b: int) -> int:\n'
        '    """Adds two integers."""\n'
        '    return a + b\n',
        encoding="utf-8",
    )

    db_file = tmp_path / "rag.db"
    manager = RAGManager(workspace_root=ws, db_path=db_file)

    # 1. Index
    stats = manager.index_codebase()
    assert stats["indexed_files"] == 2
    assert stats["total_chunks"] >= 3

    # 2. Incremental re-index skips unchanged files
    stats2 = manager.index_codebase()
    assert stats2["skipped_files"] == 2

    # 3. Retrieve
    query = RAGQuery(query="user authorization token verification", top_k=3)
    results = manager.retrieve(query)
    assert len(results) > 0
    top = results[0]
    assert "service.py" in top.chunk.file_path

    # 4. Context assembly
    ctx = manager.assemble_context(query)
    assert ctx.total_chunks > 0
    assert "AUTHORITATIVE SYSTEM INVARIANT NOTICE" in ctx.prompt_text
    assert "AuthService" in ctx.prompt_text

    # 5. Status
    status = manager.get_status()
    assert status["enabled"] is True
    assert status["total_chunks"] >= 3

    # 6. Clear
    manager.clear()
    assert manager.get_status()["total_chunks"] == 0

    manager.close()


def test_rag_manager_protected_files_skipped(tmp_path):
    ws = tmp_path / "workspace_sec"
    ws.mkdir()
    (ws / ".env").write_text("DB_PASSWORD=supersecret", encoding="utf-8")
    (ws / "app.secret").write_text("KEY=12345", encoding="utf-8")
    (ws / "cert.pem").write_text("CERT_DATA", encoding="utf-8")
    (ws / "normal.py").write_text("def ping(): return 'pong'", encoding="utf-8")

    git_dir = ws / ".git"
    git_dir.mkdir()
    (git_dir / "config").write_text("git config", encoding="utf-8")

    db_file = tmp_path / "rag_sec.db"
    manager = RAGManager(workspace_root=ws, db_path=db_file)
    stats = manager.index_codebase()

    assert stats["indexed_files"] == 1
    # Ensure no protected files in index
    chunks = manager.index.get_chunks_by_file(".env")
    assert len(chunks) == 0
    chunks_git = manager.index.get_chunks_by_file(".git/config")
    assert len(chunks_git) == 0

    manager.close()


# =====================================================================
# 8. Architectural Invariants & Failure Safety
# =====================================================================

def test_rag_failure_safety_on_corrupt_db(tmp_path):
    corrupt_db = tmp_path / "corrupt.db"
    corrupt_db.write_text("NOT A VALID SQLITE DATABASE")

    manager = RAGManager(workspace_root=tmp_path, db_path=corrupt_db)
    # retrieve should return empty list gracefully
    res = manager.retrieve(RAGQuery(query="find anything"))
    assert res == []

    # assemble_context should return empty AssembledContext safely
    ctx = manager.assemble_context(RAGQuery(query="find anything"))
    assert ctx.total_chunks == 0
    assert ctx.context_text == ""


def test_rag_hallucinated_paths_rejected_by_filesystem_authority(tmp_path):
    """Verifies invariant: RAG is advisory; filesystem is authoritative."""
    ws = tmp_path / "ws"
    ws.mkdir()
    real_file = ws / "real_file.py"
    real_file.write_text("def run(): pass\n", encoding="utf-8")

    mock_rag = MagicMock()
    # RAG returns a hallucinated file path that does NOT exist on disk
    mock_rag.retrieve.return_value = [
        RetrievedChunk(
            chunk=CodeChunk(
                chunk_id="hallucinated",
                file_path="ghost/does_not_exist.py",
                relative_path="ghost/does_not_exist.py",
                chunk_type="function",
                name="ghost_fn",
                start_line=1,
                end_line=10,
                content="def ghost_fn(): pass",
            ),
            score=0.99,
            similarity_score=0.99,
            lexical_score=0.99,
        ),
        RetrievedChunk(
            chunk=CodeChunk(
                chunk_id="real",
                file_path="real_file.py",
                relative_path="real_file.py",
                chunk_type="function",
                name="run",
                start_line=1,
                end_line=5,
                content="def run(): pass",
            ),
            score=0.95,
            similarity_score=0.95,
            lexical_score=0.95,
        ),
    ]
    mock_rag.assemble_context.return_value = AssembledContext(
        context_text="Advisory RAG text",
        retrieved_chunks=[],
        total_tokens=20,
    )

    planner = DeveloperPlanner(workspace_root=ws, rag_manager=mock_rag)
    task = DeveloperTask(
        task_type=TaskType.CODE_CHANGE,
        is_development_task=True,
        goal="Update runner",
        requested_change="update run function",
    )
    plan = planner.create_plan(task)

    # Ghost file must NOT be in relevant_files because filesystem check overrides RAG!
    assert "ghost/does_not_exist.py" not in plan.relevant_files
    assert "real_file.py" in plan.relevant_files


def test_permissions_override_rag(tmp_path):
    """Verifies invariant: RAG context cannot authorize code modification."""
    from permissions import PermissionManager
    from developer.applier import CodeChangeApplier
    from developer.models import CodeChangeProposal, ApplicationStatus

    ws = tmp_path / "ws"
    ws.mkdir()
    f = ws / "target.py"
    f.write_text("x = 1\n", encoding="utf-8")

    perm_mgr = PermissionManager(default_write_allowed=False)
    applier = CodeChangeApplier(permission_manager=perm_mgr, workspace_root=ws)

    proposal = CodeChangeProposal(
        operation="modify_file",
        target_file="target.py",
        proposed_content="x = 2\n",
        description="Update x based on high confidence RAG score 0.999",
    )

    # Even with RAG suggestion, permission check MUST deny write when permission is not granted
    result = applier.apply_proposal(proposal, explicit_permission=False)
    assert result.success is False
    assert result.permission_granted is False
    assert result.status_code == ApplicationStatus.PERMISSION_DENIED.value
    assert f.read_text(encoding="utf-8") == "x = 1\n"


def test_generator_integrates_rag_context_into_llm_prompt(tmp_path):
    """Verifies that plan.rag_context is included in the LLM prompt messages."""
    ws = tmp_path / "ws"
    ws.mkdir()
    f = ws / "example.py"
    f.write_text("def existing(): pass\n", encoding="utf-8")

    mock_llm = MagicMock()
    mock_llm.generate.return_value = json.dumps({
        "generated_code": "def existing(): pass\ndef new_func(): pass\n",
        "explanation": "Added new_func",
        "affected_symbols": ["new_func"],
        "assumptions": ["Satisfies plan"],
    })

    generator = CodeChangeGenerator(workspace_root=ws, model=mock_llm)
    task = DeveloperTask(goal="Add new_func", requested_change="add new_func to example.py")
    plan = DeveloperPlan(
        original_task=task,
        relevant_files=["example.py"],
        rag_context={
            "prompt_text": "[VIDURA RAG CONTEXT - ADVISORY ONLY]\n# Chunk from example.py",
            "total_chunks": 1,
            "total_tokens": 15,
        },
    )

    proposal = generator.generate_proposal_from_plan(task=task, plan=plan)
    assert proposal.is_valid is True
    assert "new_func" in proposal.proposed_content

    # Inspect the prompt passed to mock_llm.generate
    call_args = mock_llm.generate.call_args[0][0]
    user_message = next(m for m in call_args if m["role"] == "user")
    assert "[VIDURA RAG CONTEXT - ADVISORY ONLY]" in user_message["content"]


# =====================================================================
# 9. CLI Command Handlers Tests
# =====================================================================

def test_handle_rag_command_search_and_stats(capsys):
    from main import handle_rag_command

    mock_rag = MagicMock()
    mock_rag.retrieve.return_value = [
        RetrievedChunk(
            chunk=CodeChunk(
                chunk_id="c1",
                file_path="service.py",
                relative_path="service.py",
                chunk_type="class",
                name="AuthService",
                start_line=1,
                end_line=20,
                content="class AuthService:\n    pass\n",
            ),
            score=0.92,
            similarity_score=0.90,
            lexical_score=0.85,
        )
    ]
    mock_rag.get_status.return_value = {
        "enabled": True,
        "total_chunks": 42,
        "indexed_files": 5,
        "embedding_model": "deterministic-local-128d",
        "dimensions": 128,
        "storage_type": "sqlite",
        "db_path": ":memory:",
    }
    mock_rag.index_codebase.return_value = {
        "total_chunks": 42,
        "indexed_files": 5,
        "skipped_files": 0,
        "errors": [],
    }
    mock_rag.assemble_context.return_value = AssembledContext(
        context_text="Sample Context Text",
        total_tokens=10,
    )

    # /rag search
    handled = handle_rag_command("/rag search authentication", mock_rag)
    assert handled is True
    out = capsys.readouterr().out
    assert "AuthService" in out
    assert "service.py" in out

    # /rag status
    handled = handle_rag_command("/rag status", mock_rag)
    assert handled is True
    out = capsys.readouterr().out
    assert "Total Chunks:    42" in out
    assert "deterministic-local-128d" in out

    # /rag index
    handled = handle_rag_command("/rag index", mock_rag)
    assert handled is True
    out = capsys.readouterr().out
    assert "RAG Indexing complete:" in out

    # /rag rebuild
    handled = handle_rag_command("/rag rebuild", mock_rag)
    assert handled is True
    out = capsys.readouterr().out
    assert "RAG Index Rebuild complete:" in out

    # /rag context
    handled = handle_rag_command("/rag context Implement login", mock_rag)
    assert handled is True
    out = capsys.readouterr().out
    assert "Sample Context Text" in out

    # Unknown command
    assert handle_rag_command("/unknown", mock_rag) is False


# =====================================================================
# 10. RAG Read-Only Tools Tests (Section 27 & 28)
# =====================================================================

def test_rag_semantic_search_tool():
    mock_rag = MagicMock()
    mock_rag.retrieve.return_value = [
        RetrievedChunk(
            chunk=CodeChunk(
                chunk_id="c1",
                file_path="agent/loop.py",
                relative_path="agent/loop.py",
                chunk_type="class",
                name="AgentLoop",
                start_line=120,
                end_line=180,
                content="class AgentLoop:\n    def run(self): pass\n",
            ),
            score=0.91,
            similarity_score=0.91,
            match_reasons=["semantic", "symbol"],
        )
    ]

    tool = SearchCodebaseSemanticTool(rag_manager=mock_rag)
    assert tool.name == "search_codebase_semantic"
    res = tool.execute(query="AgentLoop execution loop", top_k=3)

    assert res["success"] is True
    data = res["data"]
    assert data["count"] == 1
    chunk_res = data["results"][0]
    assert chunk_res["file"] == "agent/loop.py"
    assert chunk_res["symbol"] == "AgentLoop"
    assert chunk_res["score"] == 0.91
    assert "semantic" in chunk_res["matched_by"]


def test_rag_codebase_context_tool():
    mock_rag = MagicMock()
    mock_rag.assemble_context.return_value = AssembledContext(
        context_text="[VIDURA RAG CONTEXT - ADVISORY ONLY]\n# AgentLoop Context",
        retrieved_chunks=[],
        target_files=["agent/loop.py"],
        token_estimate=50,
    )

    tool = GetRelevantCodeContextTool(rag_manager=mock_rag)
    assert tool.name == "get_relevant_code_context"
    res = tool.execute(task_description="Fix agent loop bug", target_files=["agent/loop.py"])

    assert res["success"] is True
    data = res["data"]
    assert "ADVISORY ONLY" in data["context_text"]
    assert data["target_files"] == ["agent/loop.py"]
    assert data["is_authoritative"] is False
