import json
from pathlib import Path
from typing import Any
from models.base import BaseLLMProvider
from codebase import (
    CodeFile,
    CodebaseScanner,
    ASTParser,
    CodebaseIndex,
    CodebaseManager,
)
from tools.codebase import (
    InspectCodebaseTool,
    FindSymbolTool,
    FindImportersTool,
    FindDependenciesTool,
)
from agent import Agent


class FakeCodebaseModelProvider(BaseLLMProvider):
    """Fake model provider for testing agent codebase context injection."""

    def __init__(self, response_text: str = "Codebase context analyzed.") -> None:
        self.response_text = response_text
        self.last_messages: list[dict[str, str]] = []

    @property
    def model_name(self) -> str:
        return "fake-codebase-model"

    @property
    def provider_name(self) -> str:
        return "Fake Codebase Provider"

    def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        self.last_messages = list(messages)
        return json.dumps({"action": "respond", "content": self.response_text})


# --- Scanner Tests ---

def test_scanner_discovers_python_files_and_ignores_secrets(tmp_path: Path) -> None:
    """Test: Scanner discovers Python files and ignores .git, .venv, __pycache__, .env, data."""
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "main.py").write_text("print('hello')", encoding="utf-8")
    (tmp_path / "app" / "utils.py").write_text("def helper(): pass", encoding="utf-8")
    
    # Excluded items
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config.py").write_text("git config", encoding="utf-8")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "lib.py").write_text("venv code", encoding="utf-8")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "main.cpython-311.pyc").write_bytes(b"bytecode")
    (tmp_path / ".env").write_text("SECRET_KEY=12345", encoding="utf-8")
    (tmp_path / "secret.key").write_text("key_data", encoding="utf-8")

    scanner = CodebaseScanner(workspace_root=tmp_path)
    files = scanner.scan()

    discovered_paths = [f.relative_path for f in files]
    assert "app/main.py" in discovered_paths
    assert "app/utils.py" in discovered_paths
    assert not any(".git" in p for p in discovered_paths)
    assert not any(".venv" in p for p in discovered_paths)
    assert not any("__pycache__" in p for p in discovered_paths)
    assert not any(".env" in p for p in discovered_paths)
    assert not any(".key" in p for p in discovered_paths)


# --- Parser Tests ---

def test_parser_extracts_module_classes_functions_and_imports(tmp_path: Path) -> None:
    """Test: ASTParser extracts module docstrings, classes, methods, functions, and imports."""
    sample_code = '''"""Sample module docstring."""
import os
from pathlib import Path
from app.utils import helper

class Calculator:
    """Calculator class docstring."""
    def add(self, a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

async def fetch_data(url: str) -> dict:
    """Fetch data asynchronously."""
    return {}
'''
    py_file = tmp_path / "sample.py"
    py_file.write_text(sample_code, encoding="utf-8")

    parser = ASTParser()
    known_locals = {"app.utils", "app"}
    mod_info = parser.parse_file(str(py_file), "sample.py", known_local_modules=known_locals)

    assert mod_info.name == "sample"
    assert mod_info.docstring == "Sample module docstring."
    assert mod_info.parse_error is None
    assert len(mod_info.imports) == 3

    # Check local import detection
    local_imps = [i for i in mod_info.imports if i.is_local]
    assert len(local_imps) == 1
    assert local_imps[0].module == "app.utils"

    # Check Class extraction
    assert len(mod_info.classes) == 1
    cls = mod_info.classes[0]
    assert cls.name == "Calculator"
    assert cls.docstring == "Calculator class docstring."
    assert len(cls.methods) == 1
    assert cls.methods[0].name == "add"
    assert cls.methods[0].is_method is True

    # Check Function extraction
    assert len(mod_info.functions) == 1
    func = mod_info.functions[0]
    assert func.name == "fetch_data"
    assert func.is_async is True
    assert func.return_annotation == "dict"


def test_parser_malformed_python_error_handling(tmp_path: Path) -> None:
    """Test: Syntax/parse error in a Python file is captured in parse_error without crashing."""
    broken_code = "def broken_func("
    py_file = tmp_path / "broken.py"
    py_file.write_text(broken_code, encoding="utf-8")

    parser = ASTParser()
    mod_info = parser.parse_file(str(py_file), "broken.py")

    assert mod_info.name == "broken"
    assert mod_info.parse_error is not None
    assert "SyntaxError" in mod_info.parse_error


# --- Index & Dependency Graph Tests ---

def test_index_build_lookups_and_dependencies(tmp_path: Path) -> None:
    """Test: CodebaseIndex symbol search, importer graph, and dependency graph."""
    code_file1 = CodeFile(path=str(tmp_path / "mod_a.py"), relative_path="mod_a.py", size=100, is_python=True)
    code_file2 = CodeFile(path=str(tmp_path / "mod_b.py"), relative_path="mod_b.py", size=150, is_python=True)

    parser = ASTParser()
    (tmp_path / "mod_a.py").write_text("class Alpha:\n    def execute(self): pass", encoding="utf-8")
    (tmp_path / "mod_b.py").write_text("import mod_a\ndef run_b(): pass", encoding="utf-8")

    mod_a = parser.parse_file(str(tmp_path / "mod_a.py"), "mod_a.py", known_local_modules={"mod_a"})
    mod_b = parser.parse_file(str(tmp_path / "mod_b.py"), "mod_b.py", known_local_modules={"mod_a"})

    index = CodebaseIndex()
    index.build(files=[code_file1, code_file2], modules=[mod_a, mod_b])

    assert len(index.list_modules()) == 2
    assert index.get_module("mod_a") is not None

    # Symbol lookups
    symbols = index.find_symbol("Alpha")
    assert len(symbols) == 1
    assert symbols[0]["kind"] == "class"

    symbols_exec = index.find_symbol("execute")
    assert len(symbols_exec) == 1
    assert symbols_exec[0]["kind"] == "method"

    # Importer and Dependency Graphs
    importers = index.find_importers("mod_a")
    assert "mod_b" in importers

    deps = index.find_dependencies("mod_b")
    assert "mod_a" in deps

    summary = index.get_summary()
    assert summary["total_files"] == 2
    assert summary["python_files"] == 2
    assert summary["total_classes"] == 1


def test_symbol_to_module_auto_resolution(tmp_path: Path) -> None:
    """Phase 6.1 Test: Auto-resolves class/symbol names to canonical module dotted names."""
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "registry.py").write_text("class ToolRegistry:\n    pass", encoding="utf-8")
    (tmp_path / "agent").mkdir()
    (tmp_path / "agent" / "loop.py").write_text("from tools.registry import ToolRegistry\nclass AgentLoop:\n    pass", encoding="utf-8")

    manager = CodebaseManager(scanner=CodebaseScanner(workspace_root=tmp_path))
    manager.scan()

    # Test exact and casing symbol variations
    mod1, sym1 = manager.resolve_module_identifier("ToolRegistry")
    assert mod1 == "tools.registry"
    assert sym1 == "ToolRegistry"

    mod2, sym2 = manager.resolve_module_identifier("toolRegistry")
    assert mod2 == "tools.registry"

    mod3, sym3 = manager.resolve_module_identifier("Tool Registry")
    assert mod3 == "tools.registry"

    mod4, sym4 = manager.resolve_module_identifier("AgentLoop")
    assert mod4 == "agent.loop"


def test_resolve_symbol_to_module_and_canonical_target(tmp_path: Path) -> None:
    """Phase 6.1 Final Correction Test: resolve_symbol_to_module & resolve_target_to_canonical_module."""
    (tmp_path / "mod_x.py").write_text("class CustomEngine:\n    pass\n\ndef run_custom(): pass", encoding="utf-8")
    (tmp_path / "mod_y.py").write_text("import mod_x\nclass Runner:\n    pass", encoding="utf-8")

    manager = CodebaseManager(scanner=CodebaseScanner(workspace_root=tmp_path))
    manager.scan()

    # 1. Test resolve_symbol_to_module on class & function
    sym_class = manager.resolve_symbol_to_module("CustomEngine")
    assert sym_class is not None
    assert sym_class["symbol"] == "CustomEngine"
    assert sym_class["module"] == "mod_x"
    assert sym_class["kind"] == "class"

    sym_func = manager.resolve_symbol_to_module("run_custom")
    assert sym_func is not None
    assert sym_func["symbol"] == "run_custom"
    assert sym_func["module"] == "mod_x"
    assert sym_func["kind"] == "function"

    # 2. Test resolve_target_to_canonical_module with canonical module input
    canon, resolved_from, is_valid = manager.resolve_target_to_canonical_module("mod_x")
    assert canon == "mod_x"
    assert resolved_from is None
    assert is_valid is True

    # 3. Test resolve_target_to_canonical_module with symbol input
    canon_sym, resolved_from_sym, is_valid_sym = manager.resolve_target_to_canonical_module("Runner")
    assert canon_sym == "mod_y"
    assert resolved_from_sym == "Runner"
    assert is_valid_sym is True

    # 4. Test resolve_target_to_canonical_module with unknown input
    canon_unk, resolved_from_unk, is_valid_unk = manager.resolve_target_to_canonical_module("DoesNotExist")
    assert canon_unk is None
    assert resolved_from_unk == "DoesNotExist"
    assert is_valid_unk is False


def test_unknown_symbol_tool_error_handling(tmp_path: Path) -> None:
    """Phase 6.1 Final Correction Test: Unresolved symbols return error_result safely without executing fake lookups."""
    (tmp_path / "mod_a.py").write_text("class Alpha:\n    pass", encoding="utf-8")

    manager = CodebaseManager(scanner=CodebaseScanner(workspace_root=tmp_path))
    manager.scan()

    importers_tool = FindImportersTool(codebase_manager=manager)
    deps_tool = FindDependenciesTool(codebase_manager=manager)

    res_imp = importers_tool.execute(module="BananaController")
    assert res_imp["success"] is False
    assert "was not found in the indexed codebase" in res_imp["error"]

    res_dep = deps_tool.execute(module="BananaController")
    assert res_dep["success"] is False
    assert "was not found in the indexed codebase" in res_dep["error"]


def test_strict_grounded_summary_payloads(tmp_path: Path) -> None:
    """Phase 6.1 Final Patch Test: Tool result payloads include explicit grounded summary strings."""
    (tmp_path / "pkg_a.py").write_text("class Engine:\n    pass", encoding="utf-8")
    (tmp_path / "pkg_b.py").write_text("import pkg_a\nclass Client:\n    pass", encoding="utf-8")

    manager = CodebaseManager(scanner=CodebaseScanner(workspace_root=tmp_path))
    manager.scan()

    importers_tool = FindImportersTool(codebase_manager=manager)
    deps_tool = FindDependenciesTool(codebase_manager=manager)
    symbol_tool = FindSymbolTool(codebase_manager=manager)

    # Importers summary
    res_imp = importers_tool.execute(module="Engine")
    assert res_imp["success"] is True
    assert "summary" in res_imp["data"]
    assert "is imported by 1 local module(s): pkg_b" in res_imp["data"]["summary"]

    # Dependencies summary
    res_dep = deps_tool.execute(module="pkg_b")
    assert res_dep["success"] is True
    assert "summary" in res_dep["data"]
    assert "depends on 1 local module(s): pkg_a" in res_dep["data"]["summary"]

    # Symbol summary
    res_sym = symbol_tool.execute(name="Engine")
    assert res_sym["success"] is True
    assert "summary" in res_sym["data"]
    assert "Symbol 'Engine' found: class Engine in module 'pkg_a'" in res_sym["data"]["summary"]


def test_find_importers_and_dependencies_symbol_auto_resolution(tmp_path: Path) -> None:
    """Phase 6.1 Test: find_importers and find_dependencies resolve class symbols automatically."""
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "registry.py").write_text("class ToolRegistry:\n    pass", encoding="utf-8")
    (tmp_path / "main.py").write_text("from tools.registry import ToolRegistry\ndef run(): pass", encoding="utf-8")

    manager = CodebaseManager(scanner=CodebaseScanner(workspace_root=tmp_path))
    manager.scan()

    importers_info = manager.find_importers_info("ToolRegistry")
    assert importers_info["success"] is True
    assert importers_info["canonical_module"] == "tools.registry"
    assert importers_info["resolved_from_symbol"] == "ToolRegistry"
    assert "main" in importers_info["importers"]

    deps_info = manager.find_dependencies_info("main.py")
    assert deps_info["success"] is True
    assert deps_info["canonical_module"] == "main"
    assert "tools.registry" in deps_info["dependencies"]


# --- Codebase Tools Tests ---

def test_codebase_tools_execution(tmp_path: Path) -> None:
    """Test: InspectCodebaseTool, FindSymbolTool, FindImportersTool, FindDependenciesTool."""
    (tmp_path / "core.py").write_text("class CoreEngine:\n    def start(self): pass", encoding="utf-8")
    (tmp_path / "cli.py").write_text("import core\ndef main(): pass", encoding="utf-8")

    manager = CodebaseManager(scanner=CodebaseScanner(workspace_root=tmp_path))
    manager.scan()

    inspect_tool = InspectCodebaseTool(codebase_manager=manager)
    symbol_tool = FindSymbolTool(codebase_manager=manager)
    importers_tool = FindImportersTool(codebase_manager=manager)
    deps_tool = FindDependenciesTool(codebase_manager=manager)

    # Test inspect_codebase
    res_summary = inspect_tool.execute(target="summary")
    assert res_summary["success"] is True
    assert res_summary["data"]["python_files"] == 2

    res_mod = inspect_tool.execute(target="core.py")
    assert res_mod["success"] is True
    assert res_mod["data"]["name"] == "core"

    # Test find_symbol
    res_sym = symbol_tool.execute(name="CoreEngine")
    assert res_sym["success"] is True
    assert res_sym["data"]["count"] == 1

    # Test find_importers with class name symbol resolution
    res_imp = importers_tool.execute(module="CoreEngine")
    assert res_imp["success"] is True
    assert res_imp["data"]["canonical_module"] == "core"
    assert "cli" in res_imp["data"]["importers"]

    # Test find_dependencies
    res_dep = deps_tool.execute(module="cli")
    assert res_dep["success"] is True
    assert "core" in res_dep["data"]["dependencies"]


# --- Agent Integration Tests ---

def test_agent_codebase_context_injection(tmp_path: Path) -> None:
    """Test: Agent injects codebase architecture context into LLM prompt when query mentions codebase."""
    (tmp_path / "agent.py").write_text("class Agent:\n    def run(self): pass", encoding="utf-8")

    manager = CodebaseManager(scanner=CodebaseScanner(workspace_root=tmp_path))
    manager.scan()

    provider = FakeCodebaseModelProvider()
    agent = Agent(model=provider, codebase_manager=manager, max_steps=5)

    response = agent.run("Where is the Agent class defined in the codebase architecture?")
    assert response == "Codebase context analyzed."

    messages_str = json.dumps(provider.last_messages)
    assert "Codebase Intelligence Rules" in messages_str
    assert "Agent" in messages_str


def test_agent_irrelevant_codebase_query_excluded(tmp_path: Path) -> None:
    """Test: Unrelated query does not inject codebase context."""
    manager = CodebaseManager(scanner=CodebaseScanner(workspace_root=tmp_path))
    manager.scan()

    provider = FakeCodebaseModelProvider()
    agent = Agent(model=provider, codebase_manager=manager, max_steps=5)

    agent.run("What is 2 + 2?")
    messages_str = json.dumps(provider.last_messages)
    assert "Codebase Intelligence Rules" not in messages_str
