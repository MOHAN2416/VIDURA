# VIDURA

VIDURA is a permission-controlled, self-developing local AI assistant built in Python.

## Architecture Status

- **Phase 1: Project Foundation**: Packaging, configuration, environment loading, structured logging, entry point.
- **Phase 2: Local Offline LLM Integration**: Ollama integration using `qwen2.5:3b-instruct` with fallback error handling.
- **Phase 3: Agent Loop and Orchestration**: `Agent`, `AgentLoop`, `AgentState`, structured JSON decisions (`respond`, `think`), maximum-step protection.
- **Phase 4: Tool Calling & Tool Registry**: Safe read-only tools (`list_directory`, `read_file`, `search_files`), `ToolRegistry`, and strict workspace boundary path validation.
- **Phase 5: Persistent Memory System**: Local SQLite persistent memory (`data/vidura.db`), 4 memory categories (`user`, `project`, `conversation`, `experience`), duplicate prevention, keyword search recall, and CLI commands (`/remember`, `/recall`, `/memories`, `/forget`).
- **Phase 6: Codebase Intelligence**: Read-only Python AST parsing, workspace scanner, deterministic in-memory index, module & symbol resolution, local dependency graph, read-only tools (`inspect_codebase`, `find_symbol`, `find_importers`, `find_dependencies`), and CLI commands (`/codebase`, `/modules`, `/symbol`, `/dependencies`, `/importers`).

---

## 🛠️ Codebase Intelligence (Phase 6)

VIDURA includes a read-only Codebase Intelligence subsystem (`codebase/`) that provides structured, machine-readable understanding of its own Python codebase using Python's standard `ast` module.

### Capabilities & Security Rules
- **AST Python Parsing**: Extracts module docstrings, classes, methods, top-level functions, type annotations, line numbers, and imports deterministically without invoking the LLM.
- **Dependency & Importer Graphs**: Maps local project imports and tracks module-to-module dependencies.
- **Strictly Read-Only**: Performs no file creation, modification, deletion, code execution, or Git operations.
- **Workspace Security Boundary**: Rejects path traversal attempts (`../`, `/etc/passwd`) and excludes secrets (`.env`, `*.key`, `*.pem`, `*.secret`) and non-source directories (`.git`, `.venv`, `__pycache__`, `.pytest_cache`, `*.egg-info`).
- **Parsing Error Resilience**: Catches syntax/parser errors gracefully and marks files in `parse_errors` without crashing the index scan.

### Codebase Tools Registered
1. `inspect_codebase`: Summarizes project structure, lists modules/files, or inspects specific module ASTs.
2. `find_symbol`: Locates classes, functions, or methods across the codebase by name.
3. `find_importers`: Lists local modules importing a target module.
4. `find_dependencies`: Lists local modules imported by a target module.

### CLI Commands
- `/codebase` — Displays codebase summary metrics (files, modules, classes, functions, top-level packages).
- `/modules` — Lists all scanned modules and their defined classes.
- `/symbol <name>` — Searches symbols (e.g. `/symbol Agent` or `/symbol ToolRegistry`).
- `/dependencies <module>` — Displays dependencies imported by a module (e.g. `/dependencies main.py`).
- `/importers <module>` — Displays modules importing a target (e.g. `/importers memory.store`).

---

## 🚀 Getting Started

### Prerequisites

1. Python 3.11+
2. `uv` package manager
3. Local Ollama running with `qwen2.5:3b-instruct`:
   ```bash
   ollama pull qwen2.5:3b-instruct
   ```

### Running VIDURA

```bash
uv run python main.py
```

### Running Tests

To run the complete test suite cleanly:

```bash
PYTHONPATH="" PYTHONNOUSERSITE=1 uv run pytest
```
