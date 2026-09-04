import logging
from typing import Any
from codebase.models import (
    CodeFile,
    ModuleInfo,
    ClassInfo,
    FunctionInfo,
)
from codebase.scanner import CodebaseScanner
from codebase.parser import ASTParser
from codebase.index import CodebaseIndex

logger = logging.getLogger("VIDURA.codebase.manager")


class CodebaseManager:
    """High-level Codebase Manager orchestrating scanning, AST parsing, and indexing."""

    def __init__(
        self,
        scanner: CodebaseScanner | None = None,
        parser: ASTParser | None = None,
        index: CodebaseIndex | None = None,
        workspace_root: str | None = None,
    ) -> None:
        self.scanner = scanner or CodebaseScanner(workspace_root=workspace_root)
        self.parser = parser or ASTParser()
        self.index = index or CodebaseIndex()
        self._is_scanned = False

    @property
    def workspace_root(self):
        """Returns the workspace root from scanner if available."""
        return self.scanner.workspace_root if self.scanner else None

    def scan(self) -> dict[str, Any]:
        """Scans workspace, parses all Python files, and builds the codebase index."""
        files = self.scanner.scan()
        python_files = [f for f in files if f.is_python]

        # Determine known local modules for local import resolution
        known_local_modules: set[str] = set()
        for pf in python_files:
            mod_name = ASTParser._path_to_module_name(pf.relative_path)
            known_local_modules.add(mod_name)
            top_level = mod_name.split(".")[0]
            if top_level:
                known_local_modules.add(top_level)

        parsed_modules: list[ModuleInfo] = []
        for pf in python_files:
            mod_info = self.parser.parse_file(
                file_path=pf.path,
                relative_path=pf.relative_path,
                known_local_modules=known_local_modules,
            )
            parsed_modules.append(mod_info)

        self.index.build(files=files, modules=parsed_modules)
        self._is_scanned = True
        return self.index.get_summary()

    def refresh(self) -> dict[str, Any]:
        """Refreshes the codebase index by re-scanning."""
        return self.scan()

    def ensure_scanned(self) -> None:
        """Ensures the workspace has been scanned at least once."""
        if not self._is_scanned:
            self.scan()

    def get_summary(self) -> dict[str, Any]:
        self.ensure_scanned()
        return self.index.get_summary()

    def list_files(self) -> list[CodeFile]:
        self.ensure_scanned()
        return self.index.list_files()

    def list_modules(self) -> list[ModuleInfo]:
        self.ensure_scanned()
        return self.index.list_modules()

    def get_module(self, name_or_path: str) -> ModuleInfo | None:
        self.ensure_scanned()
        return self.index.get_module(name_or_path)

    def find_class(self, name: str) -> list[ClassInfo]:
        self.ensure_scanned()
        return self.index.find_class(name)

    def find_function(self, name: str) -> list[FunctionInfo]:
        self.ensure_scanned()
        return self.index.find_function(name)

    def find_symbol(self, name: str) -> list[dict[str, Any]]:
        self.ensure_scanned()
        return self.index.find_symbol(name)

    def resolve_symbol_to_module(self, name: str) -> dict[str, Any] | None:
        self.ensure_scanned()
        return self.index.resolve_symbol_to_module(name)

    def resolve_target_to_canonical_module(self, target: str) -> tuple[str | None, str | None, bool]:
        self.ensure_scanned()
        return self.index.resolve_target_to_canonical_module(target)

    def resolve_module_identifier(self, identifier: str) -> tuple[str, str | None]:
        self.ensure_scanned()
        return self.index.resolve_module_identifier(identifier)

    def find_importers(self, module_name: str) -> list[str]:
        self.ensure_scanned()
        return self.index.find_importers(module_name)

    def find_importers_info(self, module_name: str) -> dict[str, Any]:
        self.ensure_scanned()
        canonical, resolved_from, is_valid = self.index.resolve_target_to_canonical_module(module_name)
        if not is_valid or not canonical:
            return {
                "success": False,
                "query_target": module_name,
                "canonical_module": None,
                "resolved_from_symbol": None,
                "error": f"Symbol or module '{module_name}' was not found in the indexed codebase.",
                "importers_count": 0,
                "importers": [],
            }

        importers = self.index.find_importers(canonical)
        return {
            "success": True,
            "query_target": module_name,
            "canonical_module": canonical,
            "resolved_from_symbol": resolved_from,
            "importers_count": len(importers),
            "importers": importers,
        }

    def find_dependencies(self, module_name: str) -> list[str]:
        self.ensure_scanned()
        return self.index.find_dependencies(module_name)

    def find_dependencies_info(self, module_name: str) -> dict[str, Any]:
        self.ensure_scanned()
        canonical, resolved_from, is_valid = self.index.resolve_target_to_canonical_module(module_name)
        if not is_valid or not canonical:
            return {
                "success": False,
                "query_target": module_name,
                "canonical_module": None,
                "resolved_from_symbol": None,
                "error": f"Symbol or module '{module_name}' was not found in the indexed codebase.",
                "dependencies_count": 0,
                "dependencies": [],
            }

        deps = self.index.find_dependencies(canonical)
        return {
            "success": True,
            "query_target": module_name,
            "canonical_module": canonical,
            "resolved_from_symbol": resolved_from,
            "dependencies_count": len(deps),
            "dependencies": deps,
        }

    def get_context_for_prompt(self, user_request: str) -> str:
        """Returns relevant codebase architecture context and tool selection guidance for LLM prompt."""
        if not user_request:
            return ""

        self.ensure_scanned()
        req_lower = user_request.lower()

        keywords = {
            "module", "modules", "class", "classes", "function", "functions",
            "symbol", "importer", "importers", "dependency", "dependencies",
            "codebase", "architecture", "structure", "import", "depends"
        }

        if not any(k in req_lower for k in keywords):
            return ""

        summary = self.index.get_summary()
        lines = [
            "[Codebase Intelligence Rules & Context]:",
            "- DEPENDENCIES: 'What does X depend on / import?' -> Call find_dependencies(X)",
            "- IMPORTERS: 'Who imports / uses X?' -> Call find_importers(X)",
            "- GROUNDING MANDATE: Tool results are authoritative facts. Do NOT use speculative words ('likely', 'might', 'may', 'probably', 'appears to', 'could be'). Report ONLY facts returned by tools.",
            f"Project Packages: {', '.join(summary['top_level_packages'])}",
        ]

        # Provide symbol-to-module mappings for key classes in project
        key_symbols = ["Agent", "AgentLoop", "ToolRegistry", "MemoryManager", "MemoryStore", "CodebaseManager", "OllamaProvider", "Config"]
        symbol_map_lines = []
        for sym in key_symbols:
            matches = self.index.find_symbol(sym)
            if matches:
                mod = matches[0].get("module")
                symbol_map_lines.append(f"{sym} -> {mod}")

        if symbol_map_lines:
            lines.append("Key Symbol -> Module Mappings:")
            lines.append("  " + ", ".join(symbol_map_lines))

        # Search for specific symbol mentions in user request
        words = [w.strip("?,.!\"'") for w in user_request.split()]
        found_symbols: list[dict[str, Any]] = []
        for word in words:
            if len(word) > 2 and not word.lower() in keywords:
                matches = self.index.find_symbol(word)
                if matches:
                    found_symbols.extend(matches)

        if found_symbols:
            lines.append("Mentioned Symbol Locations:")
            for sym in found_symbols[:5]:
                kind = sym.get("kind", "symbol")
                name = sym.get("name", "")
                mod = sym.get("module", "")
                path = sym.get("file_path", "")
                line = sym.get("line_number", 1)
                lines.append(f"- [{kind}] {name} is in module '{mod}' ({path}:L{line})")

        return "\n".join(lines)
