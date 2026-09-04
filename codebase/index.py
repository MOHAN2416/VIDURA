import logging
from typing import Any
from codebase.models import (
    CodeFile,
    ModuleInfo,
    ClassInfo,
    FunctionInfo,
)

logger = logging.getLogger("VIDURA.codebase.index")


class CodebaseIndex:
    """Deterministic in-memory index for fast codebase queries and dependency resolution."""

    def __init__(self) -> None:
        self.files: list[CodeFile] = []
        self.modules: dict[str, ModuleInfo] = {}  # Keyed by module name & relative path
        self.symbol_table: dict[str, list[dict[str, Any]]] = {}  # Symbol name -> list of symbol records
        self.import_graph: dict[str, set[str]] = {}  # module -> set of imported local modules
        self.importers_graph: dict[str, set[str]] = {}  # module -> set of modules importing it

    def clear(self) -> None:
        """Clears all indexed data."""
        self.files.clear()
        self.modules.clear()
        self.symbol_table.clear()
        self.import_graph.clear()
        self.importers_graph.clear()

    def build(self, files: list[CodeFile], modules: list[ModuleInfo]) -> None:
        """Populates index maps from scanned files and parsed module info."""
        self.clear()
        self.files = list(files)

        for mod in modules:
            # Map module by module name and relative path
            self.modules[mod.name] = mod
            self.modules[mod.relative_path] = mod

            # Populate Symbol Table
            # 1. Classes
            for cls in mod.classes:
                rec = {
                    "kind": "class",
                    "name": cls.name,
                    "module": mod.name,
                    "file_path": mod.relative_path,
                    "line_number": cls.line_number,
                    "bases": cls.bases,
                    "docstring": cls.docstring,
                }
                self.symbol_table.setdefault(cls.name, []).append(rec)

                # Class Methods
                for method in cls.methods:
                    method_rec = {
                        "kind": "method",
                        "name": method.name,
                        "class_name": cls.name,
                        "module": mod.name,
                        "file_path": mod.relative_path,
                        "line_number": method.line_number,
                        "parameters": method.parameters,
                        "return_annotation": method.return_annotation,
                        "docstring": method.docstring,
                        "is_async": method.is_async,
                    }
                    self.symbol_table.setdefault(method.name, []).append(method_rec)
                    # Also register Class.method
                    self.symbol_table.setdefault(f"{cls.name}.{method.name}", []).append(method_rec)

            # 2. Top-level Functions
            for func in mod.functions:
                func_rec = {
                    "kind": "function",
                    "name": func.name,
                    "module": mod.name,
                    "file_path": mod.relative_path,
                    "line_number": func.line_number,
                    "parameters": func.parameters,
                    "return_annotation": func.return_annotation,
                    "docstring": func.docstring,
                    "is_async": func.is_async,
                }
                self.symbol_table.setdefault(func.name, []).append(func_rec)

            # Populate Dependency Graphs
            self.import_graph[mod.name] = set(mod.local_imports)
            for imp_mod in mod.local_imports:
                self.importers_graph.setdefault(imp_mod, set()).add(mod.name)

        logger.info(
            f"Built codebase index: {len(self.files)} files, {len(modules)} modules, {len(self.symbol_table)} symbols."
        )

    def list_files(self) -> list[CodeFile]:
        return self.files

    def list_modules(self) -> list[ModuleInfo]:
        # Return unique ModuleInfo objects (since keys contain both name & relative_path)
        seen = set()
        unique_mods: list[ModuleInfo] = []
        for mod in self.modules.values():
            if mod.relative_path not in seen:
                seen.add(mod.relative_path)
                unique_mods.append(mod)
        return sorted(unique_mods, key=lambda m: m.name)

    def get_module(self, name_or_path: str) -> ModuleInfo | None:
        """Retrieves a ModuleInfo by dotted module name or relative path."""
        target = name_or_path.strip()
        if target in self.modules:
            return self.modules[target]

        # Check normalization
        if target.endswith(".py"):
            target_no_ext = target[:-3]
            if target_no_ext in self.modules:
                return self.modules[target_no_ext]

        return None

    def find_class(self, name: str) -> list[ClassInfo]:
        """Finds ClassInfo records by class name."""
        results: list[ClassInfo] = []
        for mod in self.list_modules():
            for cls in mod.classes:
                if cls.name.lower() == name.lower():
                    results.append(cls)
        return results

    def find_function(self, name: str) -> list[FunctionInfo]:
        """Finds FunctionInfo records by function name (top-level or method)."""
        results: list[FunctionInfo] = []
        for mod in self.list_modules():
            for func in mod.functions:
                if func.name.lower() == name.lower():
                    results.append(func)
            for cls in mod.classes:
                for method in cls.methods:
                    if method.name.lower() == name.lower():
                        results.append(method)
        return results

    def find_symbol(self, name: str) -> list[dict[str, Any]]:
        """Searches symbol table by exact or case-insensitive symbol name."""
        target = name.strip()
        if target in self.symbol_table:
            return self.symbol_table[target]

        # Case-insensitive search fallback
        target_lower = target.lower()
        matches: list[dict[str, Any]] = []
        for sym_name, recs in self.symbol_table.items():
            if sym_name.lower() == target_lower:
                matches.extend(recs)
        return matches

    def resolve_symbol_to_module(self, name: str) -> dict[str, Any] | None:
        """Finds the containing module details for an indexed class, function, or method symbol.

        Returns:
            Dictionary with symbol name, module, file_path, kind, and line_number, or None.
        """
        raw = name.strip()
        if not raw:
            return None

        # Search symbol table for exact or case-insensitive symbol match
        matches = self.find_symbol(raw)
        if not matches and " " in raw:
            cleaned = raw.replace(" ", "")
            matches = self.find_symbol(cleaned)

        if matches:
            first = matches[0]
            return {
                "symbol": first.get("name", raw),
                "module": first.get("module"),
                "file_path": first.get("file_path"),
                "kind": first.get("kind"),
                "line_number": first.get("line_number"),
            }
        return None

    def resolve_target_to_canonical_module(self, target: str) -> tuple[str | None, str | None, bool]:
        """Resolves a raw target (canonical module name, relative file path, or class/function symbol)
        to its canonical module dotted name.

        Returns:
            Tuple of (canonical_module_name, resolved_from_symbol_name, is_valid).
        """
        raw = target.strip()
        if not raw:
            return None, None, False

        # 1. Direct module or relative file path match in index
        mod_info = self.get_module(raw)
        if mod_info:
            return mod_info.name, None, True

        # 2. Check if target matches an indexed symbol
        sym_info = self.resolve_symbol_to_module(raw)
        if sym_info and sym_info.get("module"):
            return sym_info["module"], sym_info["symbol"], True

        # 3. Partial match in module names e.g. "loop" -> "agent.loop"
        raw_lower = raw.lower().replace(".py", "")
        for mod in self.list_modules():
            if mod.name.lower() == raw_lower or mod.name.endswith("." + raw_lower):
                return mod.name, raw, True

        # 4. Target is unresolved (neither a valid module nor an indexed symbol)
        return None, raw, False

    def resolve_module_identifier(self, identifier: str) -> tuple[str, str | None]:
        """Backwards compatible helper returning (canonical_module, resolved_from_symbol)."""
        canonical, sym, is_valid = self.resolve_target_to_canonical_module(identifier)
        return (canonical or identifier), sym

    def find_importers(self, module_name: str) -> list[str]:
        """Finds modules that import the given local module or resolved symbol's module."""
        mod_key, _ = self.resolve_module_identifier(module_name)
        importers = set(self.importers_graph.get(mod_key, set()))
        
        if not importers:
            for mod_k, imp_set in self.importers_graph.items():
                if mod_k.startswith(mod_key) or mod_key in mod_k:
                    importers.update(imp_set)

        return sorted(list(importers))

    def find_dependencies(self, module_name: str) -> list[str]:
        """Finds local modules imported by the given module or resolved symbol's module."""
        mod_key, _ = self.resolve_module_identifier(module_name)
        deps = set(self.import_graph.get(mod_key, set()))

        if not deps:
            for mod_k, dep_set in self.import_graph.items():
                if mod_k.startswith(mod_key) or mod_key in mod_k:
                    deps.update(dep_set)

        return sorted(list(deps))

    def get_summary(self) -> dict[str, Any]:
        unique_mods = self.list_modules()
        total_classes = sum(len(m.classes) for m in unique_mods)
        total_functions = sum(len(m.functions) + sum(len(c.methods) for c in m.classes) for m in unique_mods)
        parse_errors = [m for m in unique_mods if m.parse_error]

        # Top level packages
        packages = set()
        for m in unique_mods:
            parts = m.name.split(".")
            if len(parts) > 1 or (parts[0] and not m.relative_path.endswith("main.py") and not m.relative_path.endswith("config.py")):
                packages.add(parts[0])

        return {
            "total_files": len(self.files),
            "python_files": sum(1 for f in self.files if f.is_python),
            "total_modules": len(unique_mods),
            "total_classes": total_classes,
            "total_functions": total_functions,
            "top_level_packages": sorted(list(packages)),
            "parse_error_count": len(parse_errors),
            "parse_errors": [{"file": m.relative_path, "error": m.parse_error} for m in parse_errors],
        }
