from typing import Any
from tools.base import BaseTool, success_result, error_result
from codebase.manager import CodebaseManager


class InspectCodebaseTool(BaseTool):
    """Tool to inspect project summary, list of modules/files, or module details."""

    def __init__(self, codebase_manager: CodebaseManager) -> None:
        self.codebase_manager = codebase_manager

    @property
    def name(self) -> str:
        return "inspect_codebase"

    @property
    def description(self) -> str:
        return (
            "Use when asked about: project overview, list of modules/files, "
            "or the contents/classes/functions inside a specific module (e.g., 'agent.loop' or 'memory.store')."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "description": (
                        "What to inspect: 'summary' for project overview, "
                        "'modules' to list all modules, 'files' to list all files, "
                        "or a specific module name/file path (e.g., 'agent.loop' or 'config.py')."
                    ),
                    "default": "summary",
                }
            },
            "required": [],
        }

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        target = kwargs.get("target", "summary").strip()

        try:
            if target == "summary":
                summary = self.codebase_manager.get_summary()
                return success_result(self.name, data=summary)

            elif target == "modules":
                modules = self.codebase_manager.list_modules()
                mod_list = [
                    {
                        "name": m.name,
                        "file": m.relative_path,
                        "classes": [c.name for c in m.classes],
                        "functions": [f.name for f in m.functions],
                    }
                    for m in modules
                ]
                return success_result(self.name, data={"count": len(mod_list), "modules": mod_list})

            elif target == "files":
                files = self.codebase_manager.list_files()
                file_list = [f.relative_path for f in files]
                return success_result(self.name, data={"count": len(file_list), "files": file_list})

            else:
                # Inspect specific module
                mod = self.codebase_manager.get_module(target)
                if not mod:
                    # Attempt symbol resolution
                    canonical, _ = self.codebase_manager.resolve_module_identifier(target)
                    mod = self.codebase_manager.get_module(canonical)

                if not mod:
                    return error_result(self.name, f"Module or file '{target}' not found in codebase index.")

                mod_data = {
                    "name": mod.name,
                    "file": mod.relative_path,
                    "docstring": mod.docstring,
                    "imports": [
                        {"module": i.module, "names": i.names, "is_local": i.is_local}
                        for i in mod.imports
                    ],
                    "local_imports": mod.local_imports,
                    "classes": [
                        {
                            "name": c.name,
                            "line": c.line_number,
                            "bases": c.bases,
                            "methods": [m.name for m in c.methods],
                            "docstring": c.docstring,
                        }
                        for c in mod.classes
                    ],
                    "functions": [
                        {
                            "name": f.name,
                            "line": f.line_number,
                            "is_async": f.is_async,
                            "docstring": f.docstring,
                        }
                        for f in mod.functions
                    ],
                    "parse_error": mod.parse_error,
                }
                return success_result(self.name, data=mod_data)

        except Exception as err:
            return error_result(self.name, f"Error inspecting codebase: {err}")


class FindSymbolTool(BaseTool):
    """Tool to search for classes, functions, or methods by symbol name."""

    def __init__(self, codebase_manager: CodebaseManager) -> None:
        self.codebase_manager = codebase_manager

    @property
    def name(self) -> str:
        return "find_symbol"

    @property
    def description(self) -> str:
        return "Use when asked WHERE a class, function, or method is defined (e.g. 'Where is AgentLoop defined?' or 'Find symbol ToolRegistry')."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "The symbol name to find (e.g. 'AgentLoop', 'ToolRegistry', 'MemoryStore').",
                }
            },
            "required": ["name"],
        }

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        symbol_name = kwargs.get("name", "").strip()
        if not symbol_name:
            return error_result(self.name, "Parameter 'name' is required.")

        try:
            matches = self.codebase_manager.find_symbol(symbol_name)
            if not matches:
                return success_result(self.name, data={"count": 0, "symbols": [], "summary": f"No symbol matching '{symbol_name}' was found in the codebase."})
            
            sym_descriptions = [f"{s.get('kind', 'symbol')} {s.get('name')} in module '{s.get('module')}' ({s.get('file_path')}:L{s.get('line_number')})" for s in matches]
            summary_str = f"Symbol '{symbol_name}' found: " + "; ".join(sym_descriptions)
            return success_result(self.name, data={"count": len(matches), "symbols": matches, "summary": summary_str})
        except Exception as err:
            return error_result(self.name, f"Error finding symbol: {err}")


class FindImportersTool(BaseTool):
    """Tool to identify local modules that import a target module or symbol."""

    def __init__(self, codebase_manager: CodebaseManager) -> None:
        self.codebase_manager = codebase_manager

    @property
    def name(self) -> str:
        return "find_importers"

    @property
    def description(self) -> str:
        return (
            "Use when asked: 'Which modules import/use X?' or 'Who depends on X?'. "
            "Finds local modules that IMPORT a target module (e.g. 'tools.registry' or class symbol 'ToolRegistry')."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "module": {
                    "type": "string",
                    "description": "Target module or class/symbol name (e.g. 'tools.registry', 'ToolRegistry', or 'memory.store').",
                }
            },
            "required": ["module"],
        }

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        module_name = kwargs.get("module", "").strip()
        if not module_name:
            return error_result(self.name, "Parameter 'module' is required.")

        try:
            info = self.codebase_manager.find_importers_info(module_name)
            if info.get("success") is False:
                return error_result(self.name, info.get("error", f"Module/symbol '{module_name}' not found."))
            
            canonical = info.get("canonical_module")
            importers = info.get("importers", [])
            resolved_from = info.get("resolved_from_symbol")
            target_desc = f"Module '{canonical}'" + (f" (resolved from symbol '{resolved_from}')" if resolved_from else "")

            if importers:
                info["summary"] = f"{target_desc} is imported by {len(importers)} local module(s): {', '.join(importers)}."
            else:
                info["summary"] = f"{target_desc} is not imported by any local modules."

            return success_result(self.name, data=info)
        except Exception as err:
            return error_result(self.name, f"Error finding importers: {err}")


class FindDependenciesTool(BaseTool):
    """Tool to list local modules imported by a target module or symbol."""

    def __init__(self, codebase_manager: CodebaseManager) -> None:
        self.codebase_manager = codebase_manager

    @property
    def name(self) -> str:
        return "find_dependencies"

    @property
    def description(self) -> str:
        return (
            "Use when asked: 'What does X import/depend on?' or 'What are the dependencies of X?'. "
            "Finds local modules IMPORTED BY target module X (e.g. 'agent.loop' or class symbol 'AgentLoop')."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "module": {
                    "type": "string",
                    "description": "Target module or class/symbol name (e.g. 'agent.loop', 'AgentLoop', or 'main.py').",
                }
            },
            "required": ["module"],
        }

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        module_name = kwargs.get("module", "").strip()
        if not module_name:
            return error_result(self.name, "Parameter 'module' is required.")

        try:
            info = self.codebase_manager.find_dependencies_info(module_name)
            if info.get("success") is False:
                return error_result(self.name, info.get("error", f"Module/symbol '{module_name}' not found."))

            canonical = info.get("canonical_module")
            deps = info.get("dependencies", [])
            resolved_from = info.get("resolved_from_symbol")
            target_desc = f"Module '{canonical}'" + (f" (resolved from symbol '{resolved_from}')" if resolved_from else "")

            if deps:
                info["summary"] = f"{target_desc} depends on {len(deps)} local module(s): {', '.join(deps)}."
            else:
                info["summary"] = f"{target_desc} has no local module dependencies."

            return success_result(self.name, data=info)
        except Exception as err:
            return error_result(self.name, f"Error finding dependencies: {err}")
