import ast
import logging
from pathlib import Path
from codebase.models import (
    ModuleInfo,
    ImportInfo,
    ClassInfo,
    FunctionInfo,
)

logger = logging.getLogger("VIDURA.codebase.parser")


class ASTParser:
    """Parses Python source files using standard library `ast` module safely."""

    @staticmethod
    def _path_to_module_name(relative_path: str) -> str:
        """Converts relative path to dotted Python module name."""
        p = Path(relative_path)
        parts = list(p.parts)
        if parts[-1].endswith(".py"):
            parts[-1] = parts[-1][:-3]
        if parts[-1] == "__init__":
            parts.pop()
        return ".".join(parts) if parts else "root"

    @staticmethod
    def _format_annotation(node: ast.expr | None) -> str | None:
        """Helper to safely format AST return or parameter annotations."""
        if node is None:
            return None
        try:
            return ast.unparse(node)
        except Exception:
            return str(node)

    @classmethod
    def _extract_parameters(cls, args_node: ast.arguments) -> list[dict[str, str]]:
        """Extracts function parameter names and type annotations."""
        params: list[dict[str, str]] = []
        for arg in args_node.args:
            name = arg.arg
            ann = cls._format_annotation(arg.annotation)
            param_info = {"name": name}
            if ann:
                param_info["type"] = ann
            params.append(param_info)
        return params

    def parse_file(
        self,
        file_path: str,
        relative_path: str,
        known_local_modules: set[str] | None = None,
    ) -> ModuleInfo:
        """Parses a Python file and returns a structured ModuleInfo record.

        Args:
            file_path: Full path to Python file.
            relative_path: Workspace relative path.
            known_local_modules: Optional set of known local module dotted names.

        Returns:
            Populated ModuleInfo record.
        """
        module_name = self._path_to_module_name(relative_path)
        known_locals = known_local_modules or set()

        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                source_code = f.read()
        except Exception as err:
            logger.error(f"Failed to read file '{file_path}': {err}")
            return ModuleInfo(
                name=module_name,
                file_path=file_path,
                relative_path=relative_path,
                parse_error=f"IO Error: {err}",
            )

        try:
            tree = ast.parse(source_code, filename=file_path)
        except SyntaxError as err:
            logger.warning(f"Syntax error parsing '{file_path}': {err}")
            return ModuleInfo(
                name=module_name,
                file_path=file_path,
                relative_path=relative_path,
                parse_error=f"SyntaxError: {err.msg} at line {err.lineno}",
            )
        except Exception as err:
            logger.warning(f"AST parsing error in '{file_path}': {err}")
            return ModuleInfo(
                name=module_name,
                file_path=file_path,
                relative_path=relative_path,
                parse_error=f"ASTError: {err}",
            )

        module_docstring = ast.get_docstring(tree)
        imports: list[ImportInfo] = []
        local_imports_set: set[str] = set()
        classes: list[ClassInfo] = []
        functions: list[FunctionInfo] = []

        # Helper to check if a module is local
        def is_local_module(mod_name: str) -> bool:
            if not mod_name:
                return False
            top_level = mod_name.split(".")[0]
            return (
                mod_name in known_locals
                or top_level in known_locals
                or mod_name.startswith(".")
            )

        for stmt in tree.body:
            # Handle 'import foo, bar'
            if isinstance(stmt, ast.Import):
                for alias in stmt.names:
                    mod = alias.name
                    is_local = is_local_module(mod)
                    imports.append(
                        ImportInfo(
                            module=mod,
                            names=[alias.asname or alias.name],
                            is_local=is_local,
                            line_number=stmt.lineno,
                        )
                    )
                    if is_local:
                        local_imports_set.add(mod)

            # Handle 'from foo import bar'
            elif isinstance(stmt, ast.ImportFrom):
                mod = stmt.module or ""
                if stmt.level > 0: # Relative import e.g. from . import foo
                    mod = "." * stmt.level + mod
                
                imported_names = [alias.name for alias in stmt.names]
                is_local = is_local_module(mod) or stmt.level > 0
                imports.append(
                    ImportInfo(
                        module=mod,
                        names=imported_names,
                        is_local=is_local,
                        line_number=stmt.lineno,
                    )
                )
                if is_local and mod:
                    local_imports_set.add(mod)

            # Handle Class Definitions
            elif isinstance(stmt, ast.ClassDef):
                bases = [self._format_annotation(b) or "" for b in stmt.bases]
                methods: list[FunctionInfo] = []

                for item in stmt.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        methods.append(
                            FunctionInfo(
                                name=item.name,
                                module_path=module_name,
                                parameters=self._extract_parameters(item.args),
                                return_annotation=self._format_annotation(item.returns),
                                docstring=ast.get_docstring(item),
                                line_number=item.lineno,
                                is_async=isinstance(item, ast.AsyncFunctionDef),
                                is_method=True,
                            )
                        )

                classes.append(
                    ClassInfo(
                        name=stmt.name,
                        module_path=module_name,
                        bases=[b for b in bases if b],
                        methods=methods,
                        docstring=ast.get_docstring(stmt),
                        line_number=stmt.lineno,
                    )
                )

            # Handle Top-Level Functions
            elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions.append(
                    FunctionInfo(
                        name=stmt.name,
                        module_path=module_name,
                        parameters=self._extract_parameters(stmt.args),
                        return_annotation=self._format_annotation(stmt.returns),
                        docstring=ast.get_docstring(stmt),
                        line_number=stmt.lineno,
                        is_async=isinstance(stmt, ast.AsyncFunctionDef),
                        is_method=False,
                    )
                )

        return ModuleInfo(
            name=module_name,
            file_path=file_path,
            relative_path=relative_path,
            docstring=module_docstring,
            imports=imports,
            local_imports=sorted(list(local_imports_set)),
            classes=classes,
            functions=functions,
            parse_error=None,
        )
