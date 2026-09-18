"""VIDURA Code-Aware Chunker (Phase 12).

Decomposes Python and non-Python source files into structural, AST-aligned chunks
preserving symbol names, docstrings, imports, and exact line boundaries.
Enforces privacy by redacting credentials and secrets prior to chunk emission.
"""
from __future__ import annotations

import ast
import hashlib
import logging
from pathlib import Path
from typing import Any

from rag.models import CodeChunk
from models.security import redact_secrets

logger = logging.getLogger("VIDURA.rag.chunker")


class CodeAwareChunker:
    """Intelligently chunks codebase files along natural semantic and AST boundaries."""

    def __init__(
        self,
        default_chunk_size: int = 60,
        default_overlap: int = 10,
        workspace_root: str | Path | None = None,
        max_chunk_lines: int | None = None,
        chunk_overlap_lines: int | None = None,
    ) -> None:
        self.default_chunk_size = max_chunk_lines or default_chunk_size
        self.default_overlap = chunk_overlap_lines or default_overlap
        self.workspace_root = Path(workspace_root).resolve() if workspace_root else None

    def chunk_file(
        self,
        file_path: str | Path,
        relative_path: str | None = None,
        source_code: str | None = None,
    ) -> list[CodeChunk]:
        """Chunks a single file into structural CodeChunk instances."""
        path_obj = Path(file_path)
        
        # Handle case where source_code was passed as 2nd positional argument
        if relative_path is not None and source_code is None:
            if "\n" in relative_path or len(relative_path) > 100:
                source_code = relative_path
                relative_path = path_obj.name

        rel_path = relative_path or path_obj.name

        if source_code is None:
            try:
                source_code = path_obj.resolve().read_text(encoding="utf-8", errors="replace")
            except Exception as err:
                logger.error(f"Failed to read file for chunking '{path_obj}': {err}")
                return []

        # Redact secrets before any chunking or indexing
        sanitized_code = redact_secrets(source_code)
        file_hash = hashlib.sha256(sanitized_code.encode("utf-8")).hexdigest()

        # If file is Python, attempt AST-based chunking
        if path_obj.suffix.lower() == ".py" or rel_path.endswith(".py"):
            try:
                chunks = self._chunk_python(
                    source_code=sanitized_code,
                    file_path=str(path_obj),
                    relative_path=rel_path,
                    file_hash=file_hash,
                )
                if chunks:
                    return chunks
            except Exception as err:
                logger.warning(f"AST chunking failed for '{rel_path}': {err}. Falling back to sliding window.")

        # Fallback to sliding window for non-Python or unparseable files
        return self._chunk_sliding_window(
            source_code=sanitized_code,
            file_path=str(path_obj),
            relative_path=rel_path,
            file_hash=file_hash,
        )

    def _chunk_python(
        self,
        source_code: str,
        file_path: str,
        relative_path: str,
        file_hash: str,
    ) -> list[CodeChunk]:
        """Performs structural AST chunking on valid Python source."""
        tree = ast.parse(source_code, filename=file_path)
        lines = source_code.splitlines(keepends=True)
        total_lines = len(lines)

        chunks: list[CodeChunk] = []

        # 1. Module Header Chunk (Docstring, imports, and top-level definitions before first class/function)
        module_docstring = ast.get_docstring(tree)
        import_nodes: list[ast.stmt] = []
        for stmt in tree.body:
            if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                import_nodes.append(stmt)

        first_def_line = total_lines + 1
        for stmt in tree.body:
            if isinstance(stmt, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                first_def_line = stmt.lineno
                break

        module_header_end = first_def_line - 1 if first_def_line <= total_lines else total_lines
        if module_header_end >= 1:
            header_content = "".join(lines[:module_header_end]).strip()
            if header_content:
                imports_list = []
                for imp in import_nodes:
                    if isinstance(imp, ast.Import):
                        imports_list.extend(a.name for a in imp.names)
                    elif isinstance(imp, ast.ImportFrom):
                        mod = imp.module or ""
                        imports_list.extend(f"{mod}.{a.name}" for a in imp.names)

                chunks.append(
                    CodeChunk(
                        file_path=file_path,
                        relative_path=relative_path,
                        start_line=1,
                        end_line=module_header_end,
                        content=header_content,
                        chunk_type="module_header",
                        symbol_name=f"module:{Path(relative_path).stem}",
                        docstring=module_docstring,
                        imports=imports_list,
                        file_hash=file_hash,
                        metadata={"is_module_header": True},
                    )
                )

        # 2. Extract Top-level Classes and Functions
        for stmt in tree.body:
            # Classes
            if isinstance(stmt, ast.ClassDef):
                cls_chunks = self._chunk_class(
                    cls_node=stmt,
                    lines=lines,
                    file_path=file_path,
                    relative_path=relative_path,
                    file_hash=file_hash,
                )
                chunks.extend(cls_chunks)

            # Top-level Functions
            elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                fn_chunk = self._chunk_function(
                    fn_node=stmt,
                    lines=lines,
                    file_path=file_path,
                    relative_path=relative_path,
                    file_hash=file_hash,
                    parent_class=None,
                )
                chunks.append(fn_chunk)

        # If no structural chunks found (e.g. script without functions), fallback
        if not chunks:
            return self._chunk_sliding_window(
                source_code=source_code,
                file_path=file_path,
                relative_path=relative_path,
                file_hash=file_hash,
            )

        return chunks

    def _chunk_class(
        self,
        cls_node: ast.ClassDef,
        lines: list[str],
        file_path: str,
        relative_path: str,
        file_hash: str,
    ) -> list[CodeChunk]:
        """Extracts class definition header and individual methods as chunks."""
        chunks: list[CodeChunk] = []
        cls_name = cls_node.name
        start_line = cls_node.lineno
        end_line = getattr(cls_node, "end_lineno", len(lines))
        cls_docstring = ast.get_docstring(cls_node)

        # Class header chunk (class signature, docstring, bases, class attributes up to first method)
        first_method_line = end_line + 1
        methods: list[ast.FunctionDef | ast.AsyncFunctionDef] = []

        for item in cls_node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                methods.append(item)
                first_method_line = min(first_method_line, item.lineno)

        header_end = min(end_line, first_method_line - 1) if first_method_line <= end_line else end_line
        header_text = "".join(lines[start_line - 1 : header_end]).strip()

        bases_str = [ast.unparse(b) for b in cls_node.bases] if hasattr(ast, "unparse") else []

        if header_text:
            chunks.append(
                CodeChunk(
                    file_path=file_path,
                    relative_path=relative_path,
                    start_line=start_line,
                    end_line=header_end,
                    content=header_text,
                    chunk_type="class",
                    symbol_name=cls_name,
                    docstring=cls_docstring,
                    file_hash=file_hash,
                    metadata={"bases": bases_str, "method_count": len(methods)},
                )
            )

        # Individual method chunks
        for m in methods:
            method_chunk = self._chunk_function(
                fn_node=m,
                lines=lines,
                file_path=file_path,
                relative_path=relative_path,
                file_hash=file_hash,
                parent_class=cls_name,
            )
            chunks.append(method_chunk)

        return chunks

    def _chunk_function(
        self,
        fn_node: ast.FunctionDef | ast.AsyncFunctionDef,
        lines: list[str],
        file_path: str,
        relative_path: str,
        file_hash: str,
        parent_class: str | None = None,
    ) -> CodeChunk:
        """Extracts a function or method definition into a CodeChunk."""
        fn_name = fn_node.name
        symbol_name = f"{parent_class}.{fn_name}" if parent_class else fn_name
        chunk_type = "method" if parent_class else "function"

        start_line = fn_node.lineno
        end_line = getattr(fn_node, "end_lineno", len(lines))
        content = "".join(lines[start_line - 1 : end_line]).strip()
        docstring = ast.get_docstring(fn_node)

        is_async = isinstance(fn_node, ast.AsyncFunctionDef)
        params = [a.arg for a in fn_node.args.args]

        return CodeChunk(
            file_path=file_path,
            relative_path=relative_path,
            start_line=start_line,
            end_line=end_line,
            content=content,
            chunk_type=chunk_type,
            symbol_name=symbol_name,
            docstring=docstring,
            file_hash=file_hash,
            metadata={
                "is_async": is_async,
                "parameters": params,
                "parent_class": parent_class,
            },
        )

    def _chunk_sliding_window(
        self,
        source_code: str,
        file_path: str,
        relative_path: str,
        file_hash: str,
    ) -> list[CodeChunk]:
        """Sliding window fallback by line boundaries for non-Python or raw text files."""
        lines = source_code.splitlines(keepends=True)
        total_lines = len(lines)
        if total_lines == 0:
            return []

        chunk_size = self.default_chunk_size
        overlap = self.default_overlap
        step = max(1, chunk_size - overlap)

        chunks: list[CodeChunk] = []
        idx = 0

        while idx < total_lines:
            end_idx = min(total_lines, idx + chunk_size)
            chunk_lines = lines[idx:end_idx]
            content = "".join(chunk_lines).strip()

            if content:
                chunks.append(
                    CodeChunk(
                        file_path=file_path,
                        relative_path=relative_path,
                        start_line=idx + 1,
                        end_line=end_idx,
                        content=content,
                        chunk_type="block",
                        symbol_name=f"{Path(relative_path).name}:{idx + 1}-{end_idx}",
                        file_hash=file_hash,
                        metadata={"window_index": len(chunks)},
                    )
                )

            if end_idx >= total_lines:
                break
            idx += step

        return chunks
