import os
from pathlib import Path
from typing import Any
from config import load_config
from tools.base import BaseTool


class ListDirectoryTool(BaseTool):
    """Tool to list files and subdirectories within the project workspace."""

    @property
    def name(self) -> str:
        return "list_directory"

    @property
    def description(self) -> str:
        return "List files and subdirectories within an allowed project path."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path within project workspace to list (defaults to '.')."
                }
            },
            "required": []
        }

    def execute(self, path: str = ".", **kwargs: Any) -> dict[str, Any]:
        config = load_config()
        try:
            target_dir = config.validate_workspace_path(path)
        except ValueError as err:
            return self.error_result(str(err))

        if not target_dir.exists():
            return self.error_result(f"Directory '{path}' does not exist.")

        if not target_dir.is_dir():
            return self.error_result(f"Path '{path}' is a file, not a directory.")

        try:
            entries = []
            # Sort entries for consistent ordering
            for item in sorted(target_dir.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
                # Ignore hidden build/git caches if listing root to keep output concise
                if item.name in (".git", "__pycache__", ".pytest_cache", ".venv"):
                    continue
                entries.append({
                    "name": item.name,
                    "type": "directory" if item.is_dir() else "file"
                })

            rel_path = str(target_dir.relative_to(config.workspace_root))
            if rel_path == ".":
                rel_path = "./"

            return self.success_result({
                "path": rel_path,
                "entries": entries
            })
        except Exception as err:
            return self.error_result(f"Failed to list directory '{path}': {err}")


class ReadFileTool(BaseTool):
    """Tool to read text contents of a file within the project workspace."""

    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return "Read text contents of a file within the allowed project workspace."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file relative to the project workspace."
                }
            },
            "required": ["path"]
        }

    def execute(self, path: str = "", **kwargs: Any) -> dict[str, Any]:
        if not path:
            return self.error_result("Parameter 'path' is required.")

        config = load_config()
        try:
            target_file = config.validate_workspace_path(path)
        except ValueError as err:
            return self.error_result(str(err))

        if not target_file.exists():
            return self.error_result(f"File '{path}' does not exist.")

        if target_file.is_dir():
            return self.error_result(f"Path '{path}' points to a directory, not a file.")

        try:
            # Check for binary file extension or binary header
            with open(target_file, "r", encoding="utf-8", errors="replace") as f:
                content = f.read(50000) # Cap read size to 50KB to avoid memory overload
                
            rel_path = str(target_file.relative_to(config.workspace_root))
            return self.success_result({
                "path": rel_path,
                "content": content
            })
        except Exception as err:
            return self.error_result(f"Failed to read file '{path}': {err}")
