import os
from pathlib import Path
from typing import Any
from config import load_config
from tools.base import BaseTool


class SearchFilesTool(BaseTool):
    """Tool to search text files in the project workspace for a specific query string."""

    @property
    def name(self) -> str:
        return "search_files"

    @property
    def description(self) -> str:
        return "Search text files within the project workspace for a query string."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "String or pattern to search for."
                },
                "path": {
                    "type": "string",
                    "description": "Relative directory path within workspace to search in (defaults to '.')."
                }
            },
            "required": ["query"]
        }

    def execute(self, query: str = "", path: str = ".", **kwargs: Any) -> dict[str, Any]:
        if not query:
            return self.error_result("Parameter 'query' is required.")

        config = load_config()
        try:
            target_dir = config.validate_workspace_path(path)
        except ValueError as err:
            return self.error_result(str(err))

        if not target_dir.exists():
            return self.error_result(f"Search path '{path}' does not exist.")

        matches = []
        max_matches = 50
        ignored_dirs = {".git", ".venv", "__pycache__", ".pytest_cache", "build", "dist", "*.egg-info"}

        try:
            search_root = target_dir if target_dir.is_dir() else target_dir.parent
            
            for root, dirs, files in os.walk(search_root):
                # Prune ignored directories
                dirs[:] = [d for d in dirs if d not in ignored_dirs]
                
                for file_name in files:
                    file_path = Path(root) / file_name
                    
                    # Security check per file
                    try:
                        config.validate_workspace_path(file_path)
                    except ValueError:
                        continue

                    # Skip non-text extensions
                    if file_path.suffix.lower() in (".png", ".jpg", ".jpeg", ".pyc", ".so", ".bin", ".zip", ".tar", ".gz"):
                        continue

                    try:
                        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                            for idx, line in enumerate(f, start=1):
                                if query.lower() in line.lower():
                                    rel_file = str(file_path.relative_to(config.workspace_root))
                                    matches.append({
                                        "file": rel_file,
                                        "line_number": idx,
                                        "line_content": line.strip()
                                    })
                                    if len(matches) >= max_matches:
                                        break
                    except Exception:
                        continue

                    if len(matches) >= max_matches:
                        break
                if len(matches) >= max_matches:
                    break

            return self.success_result({
                "query": query,
                "matches": matches,
                "total_matches": len(matches)
            })

        except Exception as err:
            return self.error_result(f"Search operation failed for query '{query}': {err}")
