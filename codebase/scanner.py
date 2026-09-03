import os
import logging
from pathlib import Path
from codebase.models import CodeFile
from config import load_config

logger = logging.getLogger("VIDURA.codebase.scanner")

IGNORED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    "build",
    "dist",
    "data",
    ".idea",
    ".vscode",
}

IGNORED_FILE_EXTENSIONS = {
    ".pyc",
    ".pyo",
    ".pyd",
    ".db",
    ".key",
    ".pem",
    ".secret",
    ".egg-info",
}

IGNORED_FILE_EXACT = {
    ".env",
}


class CodebaseScanner:
    """Recursively scans the workspace root safely for Python and project source files."""

    def __init__(self, workspace_root: Path | str | None = None) -> None:
        if workspace_root:
            self.workspace_root = Path(workspace_root).resolve()
        else:
            self.workspace_root = load_config().workspace_root.resolve()

    def is_ignored(self, path: Path) -> bool:
        """Determines if a path or file should be excluded from codebase indexing."""
        parts = path.parts
        # Check ignored directories
        if any(ignored in parts for ignored in IGNORED_DIRS):
            return True
        if any(part.endswith(".egg-info") for part in parts):
            return True

        filename = path.name
        if filename in IGNORED_FILE_EXACT or filename.startswith(".env"):
            return True
        if path.suffix.lower() in IGNORED_FILE_EXTENSIONS:
            return True

        return False

    def scan(self) -> list[CodeFile]:
        """Scans the workspace root and returns discovered CodeFile records."""
        discovered: list[CodeFile] = []
        root = self.workspace_root

        if not root.exists() or not root.is_dir():
            logger.warning(f"Workspace root '{root}' is invalid or missing.")
            return discovered

        for current_root, dirs, files in os.walk(root, followlinks=False):
            current_path = Path(current_root)

            # Filter out ignored directories in-place to prevent traversing down into them
            dirs[:] = [
                d for d in dirs
                if d not in IGNORED_DIRS
                and not d.endswith(".egg-info")
                and not d.startswith(".")
            ]

            for file_name in files:
                file_path = (current_path / file_name).resolve()

                # Enforce path security containment
                try:
                    if not file_path.is_relative_to(root):
                        logger.warning(f"Skipping file outside workspace root: {file_path}")
                        continue
                except ValueError:
                    continue

                if self.is_ignored(file_path):
                    continue

                try:
                    relative_path = str(file_path.relative_to(root))
                    stat = file_path.stat()
                    is_python = file_name.endswith(".py")
                    
                    discovered.append(
                        CodeFile(
                            path=str(file_path),
                            relative_path=relative_path,
                            size=stat.st_size,
                            is_python=is_python,
                        )
                    )
                except Exception as err:
                    logger.error(f"Error accessing file '{file_path}': {err}")

        logger.info(f"Scanned {len(discovered)} files in '{root}'.")
        return discovered
