import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

# Base Directory of the project
BASE_DIR = Path(__file__).resolve().parent

# Load environment variables from .env if available
load_dotenv(BASE_DIR / ".env")


@dataclass(frozen=True)
class Config:
    """VIDURA Configuration Settings."""
    app_name: str = field(default_factory=lambda: os.getenv("APP_NAME", "VIDURA"))
    app_env: str = field(default_factory=lambda: os.getenv("APP_ENV", "development"))
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    base_dir: Path = BASE_DIR
    workspace_root: Path = field(
        default_factory=lambda: Path(os.getenv("VIDURA_WORKSPACE_ROOT", str(BASE_DIR))).resolve()
    )
    db_path: Path = field(
        default_factory=lambda: Path(os.getenv("VIDURA_DB_PATH", str(BASE_DIR / "data" / "vidura.db"))).resolve()
    )
    
    # Local model configuration defaults
    ollama_host: str = field(
        default_factory=lambda: os.getenv("OLLAMA_HOST") or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434"
    )
    ollama_model: str = field(
        default_factory=lambda: os.getenv("OLLAMA_MODEL", "qwen2.5:3b-instruct")
    )
    
    # Agent Loop settings
    agent_max_steps: int = field(
        default_factory=lambda: int(os.getenv("AGENT_MAX_STEPS", "10"))
    )
    
    # Optional developer cloud model configuration (never hardcoded, for future phases)
    ollama_cloud_api_key: str | None = field(default_factory=lambda: os.getenv("OLLAMA_CLOUD_API_KEY") or None)
    ollama_cloud_model: str | None = field(default_factory=lambda: os.getenv("OLLAMA_CLOUD_MODEL") or None)

    def validate_workspace_path(self, target_path: str | Path) -> Path:
        """Resolves target_path and verifies it lies strictly within workspace_root.

        Args:
            target_path: Path string or Path object to validate.

        Returns:
            The resolved canonical Path.

        Raises:
            ValueError: If target_path resolves outside workspace_root.
        """
        root = self.workspace_root.resolve()
        path_obj = Path(target_path)
        
        if path_obj.is_absolute():
            resolved = path_obj.resolve()
        else:
            resolved = (root / path_obj).resolve()

        if not resolved.is_relative_to(root):
            raise ValueError(
                f"Access denied: Requested path '{target_path}' (resolved: '{resolved}') "
                f"is outside the allowed workspace boundary '{root}'."
            )
            
        return resolved


def configure_logging(level: str = "INFO") -> None:
    """Configures application-wide structured logging."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def load_config() -> Config:
    """Loads and returns the application configuration instance."""
    config = Config()
    configure_logging(config.log_level)
    return config
