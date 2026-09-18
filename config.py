import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

# Base Directory of the project
BASE_DIR = Path(__file__).resolve().parent

# Load environment variables from .env if available
load_dotenv(BASE_DIR / ".env")


# Model Router Allowlists
ALLOWED_PROVIDERS = {"local", "cloud", "auto"}
ALLOWED_ROUTING_MODES = {"auto", "local", "cloud"}
ALLOWED_LOCAL_MODELS = {"gemma4:e4b-it-qat"}
ALLOWED_CLOUD_MODELS = {"gemma4:31b-cloud"}


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
    
    # Model configuration defaults & router settings
    ollama_host: str = field(
        default_factory=lambda: os.getenv("OLLAMA_HOST") or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434"
    )
    ollama_model: str = field(
        default_factory=lambda: os.getenv("VIDURA_LOCAL_MODEL") or os.getenv("OLLAMA_MODEL", "gemma4:e4b-it-qat")
    )
    vidura_local_model: str = field(
        default_factory=lambda: os.getenv("VIDURA_LOCAL_MODEL") or os.getenv("OLLAMA_MODEL", "gemma4:e4b-it-qat")
    )
    vidura_cloud_model: str = field(
        default_factory=lambda: os.getenv("VIDURA_CLOUD_MODEL") or os.getenv("OLLAMA_CLOUD_MODEL", "gemma4:31b-cloud")
    )
    vidura_routing_mode: str = field(
        default_factory=lambda: os.getenv("VIDURA_ROUTING_MODE", "auto")
    )
    vidura_model_provider: str = field(
        default_factory=lambda: os.getenv("VIDURA_MODEL_PROVIDER", "local")
    )
    vidura_developer_model_provider: str = field(
        default_factory=lambda: os.getenv("VIDURA_DEVELOPER_MODEL_PROVIDER") or "auto"
    )
    
    # Agent Loop settings
    agent_max_steps: int = field(
        default_factory=lambda: int(os.getenv("AGENT_MAX_STEPS", "10"))
    )
    
    # Cloud model configuration (never hardcoded, read at runtime)
    ollama_cloud_api_key: str | None = field(
        default_factory=lambda: os.getenv("VIDURA_CLOUD_API_KEY") or os.getenv("OLLAMA_CLOUD_API_KEY") or os.getenv("OLLAMA_API_KEY") or None
    )
    ollama_cloud_endpoint: str = field(
        default_factory=lambda: os.getenv("VIDURA_CLOUD_ENDPOINT") or os.getenv("OLLAMA_CLOUD_ENDPOINT") or "https://ollama.com"
    )
    vidura_cloud_timeout_seconds: float = field(
        default_factory=lambda: float(os.getenv("VIDURA_CLOUD_TIMEOUT_SECONDS", "60.0"))
    )
    ollama_cloud_model: str | None = field(
        default_factory=lambda: os.getenv("VIDURA_CLOUD_MODEL") or os.getenv("OLLAMA_CLOUD_MODEL", "gemma4:31b-cloud")
    )
    vidura_cloud_fallback_enabled: bool = field(
        default_factory=lambda: os.getenv("VIDURA_CLOUD_FALLBACK_ENABLED", "false").lower() in ("true", "1", "yes")
    )
    vidura_cloud_enabled: bool = field(
        default_factory=lambda: os.getenv("VIDURA_CLOUD_ENABLED", "true").lower() in ("true", "1", "yes")
    )
    vidura_cloud_max_requests: int | None = field(
        default_factory=lambda: int(os.getenv("VIDURA_CLOUD_MAX_REQUESTS")) if os.getenv("VIDURA_CLOUD_MAX_REQUESTS") and int(os.getenv("VIDURA_CLOUD_MAX_REQUESTS")) > 0 else None
    )
    vidura_cloud_max_developer_requests: int | None = field(
        default_factory=lambda: int(os.getenv("VIDURA_CLOUD_MAX_DEVELOPER_REQUESTS")) if os.getenv("VIDURA_CLOUD_MAX_DEVELOPER_REQUESTS") and int(os.getenv("VIDURA_CLOUD_MAX_DEVELOPER_REQUESTS")) > 0 else None
    )
    vidura_cloud_max_context_tokens: int = field(
        default_factory=lambda: int(os.getenv("VIDURA_CLOUD_MAX_CONTEXT_TOKENS", "32000"))
    )
    vidura_cloud_usage_retention_days: int = field(
        default_factory=lambda: int(os.getenv("VIDURA_CLOUD_USAGE_RETENTION_DAYS", "30"))
    )
    vidura_cloud_input_token_price: float | None = field(
        default_factory=lambda: float(os.getenv("VIDURA_CLOUD_INPUT_TOKEN_PRICE")) if os.getenv("VIDURA_CLOUD_INPUT_TOKEN_PRICE") else None
    )
    vidura_cloud_output_token_price: float | None = field(
        default_factory=lambda: float(os.getenv("VIDURA_CLOUD_OUTPUT_TOKEN_PRICE")) if os.getenv("VIDURA_CLOUD_OUTPUT_TOKEN_PRICE") else None
    )

    # Developer Testing settings
    test_timeout_seconds: int = field(
        default_factory=lambda: int(os.getenv("VIDURA_TEST_TIMEOUT_SECONDS", "30"))
    )
    test_output_limit: int = field(
        default_factory=lambda: int(os.getenv("VIDURA_TEST_OUTPUT_LIMIT", "50000"))
    )

    # Self-Development settings (Phase 10)
    vidura_self_development_enabled: bool = field(
        default_factory=lambda: os.getenv("VIDURA_SELF_DEVELOPMENT_ENABLED", "false").lower() in ("true", "1", "yes")
    )
    vidura_self_development_safety_mode: bool = field(
        default_factory=lambda: os.getenv("VIDURA_SELF_DEVELOPMENT_SAFETY_MODE", "true").lower() in ("true", "1", "yes")
    )

    # Experience Memory + Learning settings (Phase 11)
    vidura_experience_memory_enabled: bool = field(
        default_factory=lambda: os.getenv("VIDURA_EXPERIENCE_MEMORY_ENABLED", "true").lower() in ("true", "1", "yes")
    )
    vidura_experience_retrieval_enabled: bool = field(
        default_factory=lambda: os.getenv("VIDURA_EXPERIENCE_RETRIEVAL_ENABLED", "true").lower() in ("true", "1", "yes")
    )
    vidura_experience_max_results: int = field(
        default_factory=lambda: int(os.getenv("VIDURA_EXPERIENCE_MAX_RESULTS", "5"))
    )
    vidura_experience_min_relevance: float = field(
        default_factory=lambda: float(os.getenv("VIDURA_EXPERIENCE_MIN_RELEVANCE", "0.2"))
    )

    # RAG + Advanced Codebase Intelligence settings (Phase 12)
    vidura_rag_enabled: bool = field(
        default_factory=lambda: os.getenv("VIDURA_RAG_ENABLED", "true").lower() in ("true", "1", "yes")
    )
    vidura_rag_embedding_model: str = field(
        default_factory=lambda: os.getenv("VIDURA_RAG_EMBEDDING_MODEL", "all-minilm")
    )
    vidura_rag_top_k: int = field(
        default_factory=lambda: int(os.getenv("VIDURA_RAG_TOP_K", "5"))
    )
    vidura_rag_chunk_size: int = field(
        default_factory=lambda: int(os.getenv("VIDURA_RAG_CHUNK_SIZE", "500"))
    )
    vidura_rag_chunk_overlap: int = field(
        default_factory=lambda: int(os.getenv("VIDURA_RAG_CHUNK_OVERLAP", "50"))
    )
    vidura_rag_max_context_tokens: int = field(
        default_factory=lambda: int(os.getenv("VIDURA_RAG_MAX_CONTEXT_TOKENS", "4000"))
    )
    vidura_rag_min_score: float = field(
        default_factory=lambda: float(os.getenv("VIDURA_RAG_MIN_SCORE", "0.15"))
    )

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
