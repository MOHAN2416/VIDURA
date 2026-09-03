import pytest
from config import load_config, Config


def test_config_defaults() -> None:
    """Verifies default configuration values load correctly."""
    config = load_config()
    assert isinstance(config, Config)
    assert config.app_name == "VIDURA"
    assert config.app_env in ["development", "test", "production"]
    assert config.base_dir.exists()
    assert config.base_dir.is_dir()
    assert config.ollama_host == "http://localhost:11434"
    assert config.ollama_model == "qwen2.5:3b-instruct"


def test_config_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that environment variables override configuration defaults."""
    monkeypatch.setenv("APP_NAME", "VIDURA_TEST")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    monkeypatch.setenv("OLLAMA_MODEL", "qwen2.5:3b-instruct-custom")
    
    config = load_config()
    assert config.app_name == "VIDURA_TEST"
    assert config.log_level == "DEBUG"
    assert config.ollama_host == "http://127.0.0.1:11434"
    assert config.ollama_model == "qwen2.5:3b-instruct-custom"
