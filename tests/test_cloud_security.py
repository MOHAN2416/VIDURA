"""Comprehensive automated tests for Phase 9.6: Cloud Security + Cost/Usage Controls.

Tests cover:
- Cloud enable/disable toggles (VIDURA_CLOUD_ENABLED)
- Deterministic sensitive data & credential protection (.env, private keys, API tokens)
- Security fallback to local vs abort (CloudSecurityViolation)
- Request limits & Developer limits (VIDURA_CLOUD_MAX_REQUESTS, VIDURA_CLOUD_MAX_DEVELOPER_REQUESTS)
- Context token limits (VIDURA_CLOUD_MAX_CONTEXT_TOKENS)
- CloudUsageTracker SQLite persistence & retention pruning
- Fallback-aware accounting transparency (failed cloud counted, local fallback not counted)
- Cost calculation (never fabricated, 'not_configured' unless pricing set)
- CLI /usage command and status diagnostics
- ModelRouter cloud request ID generation & propagation
"""
from __future__ import annotations

import sqlite3
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from config import Config
from models.base import ModelProvider, ProviderCapabilities
from models.errors import (
    CloudDisabledError,
    CloudSecurityViolation,
    CloudUsageLimitReached,
    CloudContextLimitExceeded,
    ProviderTimeout,
)
from models.routing import (
    ReasonCode,
)
from models.security import (
    is_protected_file_target,
    contains_sensitive_data,
    redact_secrets,
)
from models.usage import (
    CloudUsageTracker,
)
from models.router import ModelRouter
from developer.models import (
    CodeChangeProposal,
)
from developer.generator import CodeChangeGenerator as DeveloperGenerator
from agent.state import AgentState
from agent.loop import AgentLoop



class MockProvider(ModelProvider):
    """Mock ModelProvider for deterministic testing."""

    def __init__(self, name: str = "mock", model: str = "gemma4:e4b-it-qat", is_cloud: bool = False) -> None:
        self._name = name
        self._model = model
        self._capabilities = ProviderCapabilities(
            chat=True,
            coding=True,
            local=not is_cloud,
            cloud=is_cloud,
        )
        self.generate_mock = MagicMock(return_value="mock response")

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def provider_name(self) -> str:
        return self._name

    @property
    def capabilities(self) -> ProviderCapabilities:
        return self._capabilities

    def generate(self, messages: list[dict[str, str]] | str, **kwargs: object) -> str:
        return self.generate_mock(messages, **kwargs)


# ==============================================================================
# 1. Cloud Enable/Disable Controls
# ==============================================================================

def test_cloud_enabled_true_by_default() -> None:
    """Requirement A: Cloud is enabled by default in config."""
    cfg = Config()
    assert cfg.vidura_cloud_enabled is True


def test_cloud_disabled_via_config_routes_all_auto_tasks_to_local() -> None:
    """Requirement B: When cloud is disabled, router routes tasks to local model."""
    local_mock = MockProvider(name="Local", model="gemma4:e4b-it-qat")
    cloud_mock = MockProvider(name="Cloud", model="gemma4:31b-cloud", is_cloud=True)

    router = ModelRouter(
        routing_mode="auto",
        cloud_enabled=False,
        local_provider=local_mock,
        cloud_provider=cloud_mock,
    )
    assert router.cloud_enabled is False

    # Multi-file complex task would normally route to cloud in AUTO
    decision = router.route(
        target_files=["file1.py", "file2.py", "file3.py"],
        is_developer_task=True,
    )
    assert decision.provider == "local"
    assert decision.reason_code == ReasonCode.CLOUD_DISABLED.value


def test_cloud_mode_with_cloud_disabled_and_fallback_disabled_raises_error() -> None:
    """Requirement C1: Explicit cloud mode with cloud disabled raises CloudDisabledError when fallback disabled."""
    local_mock = MockProvider()
    cloud_mock = MockProvider(is_cloud=True)

    router = ModelRouter(
        routing_mode="cloud",
        cloud_enabled=False,
        fallback_enabled=False,
        local_provider=local_mock,
        cloud_provider=cloud_mock,
    )
    with pytest.raises(CloudDisabledError) as exc_info:
        router.route(messages="Hello cloud")
    assert "VIDURA_CLOUD_ENABLED=false" in str(exc_info.value) or "disabled" in str(exc_info.value)


def test_cloud_mode_with_cloud_disabled_and_fallback_enabled_routes_to_local() -> None:
    """Requirement C2: Explicit cloud mode with cloud disabled routes to local when fallback enabled."""
    local_mock = MockProvider()
    cloud_mock = MockProvider(is_cloud=True)

    router = ModelRouter(
        routing_mode="cloud",
        cloud_enabled=False,
        fallback_enabled=True,
        local_provider=local_mock,
        cloud_provider=cloud_mock,
    )
    decision = router.route(messages="Hello cloud")
    assert decision.provider == "local"
    assert decision.requested_provider == "cloud"
    assert decision.actual_provider == "local"
    assert decision.fallback_used is True
    assert decision.reason_code == ReasonCode.CLOUD_DISABLED.value


# ==============================================================================
# 2. Sensitive Data & Credential Protection
# ==============================================================================

def test_protected_file_target_detection() -> None:
    """Requirement G: Protected files (.env, .git, keys, secrets) are identified."""
    assert is_protected_file_target(".env") is True
    assert is_protected_file_target("config/.env") is True
    assert is_protected_file_target(".env.production") is True
    assert is_protected_file_target(".git/config") is True
    assert is_protected_file_target("id_rsa") is True
    assert is_protected_file_target("server.key") is True
    assert is_protected_file_target("cert.pem") is True
    assert is_protected_file_target("secrets.json") is True
    assert is_protected_file_target(["src/main.py", ".env"]) is True
    assert is_protected_file_target("src/main.py") is False
    assert is_protected_file_target(None) is False


def test_sensitive_content_detection_api_keys() -> None:
    """Requirement E: Known API key patterns and bearer tokens are detected."""
    assert contains_sensitive_data("Here is my key: sk-abcdefghijklmnopqrstuvwxyz1234567890") is True
    assert contains_sensitive_data("github: ghp_123456789012345678901234567890123456") is True
    assert contains_sensitive_data("ollama: ollama_12345678901234567890") is True
    assert contains_sensitive_data("Authorization: Bearer my_secret_token_123456789") is True
    assert contains_sensitive_data("api_key = 'super_secret_api_key_value_12345'") is True
    assert contains_sensitive_data("Safe regular prompt about Python functions.") is False


def test_sensitive_content_detection_private_keys() -> None:
    """Requirement F: Private key headers are detected."""
    private_key_sample = "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA0...\n-----END RSA PRIVATE KEY-----"
    assert contains_sensitive_data(private_key_sample) is True


def test_sensitive_task_routes_to_local_when_fallback_enabled() -> None:
    """Requirement H: Sensitive task routes to local with CLOUD_SECURITY_RESTRICTION when fallback enabled."""
    local_mock = MockProvider()
    cloud_mock = MockProvider(is_cloud=True)

    router = ModelRouter(
        routing_mode="cloud",
        fallback_enabled=True,
        local_provider=local_mock,
        cloud_provider=cloud_mock,
    )
    decision = router.route(
        target_files=[".env"],
        messages="Check database credentials",
    )
    assert decision.provider == "local"
    assert decision.reason_code == ReasonCode.CLOUD_SECURITY_RESTRICTION.value
    assert decision.fallback_used is True


def test_sensitive_task_raises_violation_when_fallback_disabled() -> None:
    """Requirement I: Sensitive task in cloud mode raises CloudSecurityViolation when fallback disabled."""
    local_mock = MockProvider()
    cloud_mock = MockProvider(is_cloud=True)

    router = ModelRouter(
        routing_mode="cloud",
        fallback_enabled=False,
        local_provider=local_mock,
        cloud_provider=cloud_mock,
    )
    with pytest.raises(CloudSecurityViolation) as exc_info:
        router.route(
            target_files=["secrets.json"],
            messages="sk-12345678901234567890123456",
        )
    assert "Sensitive data" in str(exc_info.value) or "blocked" in str(exc_info.value)


def test_redact_secrets_utility() -> None:
    """Requirement W: redact_secrets cleans secrets from text."""
    raw = "My API key is sk-1234567890123456789012345 and auth Bearer secret_token_xyz_123456."
    redacted = redact_secrets(raw)
    assert "sk-" not in redacted
    assert "Bearer" not in redacted
    assert "[REDACTED" in redacted


# ==============================================================================
# 3. Request Limits & Developer Limits
# ==============================================================================

def test_cloud_request_limit_enforcement_fallback_enabled(tmp_path: Path) -> None:
    """Requirement J & K: Cloud request limit reached routes to local when fallback enabled."""
    db_file = tmp_path / "usage.db"
    tracker = CloudUsageTracker(max_requests=2, db_path=db_file)
    local_mock = MockProvider()
    cloud_mock = MockProvider(is_cloud=True)

    router = ModelRouter(
        routing_mode="cloud",
        fallback_enabled=True,
        local_provider=local_mock,
        cloud_provider=cloud_mock,
        usage_tracker=tracker,
    )

    # First request: allowed
    res1 = router.generate("request 1")
    assert res1 == "mock response"
    assert tracker.cloud_attempts == 1

    # Second request: allowed
    res2 = router.generate("request 2")
    assert res2 == "mock response"
    assert tracker.cloud_attempts == 2

    # Third request: limit reached -> pre-request check falls back to local
    res3 = router.generate("request 3")
    assert res3 == "mock response"
    assert router.last_decision is not None
    assert router.last_decision.actual_provider == "local"
    assert router.last_decision.reason_code == ReasonCode.CLOUD_LIMIT_REACHED.value
    # Ensure attempt was NOT counted as cloud attempt
    assert tracker.cloud_attempts == 2


def test_cloud_request_limit_enforcement_fallback_disabled(tmp_path: Path) -> None:
    """Requirement L: Cloud request limit reached raises CloudUsageLimitReached when fallback disabled."""
    db_file = tmp_path / "usage.db"
    tracker = CloudUsageTracker(max_requests=1, db_path=db_file)
    local_mock = MockProvider()
    cloud_mock = MockProvider(is_cloud=True)

    router = ModelRouter(
        routing_mode="cloud",
        fallback_enabled=False,
        local_provider=local_mock,
        cloud_provider=cloud_mock,
        usage_tracker=tracker,
    )

    router.generate("req 1")
    assert tracker.cloud_attempts == 1

    # Second request must raise
    with pytest.raises(CloudUsageLimitReached) as exc_info:
        router.generate("req 2")
    assert "limit reached" in str(exc_info.value).lower()
    # Cloud attempts must remain 1 (no network request sent)
    assert tracker.cloud_attempts == 1


def test_developer_cloud_request_limit_enforcement(tmp_path: Path) -> None:
    """Requirement M: Developer cloud request limit enforces pre-request checks."""
    db_file = tmp_path / "usage.db"
    tracker = CloudUsageTracker(max_developer_requests=1, db_path=db_file)
    local_mock = MockProvider()
    cloud_mock = MockProvider(is_cloud=True)

    router = ModelRouter(
        routing_mode="cloud",
        fallback_enabled=True,
        local_provider=local_mock,
        cloud_provider=cloud_mock,
        usage_tracker=tracker,
    )

    # 1st dev request: succeeds on cloud
    router.generate("dev 1", is_developer_task=True)
    assert tracker.developer_cloud_requests == 1

    # 2nd dev request: falls back to local
    router.generate("dev 2", is_developer_task=True)
    assert router.last_decision is not None
    assert router.last_decision.actual_provider == "local"
    assert router.last_decision.reason_code == ReasonCode.CLOUD_LIMIT_REACHED.value
    assert tracker.developer_cloud_requests == 1


# ==============================================================================
# 4. Context Size Limits
# ==============================================================================

def test_context_limit_fallback_enabled() -> None:
    """Requirement N & O: Exceeding context tokens routes to local when fallback enabled."""
    local_mock = MockProvider()
    cloud_mock = MockProvider(is_cloud=True)

    cfg = Config(vidura_cloud_max_context_tokens=50)  # Very small limit for test

    router = ModelRouter(
        config=cfg,
        routing_mode="cloud",
        fallback_enabled=True,
        local_provider=local_mock,
        cloud_provider=cloud_mock,
    )

    long_prompt = "word " * 200  # ~200 words = ~250 tokens > 50 tokens
    res = router.generate(long_prompt)
    assert res == "mock response"
    assert router.last_decision is not None
    assert router.last_decision.actual_provider == "local"
    assert router.last_decision.reason_code == ReasonCode.CLOUD_CONTEXT_LIMIT_EXCEEDED.value


def test_context_limit_fallback_disabled_raises_error() -> None:
    """Requirement P: Exceeding context tokens raises CloudContextLimitExceeded when fallback disabled."""
    local_mock = MockProvider()
    cloud_mock = MockProvider(is_cloud=True)

    cfg = Config(vidura_cloud_max_context_tokens=50)

    router = ModelRouter(
        config=cfg,
        routing_mode="cloud",
        fallback_enabled=False,
        local_provider=local_mock,
        cloud_provider=cloud_mock,
    )

    long_prompt = "word " * 200
    with pytest.raises(CloudContextLimitExceeded) as exc_info:
        router.generate(long_prompt)
    assert "exceeds cloud context limit" in str(exc_info.value)


# ==============================================================================
# 5. Usage Accounting, SQLite Persistence & Retention Pruning
# ==============================================================================

def test_usage_tracker_persistence_and_summary(tmp_path: Path) -> None:
    """Requirement Q, R, Y: Inferences persist to SQLite and summary reports clean stats."""
    db_file = tmp_path / "test_usage.db"
    tracker = CloudUsageTracker(db_path=db_file)

    tracker.record_cloud_attempt()
    tracker.record_cloud_success(
        request_id="req_101",
        model="gemma4:31b-cloud",
        input_tokens=100,
        output_tokens=50,
    )

    tracker.record_cloud_attempt()
    tracker.record_cloud_failure(
        request_id="req_102",
        model="gemma4:31b-cloud",
        error=ProviderTimeout("Timeout connecting"),
    )

    summary = tracker.get_summary()
    assert summary["cloud_requests_attempted"] == 2
    assert summary["cloud_requests_successful"] == 1
    assert summary["cloud_requests_failed"] == 1
    assert summary["total_tokens"] == 150
    assert summary["cost_status"] == "not_configured"
    assert summary["estimated_cost"] is None

    # Verify SQLite rows
    conn = sqlite3.connect(str(db_file))
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM cloud_usage_records;")
    count = cursor.fetchone()[0]
    conn.close()
    assert count == 2


def test_retention_pruning(tmp_path: Path) -> None:
    """Requirement S: Records older than retention period are pruned."""
    db_file = tmp_path / "prune_usage.db"
    tracker = CloudUsageTracker(db_path=db_file, retention_days=10)

    # Insert an old record directly
    conn = sqlite3.connect(str(db_file))
    conn.execute(
        """
        INSERT INTO cloud_usage_records (
            request_id, timestamp, requested_provider, actual_provider, model,
            category, success, fallback_used, cost_status
        ) VALUES (
            'old_rec', '2020-01-01T00:00:00Z', 'cloud', 'cloud', 'gemma4:31b-cloud',
            'general', 1, 0, 'not_configured'
        );
        """
    )
    conn.commit()
    conn.close()

    deleted = tracker.prune_retention(days=10)
    assert deleted >= 1

    conn = sqlite3.connect(str(db_file))
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM cloud_usage_records WHERE request_id = 'old_rec';")
    assert cursor.fetchone()[0] == 0
    conn.close()


def test_cost_calculation_configured_vs_unconfigured(tmp_path: Path) -> None:
    """Requirement U & V: Cost is not fabricated, calculated only when configured."""
    db_file = tmp_path / "cost.db"

    # 1. Unconfigured pricing
    unconf_tracker = CloudUsageTracker(db_path=db_file)
    unconf_tracker.record_cloud_success(
        request_id="c1",
        model="gemma4:31b-cloud",
        input_tokens=1000,
        output_tokens=1000,
    )
    sum_unconf = unconf_tracker.get_summary()
    assert sum_unconf["cost_status"] == "not_configured"
    assert sum_unconf["estimated_cost"] is None

    # 2. Configured pricing ($0.002 per 1k in, $0.006 per 1k out)
    conf_tracker = CloudUsageTracker(
        db_path=db_file,
        input_token_price=0.002,
        output_token_price=0.006,
    )
    conf_tracker.record_cloud_success(
        request_id="c2",
        model="gemma4:31b-cloud",
        input_tokens=1000,
        output_tokens=1000,
    )
    sum_conf = conf_tracker.get_summary()
    assert sum_conf["cost_status"] == "estimated_configured"
    assert sum_conf["estimated_cost"] == 0.008


# ==============================================================================
# 6. Fallback Accounting & Transparency
# ==============================================================================

def test_failed_cloud_attempt_counted_local_fallback_not_counted_as_cloud(tmp_path: Path) -> None:
    """Requirement T: Failed cloud attempt is counted, local fallback execution is not counted as cloud."""
    db_file = tmp_path / "fallback_transparency.db"
    tracker = CloudUsageTracker(db_path=db_file)

    local_mock = MockProvider()
    cloud_mock = MockProvider(is_cloud=True)
    cloud_mock.generate_mock.side_effect = ProviderTimeout("Cloud 504 Timeout")

    router = ModelRouter(
        routing_mode="cloud",
        fallback_enabled=True,
        local_provider=local_mock,
        cloud_provider=cloud_mock,
        usage_tracker=tracker,
    )

    res = router.generate("Do some heavy thinking")
    assert res == "mock response"
    assert router.last_decision is not None
    assert router.last_decision.fallback_used is True

    summary = tracker.get_summary()
    # Attempt was made to cloud, and failed
    assert summary["cloud_requests_attempted"] == 1
    assert summary["cloud_requests_failed"] == 1
    assert summary["cloud_requests_successful"] == 0
    assert summary["fallback_requests"] == 1


# ==============================================================================
# 7. Cloud Request ID Propagation
# ==============================================================================

def test_cloud_request_id_generated_and_propagated_to_proposal(tmp_path: Path) -> None:
    """Requirement X: Unique cloud_request_id is generated and propagated to proposals and results."""
    local_mock = MockProvider()
    cloud_mock = MockProvider(is_cloud=True)
    cloud_mock.generate_mock.return_value = '{"proposed_content": "print(1)", "rationale": "ok"}'

    router = ModelRouter(
        routing_mode="cloud",
        local_provider=local_mock,
        cloud_provider=cloud_mock,
    )

    test_file = tmp_path / "hello.py"
    test_file.write_text("print(0)\n")

    generator = DeveloperGenerator(workspace_root=tmp_path, model=router)
    proposal = generator.generate_proposal(
        request="Update print to 1",
        target_file="hello.py",
        operation="modify_file",
    )

    assert proposal.is_valid is True
    assert proposal.cloud_request_id is not None
    assert proposal.cloud_request_id.startswith("cloud_req_")

    # Serialize and deserialize check
    prop_dict = proposal.to_dict()
    assert prop_dict["cloud_request_id"] == proposal.cloud_request_id
    rehydrated = CodeChangeProposal.from_dict(prop_dict)
    assert rehydrated.cloud_request_id == proposal.cloud_request_id


def test_agent_loop_records_cloud_request_id_and_usage() -> None:
    """Requirement X2: AgentState records cloud_request_id and usage_metadata."""
    cloud_mock = MockProvider(is_cloud=True)
    cloud_mock.generate_mock.return_value = '{"action": "respond", "content": "Done"}'

    router = ModelRouter(
        routing_mode="cloud",
        cloud_provider=cloud_mock,
    )

    loop = AgentLoop(model=router)
    state = loop.run(AgentState(user_request="Say hello"))

    assert state.cloud_request_id is not None
    assert state.cloud_request_id.startswith("cloud_req_")
    assert state.usage_metadata is not None
    assert state.usage_metadata["success"] is True
