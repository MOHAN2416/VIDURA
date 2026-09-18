import json
from typing import Any
from models.base import BaseLLMProvider
from agent import Agent, AgentLoop, AgentState


class FakeModelProvider(BaseLLMProvider):
    """Fake model provider for deterministic unit testing without Ollama."""

    def __init__(self, responses: list[str] | None = None) -> None:
        self.responses = responses or []
        self.call_count = 0

    @property
    def model_name(self) -> str:
        return "fake-model"

    @property
    def provider_name(self) -> str:
        return "Fake Provider"

    def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        if not self.responses:
            return json.dumps({"action": "respond", "content": "Default fake response"})
        
        response = self.responses[min(self.call_count, len(self.responses) - 1)]
        self.call_count += 1
        return response


class FailingModelProvider(BaseLLMProvider):
    """Model provider that raises an exception when generate() is called."""

    @property
    def model_name(self) -> str:
        return "failing-model"

    @property
    def provider_name(self) -> str:
        return "Failing Provider"

    def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        raise RuntimeError("Simulated model connection crash")


def test_direct_response() -> None:
    """Test 1: Model returns direct respond action."""
    fake_response = json.dumps({"action": "respond", "content": "Python is a high-level programming language."})
    provider = FakeModelProvider([fake_response])
    agent = Agent(model=provider, max_steps=10)

    result = agent.run("What is Python?")

    assert result == "Python is a high-level programming language."
    assert provider.call_count == 1


def test_multi_step_loop() -> None:
    """Test 2: Multi-step loop with think followed by respond."""
    step1 = json.dumps({"action": "think", "content": "Analyzing two-step request"})
    step2 = json.dumps({"action": "respond", "content": "Two-step task finished."})
    provider = FakeModelProvider([step1, step2])
    agent = Agent(model=provider, max_steps=10)

    result = agent.run("Perform a two-step task.")

    assert result == "Two-step task finished."
    assert provider.call_count == 2


def test_max_steps_exceeded() -> None:
    """Test 3: Infinite think requests hit maximum step limit."""
    think_response = json.dumps({"action": "think", "content": "Endless thinking..."})
    provider = FakeModelProvider([think_response])
    
    max_steps = 3
    state = AgentState(user_request="Loop forever", max_steps=max_steps)
    loop = AgentLoop(model=provider)

    final_state = loop.run(state)

    assert final_state.step == max_steps
    assert final_state.completed is False
    assert "Reached maximum step limit" in final_state.final_response


def test_invalid_action_handling() -> None:
    """Test 4: Unsupported action is handled gracefully without crashing."""
    invalid_action = json.dumps({"action": "nonexistent_action", "content": "test"})
    valid_respond = json.dumps({"action": "respond", "content": "Recovered answer."})
    provider = FakeModelProvider([invalid_action, valid_respond])
    
    agent = Agent(model=provider, max_steps=5)
    result = agent.run("Test invalid action")

    assert result == "Recovered answer."
    assert provider.call_count == 2


def test_model_failure_handling() -> None:
    """Test 5: Exception raised by model is caught cleanly by agent loop."""
    failing_provider = FailingModelProvider()
    agent = Agent(model=failing_provider, max_steps=5)

    result = agent.run("Trigger model failure")

    assert "Error during model generation" in result
    assert "Simulated model connection crash" in result
