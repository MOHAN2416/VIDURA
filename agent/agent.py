import logging
from models.base import BaseLLMProvider
from tools.registry import ToolRegistry
from memory import MemoryManager, extract_explicit_memory_request
from codebase import CodebaseManager
from agent.state import AgentState
from agent.loop import AgentLoop

logger = logging.getLogger("VIDURA.agent")


class Agent:
    """VIDURA High-Level Agent Orchestrator."""

    def __init__(
        self,
        model: BaseLLMProvider,
        tool_registry: ToolRegistry | None = None,
        memory_manager: MemoryManager | None = None,
        codebase_manager: CodebaseManager | None = None,
        max_steps: int = 10,
    ) -> None:
        self.model = model
        self.tool_registry = tool_registry
        self.memory_manager = memory_manager
        self.codebase_manager = codebase_manager
        self.max_steps = max_steps
        self.loop = AgentLoop(model=self.model, tool_registry=self.tool_registry)
        self.session_messages: list[dict[str, str]] = []

    def run(self, user_request: str) -> str:
        """Executes a user request through the agent loop and returns the final response string.

        Args:
            user_request: The user's input string.

        Returns:
            The final response string from the agent.
        """
        # 1. Detect explicit natural language memory requests
        if self.memory_manager:
            try:
                stmt, cat = extract_explicit_memory_request(user_request)
                if stmt:
                    saved_mem = self.memory_manager.remember(content=stmt, memory_type=cat)
                    logger.info(f"Explicit memory automatically stored: [{cat}] '{saved_mem.content}'")
            except Exception as err:
                logger.error(f"Error extracting memory request: {err}")

        # 2. Recall relevant memories for context injection
        memory_context = ""
        if self.memory_manager:
            try:
                memory_context = self.memory_manager.get_context_for_prompt(user_request)
            except Exception as err:
                logger.error(f"Error fetching memory context: {err}")

        # 3. Retrieve relevant codebase architecture context
        codebase_context = ""
        if self.codebase_manager:
            try:
                codebase_context = self.codebase_manager.get_context_for_prompt(user_request)
            except Exception as err:
                logger.error(f"Error fetching codebase context: {err}")

        # Append latest user request to session conversation history
        self.session_messages.append({"role": "user", "content": user_request})
        
        # Prepare working copy of messages
        working_messages = list(self.session_messages)
        
        # Inject memory and codebase contexts as system messages if available
        if memory_context:
            working_messages.insert(-1, {"role": "system", "content": memory_context})
        if codebase_context:
            working_messages.insert(-1, {"role": "system", "content": codebase_context})

        # Construct state
        state = AgentState(
            user_request=user_request,
            messages=working_messages,
            step=0,
            max_steps=self.max_steps,
        )

        # Run the agent execution loop
        final_state = self.loop.run(state)
        
        response = final_state.final_response or "No response generated."
        
        # Append assistant response to session conversation history
        self.session_messages.append({"role": "assistant", "content": response})

        return response
