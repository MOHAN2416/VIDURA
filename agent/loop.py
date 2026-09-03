import json
import logging
import re
from typing import Any
from models.base import BaseLLMProvider
from tools.registry import ToolRegistry
from agent.state import AgentState

logger = logging.getLogger("VIDURA.agent.loop")


def is_apply_request(user_request: str) -> bool:
    """Determines whether user_request expresses an explicit intent to apply a pending code change proposal."""
    if not user_request:
        return False
    req_lower = user_request.lower().strip()
    apply_phrases = [
        "apply the proposed",
        "apply the proposal",
        "apply proposal",
        "apply changes",
        "apply the change",
        "apply code change",
        "make the proposed change",
        "apply approved change",
        "apply pending proposal",
        "apply approved code changes",
        "now apply",
        "do it",
    ]
    if any(phrase in req_lower for phrase in apply_phrases):
        return True
    if req_lower.startswith("apply ") or req_lower == "apply":
        return True
    return False


def build_system_instruction(tool_registry: ToolRegistry | None = None) -> str:
    """Builds the base system instruction prompt for VIDURA's LLM Agent."""
    tools_description = ""
    if tool_registry:
        tools_list = []
        for tool in tool_registry.list_tools():
            tools_list.append(f"- {tool.name}: {tool.description}\n  Parameters: {json.dumps(tool.parameters)}")
        tools_description = "\n".join(tools_list)

    return f"""You are VIDURA, a deterministic, local AI pair-programming assistant built for strict codebase analysis and controlled developer actions.

Available Tools:
{tools_description or "No tools registered."}

STRICT EXECUTION GUIDELINES:
1. EXPLICIT READ-ONLY METHOD SEARCH MANDATE:
   - "Find symbol X" or "Search for function/class X" -> Call find_symbol(name=X).
   - "Find dependencies of module X" -> Call find_dependencies(module=X).
   - "Find importers of module X" or "Who imports X?" -> Call find_importers(module=X).
2. FINDING SYMBOLS & MODULE OVERVIEWS:
   - "Where is class/function X defined?" -> Call find_symbol(name=X).
   - "What modules exist?" or "What is inside module X?" -> Call inspect_codebase(target=X).
3. AUTHORITATIVE GROUNDING & ANTI-SPECULATION MANDATE:
   - Codebase tool results are authoritative facts. Report ONLY facts returned by tools or present in indexed data.
   - NEVER use speculative words ('likely', 'might', 'may', 'probably', 'appears to', 'could be', 'suggests') when describing codebase relationships. Use factual terms ('imports', 'depends on', 'is defined in').
   - If A imports B: A depends on B, and B is imported by A. Never reverse this relationship.
   - Do NOT claim a module 'uses' a symbol or module unless explicitly established. Use 'imports' for import relationships.
   - If a tool result does not establish a fact or returns success=false, state: "The indexed codebase does not provide enough information to determine this." Never invent facts or infer unreturned dependencies/importers.
4. CODE CHANGE PROPOSAL & UNSUPPORTED OPERATIONS MANDATES:
   - READ-ONLY PROPOSALS ONLY: The propose_code_change tool generates READ-ONLY proposals. NO files on disk are created, written, modified, or deleted.
   - NO APPLIED CLAIM RULE: NEVER claim a change "has been made", "has been applied", "file has been created", "file has been updated", or "I will delete...". ALWAYS frame the output strictly as a PROPOSED change and state clearly that no files on disk were modified or created.
   - EXPLICIT REJECTION OF UNSUPPORTED OPERATIONS: Deletion operations (delete_file, remove, destroy) are strictly unsupported. If requested to delete or remove a file, EXPLICITLY REJECT the request (e.g. "Deletion operations are unsupported in VIDURA. No files can be deleted."). NEVER present unsupported operations like deletion as proposals.
5. POST-CHANGE VERIFICATION MANDATE:
   - You MUST NOT claim a code change was successfully made or applied unless tool output explicitly contains verification_success=True and status_code='applied_and_verified'.
   - If tool output indicates verification_failed, application_failed, or stale_proposal, you MUST report the failure accurately as returned by the tool. Never override tool results to claim success.

For every step, you MUST respond ONLY with a valid JSON object in one of the following formats:

Format 1: To provide a final answer to the user:
{{
  "action": "respond",
  "content": "Your complete answer here"
}}

Format 2: To perform a reasoning step:
{{
  "action": "think",
  "content": "Description of what to analyze or think about"
}}

Format 3: To call an available tool:
{{
  "action": "tool_call",
  "tool_name": "tool_name_here",
  "arguments": {{ "arg1": "value1" }}
}}

Do not include any text outside the JSON object.
"""


from task_understanding.analyzer import DeveloperTaskAnalyzer


class AgentLoop:
    """Manages the iterative decision-action-observation execution loop for VIDURA."""

    def __init__(
        self,
        model: BaseLLMProvider,
        tool_registry: ToolRegistry | None = None,
        task_analyzer: DeveloperTaskAnalyzer | None = None,
    ) -> None:
        self.model = model
        self.tool_registry = tool_registry
        self.task_analyzer = task_analyzer or DeveloperTaskAnalyzer()

    def _parse_decision(self, raw_text: str) -> dict[str, Any] | None:
        """Extracts and parses JSON decision from raw model output."""
        if not raw_text:
            return None
            
        cleaned = raw_text.strip()
        # Try extracting JSON code block if wrapped in markdown ```json ... ```
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
        if json_match:
            cleaned = json_match.group(1).strip()
        else:
            # Fallback: search for first { and last }
            brace_match = re.search(r"(\{.*\})", cleaned, re.DOTALL)
            if brace_match:
                cleaned = brace_match.group(1).strip()

        try:
            data = json.loads(cleaned)
            if isinstance(data, dict) and "action" in data:
                return data
        except Exception:
            pass
            
        return None

    def run(self, state: AgentState) -> AgentState:
        """Executes the agent loop until completion or maximum steps reached."""
        logger.info(f"Starting Agent Loop for request: '{state.user_request}' (Max steps: {state.max_steps})")

        # Perform Task Understanding analysis (Read-Only)
        if not state.task_understanding and self.task_analyzer:
            try:
                state.task_understanding = self.task_analyzer.analyze(state.user_request)
                if state.task_understanding and state.task_understanding.is_development_task:
                    logger.info(
                        f"Developer Task Identified: type='{state.task_understanding.task_type.value}', "
                        f"files={state.task_understanding.target_files}, symbols={state.task_understanding.target_symbols}"
                    )
            except Exception as err:
                logger.warning(f"Task analysis failed closed: {err}")
        
        system_prompt = build_system_instruction(self.tool_registry)
        
        # Ensure base system prompt is at index 0 without overwriting other system messages
        if not state.messages or "You are VIDURA" not in state.messages[0].get("content", ""):
            state.messages.insert(0, {"role": "system", "content": system_prompt})
        else:
            state.messages[0] = {"role": "system", "content": system_prompt}

        is_apply = is_apply_request(state.user_request)

        while not state.is_finished():
            state.step += 1
            logger.info(f"[Step {state.step}/{state.max_steps}] Requesting model decision...")

            try:
                raw_response = self.model.generate(state.messages)
            except Exception as err:
                logger.error(f"[Step {state.step}] Model generation failed: {err}")
                state.final_response = f"Error during model generation: {err}"
                state.completed = False
                break

            decision = self._parse_decision(raw_response)

            # Deterministic Application Routing:
            # If user explicitly requests code application and tool 'apply_code_change' is available,
            # force routing to apply_code_change rather than accepting LLM natural language text or non-tool actions.
            if is_apply and self.tool_registry and self.tool_registry.get("apply_code_change") is not None:
                if not decision or decision.get("action") == "respond":
                    logger.info(f"[Step {state.step}] Apply request detected. Deterministically routing to tool 'apply_code_change'.")
                    decision = {
                        "action": "tool_call",
                        "tool_name": "apply_code_change",
                        "arguments": {"confirm_permission": True}
                    }

            if not decision:
                logger.warning(f"[Step {state.step}] Failed to parse JSON decision. Output: '{raw_response[:100]}...'")
                
                # If apply request, do NOT accept text fallback response as proof
                if is_apply:
                    logger.warning(f"[Step {state.step}] Direct text fallback prohibited for apply requests.")
                    decision = {
                        "action": "tool_call",
                        "tool_name": "apply_code_change",
                        "arguments": {"confirm_permission": True}
                    }
                else:
                    if raw_response and not raw_response.strip().startswith("{"):
                        logger.info(f"[Step {state.step}] Accepting direct text fallback response.")
                        state.current_action = "respond"
                        state.action_content = raw_response
                        state.final_response = raw_response
                        state.completed = True
                        break

                    feedback = (
                        "Your previous response was not valid JSON matching the required schema. "
                        'Please respond strictly with {"action": "respond", "content": "..."} or {"action": "tool_call", "tool_name": "...", "arguments": {...}}.'
                    )
                    state.messages.append({"role": "assistant", "content": raw_response})
                    state.messages.append({"role": "user", "content": feedback})
                    continue

            action = decision.get("action")
            content = decision.get("content", "")

            # Action Normalization: If LLM outputs a tool name directly as action (e.g. {"action": "propose_code_change", ...})
            if action and action not in ("respond", "think", "tool_call", "tool") and self.tool_registry and self.tool_registry.get(action) is not None:
                tool_name = action
                args = decision.get("arguments") or decision.get("args") or {
                    k: v for k, v in decision.items() if k not in ("action", "content")
                }
                decision = {
                    "action": "tool_call",
                    "tool_name": tool_name,
                    "arguments": args
                }
                action = "tool_call"

            state.current_action = action
            state.action_content = content
            logger.info(f"[Step {state.step}] Selected Action: '{action}'")

            if action == "respond":
                state.final_response = str(content)
                state.completed = True
                logger.info(f"[Step {state.step}] Action 'respond' executed successfully.")
                break

            elif action == "think":
                state.observation = f"Simulated thought processed: {content}"
                logger.info(f"[Step {state.step}] Observation: {state.observation}")
                state.messages.append({"role": "assistant", "content": json.dumps(decision)})
                state.messages.append({"role": "user", "content": f"Observation: {state.observation}"})

            elif action in ("tool_call", "tool"):
                tool_name = decision.get("tool_name") or decision.get("name") or ""
                arguments = decision.get("arguments") or decision.get("args") or {}

                state.tool_name = tool_name
                state.tool_arguments = arguments

                if not self.tool_registry:
                    result = {
                        "success": False,
                        "tool": tool_name,
                        "data": None,
                        "error": "No ToolRegistry configured in Agent."
                    }
                else:
                    result = self.tool_registry.execute(tool_name, arguments)

                state.observation = result
                logger.info(f"[Step {state.step}] Tool Result ({tool_name}): success={result.get('success')}")

                # Trusted Result Reporting for apply_code_change
                if tool_name == "apply_code_change":
                    if result.get("success") is True and result.get("data", {}).get("verification_success") is True:
                        target = result.get("data", {}).get("target_file", "file")
                        formatted = f"✅ The proposed code change to '{target}' was successfully applied and verified on disk."
                    else:
                        err_reason = result.get("error") or result.get("data", {}).get("reason") or "Application failed."
                        formatted = f"🛑 Code change application/verification failed: {err_reason}"

                    state.final_response = formatted
                    state.completed = True
                    logger.info(f"[Step {state.step}] Authoritative apply_code_change result enforced: '{formatted}'")
                    break

                state.messages.append({"role": "assistant", "content": json.dumps(decision)})
                state.messages.append({"role": "user", "content": f"Tool Result: {json.dumps(result)}"})

            else:
                logger.warning(f"[Step {state.step}] Unknown action '{action}'.")
                feedback = f"Action '{action}' is not supported. Valid actions are 'respond', 'think', or 'tool_call'."
                state.messages.append({"role": "assistant", "content": json.dumps(decision)})
                state.messages.append({"role": "user", "content": feedback})

        # Max steps limit check
        if state.step >= state.max_steps and not state.completed:
            logger.warning(f"Agent loop reached maximum step limit ({state.max_steps}) without completing.")
            if not state.final_response:
                state.final_response = f"Agent execution stopped: Reached maximum step limit ({state.max_steps}) without completing."
            state.completed = False

        return state
