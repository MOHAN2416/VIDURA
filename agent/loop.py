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
        "/approve",
        "approve",
        "approved",
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
        "yes",
        "proceed",
    ]
    if any(phrase == req_lower or req_lower.startswith(f"{phrase} ") for phrase in apply_phrases):
        return True
    if any(phrase in req_lower for phrase in apply_phrases):
        return True
    if req_lower.startswith("apply ") or req_lower == "apply":
        return True
    return False


def is_deny_request(user_request: str) -> bool:
    """Determines whether user_request expresses explicit intent to deny or reject a pending code change proposal."""
    if not user_request:
        return False
    req_lower = user_request.lower().strip()
    deny_phrases = ["/deny", "deny", "denied", "rejected", "reject", "cancel", "do not apply", "don't apply", "no"]
    return any(req_lower == p or req_lower.startswith(f"{p} ") for p in deny_phrases)


def claims_file_modification(text: str) -> bool:
    """Detects if natural language text claims that a file was created, modified, updated, or deleted."""
    if not text:
        return False
    trimmed = text.strip()
    if trimmed.startswith("🛑") or trimmed.startswith("✅"):
        return False
    lower = trimmed.lower()
    if "not applied" in lower or "permission denied" in lower or "rejected:" in lower or "application/verification failed" in lower:
        return False
    claim_phrases = [
        "successfully modified",
        "successfully created",
        "successfully updated",
        "successfully applied",
        "successfully changed",
        "successfully written",
        "successfully edited",
        "modified the file",
        "updated the file",
        "created the file",
        "edited the file",
        "written to the file",
        "file was modified",
        "file has been modified",
        "file has been updated",
        "file has been created",
        "file has been written",
        "file was successfully",
        "change has been made",
        "change has been applied",
        "changes have been applied",
        "changes have been made",
        "i changed the file",
        "i have changed",
        "i modified the file",
        "i have modified",
        "i have updated",
        "i created the file",
        "i have created",
        "i applied the change",
        "i have applied",
        "i have completed the change",
        "code has been applied",
        "code has been updated",
        "code has been written",
    ]
    return any(phrase in lower for phrase in claim_phrases)


def claims_test_success(text: str) -> bool:
    """Detects if natural language text claims that tests were run and passed."""
    if not text:
        return False
    trimmed = text.strip()
    if trimmed.startswith("🛑"):
        return False
    lower = trimmed.lower()
    test_phrases = [
        "tests passed",
        "all tests passed",
        "test suite passed",
        "tests were run and passed",
        "tests have passed",
        "tests were successful",
        "testing passed",
        "all tests are passing",
        "passed all tests",
        "tests succeed",
        "tests succeeded",
        "test run passed",
        "tests ran and passed",
        "all tests succeed",
    ]
    if any(p in lower for p in test_phrases):
        return True
    if re.search(r"\ball\s+\d+\s+passed\b", lower):
        return True
    if re.search(r"\b\d+\s+passed\b", lower) and ("test" in lower or "pytest" in lower):
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
from planning.planner import DeveloperPlanner
from codebase.manager import CodebaseManager
from developer.generator import DeveloperCodeGenerator
from developer.executor import DeveloperExecutor
from developer.applier import CodeChangeApplier


class AgentLoop:
    """Manages the iterative decision-action-observation execution loop for VIDURA."""

    def __init__(
        self,
        model: BaseLLMProvider,
        tool_registry: ToolRegistry | None = None,
        task_analyzer: DeveloperTaskAnalyzer | None = None,
        planner: DeveloperPlanner | None = None,
        codebase_manager: CodebaseManager | None = None,
        developer_generator: DeveloperCodeGenerator | None = None,
        executor: DeveloperExecutor | None = None,
    ) -> None:
        self.model = model
        self.tool_registry = tool_registry
        self.task_analyzer = task_analyzer or DeveloperTaskAnalyzer()
        self.codebase_manager = codebase_manager
        self.planner = planner or DeveloperPlanner(codebase_manager=self.codebase_manager, model=self.model)
        ws_root = getattr(self.codebase_manager, "workspace_root", None) if self.codebase_manager else None

        applier_inst = None
        if self.tool_registry and self.tool_registry.get("apply_code_change"):
            app_tool = self.tool_registry.get("apply_code_change")
            if hasattr(app_tool, "applier"):
                applier_inst = app_tool.applier

        if not applier_inst:
            applier_inst = CodeChangeApplier(workspace_root=ws_root)

        self.developer_generator = developer_generator or DeveloperCodeGenerator(
            workspace_root=ws_root,
            model=self.model,
            codebase_manager=self.codebase_manager,
            applier=applier_inst,
        )
        self.developer_executor = executor or DeveloperExecutor(
            applier=applier_inst,
            workspace_root=ws_root,
        )
        self.developer_generator.applier = applier_inst
        if self.tool_registry and self.tool_registry.get("apply_code_change"):
            app_tool = self.tool_registry.get("apply_code_change")
            if hasattr(app_tool, "executor"):
                app_tool.executor = self.developer_executor

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
                    # Perform Codebase-Aware Development Planning (Read-Only)
                    if not state.developer_plan and self.planner:
                        state.developer_plan = self.planner.plan(state.task_understanding)
                        logger.info(
                            f"Developer Plan Generated: files={state.developer_plan.relevant_files}, "
                            f"symbols={state.developer_plan.relevant_symbols}, "
                            f"requires_more_info={state.developer_plan.requires_more_information}"
                        )
            except Exception as err:
                logger.warning(f"Task analysis or planning failed closed: {err}")
        
        system_prompt = build_system_instruction(self.tool_registry)
        
        # Ensure base system prompt is at index 0 without overwriting other system messages
        if not state.messages or "You are VIDURA" not in state.messages[0].get("content", ""):
            state.messages.insert(0, {"role": "system", "content": system_prompt})
        else:
            state.messages[0] = {"role": "system", "content": system_prompt}

        # Check for explicit deny request
        if is_deny_request(state.user_request):
            logger.info("Deny request detected. Clearing active pending proposal and revoking permissions.")
            deny_res = self.developer_executor.deny_proposal()
            state.execution_result = deny_res
            state.final_response = "Change was not applied because permission was not granted."
            state.completed = True
            return state

        is_apply = is_apply_request(state.user_request)

        # If apply request and no tool registry available to route through, execute directly
        if is_apply and (not self.tool_registry or self.tool_registry.get("apply_code_change") is None):
            if not self.developer_executor.get_pending_proposal():
                state.final_response = "🛑 Code change application/verification failed: No active pending code change proposal found. Generate a proposal first."
                state.completed = True
                return state
            exec_res = self.developer_executor.execute_proposal(explicit_permission=True)
            state.execution_result = exec_res
            if not exec_res.success:
                state.final_response = f"🛑 Code change application/verification failed: {exec_res.error or exec_res.summary}"
            else:
                state.final_response = exec_res.summary
            state.completed = True
            return state

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
                        if claims_file_modification(raw_response):
                            logger.warning(f"[Step {state.step}] Suppressed false model claim in direct text fallback.")
                            state.current_action = "respond"
                            state.action_content = raw_response
                            state.final_response = "🛑 No code change was applied. Modifying files requires generating a proposal and receiving explicit user permission."
                            state.completed = True
                            break

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
                # Check for false claim: LLM claiming file was modified/created/applied without successful execution
                if claims_file_modification(str(content)):
                    was_verified = (
                        state.execution_result is not None and state.execution_result.success
                    ) or (
                        state.observation and isinstance(state.observation, dict) and state.observation.get("success") and state.observation.get("data", {}).get("verification_success")
                    )
                    if not was_verified:
                        logger.warning(f"[Step {state.step}] Suppressed false model claim of file modification.")
                        state.final_response = "🛑 No code change was applied. Modifying files requires generating a proposal and receiving explicit user permission."
                        state.completed = True
                        break

                # Check for false claim: LLM claiming tests passed without verified passing test result
                if claims_test_success(str(content)):
                    has_passing_tests = (
                        state.execution_result is not None
                        and state.execution_result.test_result is not None
                        and getattr(state.execution_result.test_result, "status", None) == "passed"
                    )
                    if not has_passing_tests:
                        logger.warning(f"[Step {state.step}] Suppressed false model claim of test success.")
                        state.final_response = "🛑 Tests did not pass or were not executed through the controlled test runner."
                        state.completed = True
                        break

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
                    if self.developer_executor and hasattr(self.developer_executor, "get_last_execution_result"):
                        state.execution_result = self.developer_executor.get_last_execution_result()
                    if result.get("success") is True and result.get("data", {}).get("verification_success") is True:
                        target = result.get("data", {}).get("target_file", "file")
                        exec_res = state.execution_result
                        if exec_res and exec_res.summary:
                            formatted = exec_res.summary
                        else:
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

        # Final false-claim safety check
        if state.final_response and claims_file_modification(state.final_response):
            was_verified = (
                state.execution_result is not None and state.execution_result.success
            ) or (
                state.observation and isinstance(state.observation, dict) and state.observation.get("success") and state.observation.get("data", {}).get("verification_success")
            )
            if not was_verified:
                logger.warning("Suppressed false model claim in final response.")
                state.final_response = "🛑 No code change was applied. Modifying files requires generating a proposal and receiving explicit user permission."

        if state.final_response and claims_test_success(state.final_response):
            has_passing_tests = (
                state.execution_result is not None
                and state.execution_result.test_result is not None
                and getattr(state.execution_result.test_result, "status", None) == "passed"
            )
            if not has_passing_tests:
                logger.warning("Suppressed false model claim of test success in final response.")
                state.final_response = "🛑 Tests did not pass or were not executed through the controlled test runner."

        # Max steps limit check
        if state.step >= state.max_steps and not state.completed:
            logger.warning(f"Agent loop reached maximum step limit ({state.max_steps}) without completing.")
            if not state.final_response:
                state.final_response = f"Agent execution stopped: Reached maximum step limit ({state.max_steps}) without completing."
            state.completed = False

        return state
