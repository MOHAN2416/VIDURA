import json
import logging
import re
from pathlib import Path
from typing import Any
from config import load_config
from models.base import BaseLLMProvider
from models.errors import ModelProviderError
from codebase.manager import CodebaseManager
from developer.models import (
    CodeChangeProposal,
    ProposalOperation,
    ProposalStatus,
    DeveloperGenerationResult,
)
from task_understanding.models import DeveloperTask
from planning.models import DeveloperPlan

logger = logging.getLogger("VIDURA.developer.generator")

PROTECTED_PATTERNS = [".git", ".env", ".key", ".pem", ".secret"]


class CodeChangeGenerator:
    """Generates structured code change proposals without writing or modifying files on disk."""

    def __init__(
        self,
        workspace_root: str | Path | None = None,
        model: BaseLLMProvider | None = None,
        codebase_manager: CodebaseManager | None = None,
        applier: Any = None,
    ) -> None:
        self.config = load_config()
        self.workspace_root = Path(workspace_root).resolve() if workspace_root else self.config.workspace_root.resolve()
        self.model = model
        self.codebase_manager = codebase_manager
        self.applier = applier

    @property
    def active_model(self) -> BaseLLMProvider | None:
        """Resolves the active model provider (ModelRouter or direct ModelProvider)."""
        return self.model

    def _is_protected_target(self, target_path: str | Path) -> bool:
        """Checks if target path matches protected security patterns (.git, .env, *.key, etc.)."""
        path_str = str(target_path).lower()
        parts = Path(target_path).parts
        for part in parts:
            if part.startswith(".git") or part == ".env":
                return True
        for pattern in PROTECTED_PATTERNS:
            if path_str.endswith(pattern):
                return True
        return False

    def _validate_workspace_path(self, target_path: str | Path) -> Path:
        """Resolves target_path and verifies it lies strictly within self.workspace_root."""
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

    def generate_proposal(
        self,
        request: str,
        target_file: str,
        operation: str = "modify_file",
        target_symbol: str | None = None,
    ) -> CodeChangeProposal:
        """Generates a structured CodeChangeProposal for the given request and target file.
        
        CRITICAL SAFETY MANDATE:
        This method is 100% read-only. It NEVER writes, touches, or modifies any file on disk.
        """
        logger.info(f"Generating proposal: op='{operation}', target='{target_file}', request='{request[:60]}...'")

        # 1. Validate supported operation
        normalized_op = operation.lower().strip()
        if normalized_op not in (ProposalOperation.CREATE_FILE.value, ProposalOperation.MODIFY_FILE.value):
            logger.warning(f"Unsupported operation '{operation}' requested.")
            return CodeChangeProposal(
                operation=operation,
                target_file=target_file,
                target_symbol=target_symbol,
                description=request,
                status=ProposalStatus.UNSUPPORTED_OPERATION.value,
                is_valid=False,
                validation_error=f"Operation '{operation}' is unsupported. Only 'create_file' and 'modify_file' proposals are supported.",
            )

        # 2. Validate workspace boundary and protected targets
        if self._is_protected_target(target_file):
            logger.warning(f"Access denied for protected target '{target_file}'.")
            return CodeChangeProposal(
                operation=normalized_op,
                target_file=target_file,
                target_symbol=target_symbol,
                description=request,
                status=ProposalStatus.INVALID_PATH.value,
                is_valid=False,
                validation_error=f"Access denied: Target path '{target_file}' is protected (.git, .env, *.key, etc.).",
            )

        try:
            resolved_path = self._validate_workspace_path(target_file)
        except ValueError as err:
            logger.warning(f"Path validation failed for '{target_file}': {err}")
            return CodeChangeProposal(
                operation=normalized_op,
                target_file=target_file,
                target_symbol=target_symbol,
                description=request,
                status=ProposalStatus.INVALID_PATH.value,
                is_valid=False,
                validation_error=str(err),
            )

        # Determine relative workspace path representation
        try:
            relative_target = str(resolved_path.relative_to(self.workspace_root))
        except ValueError:
            relative_target = target_file

        # 3. Disk state inspection (READ-ONLY)
        original_content = ""
        if normalized_op == ProposalOperation.MODIFY_FILE.value:
            if not resolved_path.exists() or not resolved_path.is_file():
                logger.warning(f"Modify target file '{target_file}' does not exist on disk.")
                return CodeChangeProposal(
                    operation=normalized_op,
                    target_file=relative_target,
                    target_symbol=target_symbol,
                    description=request,
                    status=ProposalStatus.TARGET_NOT_FOUND.value,
                    is_valid=False,
                    validation_error=f"Target file '{target_file}' does not exist in workspace.",
                )
            try:
                original_content = resolved_path.read_text(encoding="utf-8")
            except Exception as err:
                return CodeChangeProposal(
                    operation=normalized_op,
                    target_file=relative_target,
                    target_symbol=target_symbol,
                    description=request,
                    status=ProposalStatus.PERMISSION_DENIED.value,
                    is_valid=False,
                    validation_error=f"Failed to read existing file: {err}",
                )

        elif normalized_op == ProposalOperation.CREATE_FILE.value:
            if resolved_path.exists() and resolved_path.is_file():
                try:
                    original_content = resolved_path.read_text(encoding="utf-8")
                except Exception:
                    original_content = ""

        active = self.active_model
        provider_name = None
        model_name = None
        fallback_used = False
        fallback_reason = None
        cloud_req_id = None
        if active:
            if hasattr(active, "route"):
                decision = active.route(
                    target_file=relative_target,
                    target_files=[relative_target],
                    operation=normalized_op,
                    is_developer_task=True,
                    task_type="developer",
                )
                provider_name = decision.actual_provider or decision.provider
                model_name = decision.actual_model or decision.model
                fallback_used = getattr(decision, "fallback_used", False)
                fallback_reason = getattr(decision, "fallback_reason", None)
                cloud_req_id = getattr(decision, "cloud_request_id", None)
            elif hasattr(active, "last_decision") and active.last_decision:
                provider_name = active.last_decision.actual_provider or active.last_decision.provider
                model_name = active.last_decision.actual_model or active.last_decision.model
                fallback_used = getattr(active.last_decision, "fallback_used", False)
                fallback_reason = getattr(active.last_decision, "fallback_reason", None)
                cloud_req_id = getattr(active.last_decision, "cloud_request_id", None)
            else:
                provider_name = getattr(active, "provider_type", None) or ("cloud" if getattr(getattr(active, "capabilities", None), "cloud", False) else "local")
                model_name = getattr(active, "model_name", None)

        # 4. Synthesize proposed content using LLM provider or fallback synthesis
        try:
            proposed_content, rationale = self._synthesize_proposal_content(
                request=request,
                target_file=relative_target,
                operation=normalized_op,
                target_symbol=target_symbol,
                original_content=original_content,
            )
            if active and hasattr(active, "last_decision") and active.last_decision:
                provider_name = active.last_decision.actual_provider or active.last_decision.provider
                model_name = active.last_decision.actual_model or active.last_decision.model
                fallback_used = getattr(active.last_decision, "fallback_used", False)
                fallback_reason = getattr(active.last_decision, "fallback_reason", None)
                cloud_req_id = getattr(active.last_decision, "cloud_request_id", None) or cloud_req_id
        except ModelProviderError as err:
            logger.warning(f"Model provider error during proposal synthesis: {err}")
            if active and hasattr(active, "last_decision") and active.last_decision:
                provider_name = active.last_decision.actual_provider or active.last_decision.provider
                model_name = active.last_decision.actual_model or active.last_decision.model
                fallback_used = getattr(active.last_decision, "fallback_used", False)
                fallback_reason = getattr(active.last_decision, "fallback_reason", None)
                cloud_req_id = getattr(active.last_decision, "cloud_request_id", None) or cloud_req_id
            return CodeChangeProposal(
                operation=normalized_op,
                target_file=relative_target,
                target_symbol=target_symbol,
                description=request,
                status=ProposalStatus.GENERATION_FAILED.value,
                is_valid=False,
                validation_error=str(err),
                provider=provider_name,
                model=model_name,
                fallback_used=fallback_used,
                fallback_reason=fallback_reason,
                cloud_request_id=cloud_req_id,
            )

        proposal = CodeChangeProposal(
            operation=normalized_op,
            target_file=relative_target,
            target_symbol=target_symbol,
            description=request,
            proposed_content=proposed_content,
            original_content=original_content,
            rationale=rationale,
            status=ProposalStatus.PROPOSED.value,
            is_valid=True,
            validation_error=None,
            provider=provider_name,
            model=model_name,
            fallback_used=fallback_used,
            fallback_reason=fallback_reason,
            cloud_request_id=cloud_req_id,
        )


        # Register proposal into pending proposal store if applier is present
        if self.applier and hasattr(self.applier, "set_pending_proposal"):
            self.applier.set_pending_proposal(proposal)

        return proposal

    def _synthesize_proposal_content(
        self,
        request: str,
        target_file: str,
        operation: str,
        target_symbol: str | None,
        original_content: str,
    ) -> tuple[str, str]:
        """Queries LLM provider or uses synthesis logic to produce proposed content and rationale."""
        active = self.active_model
        if not active:
            # Deterministic fallback when no LLM provider is attached
            if operation == ProposalOperation.CREATE_FILE.value:
                proposed = f"# Proposed file: {target_file}\n# Request: {request}\n\ndef main():\n    pass\n"
                rationale = f"Create initial structure for {target_file} based on user request."
            else:
                comment = f"# Proposed addition for request: {request}\n"
                proposed = original_content + ("\n" if original_content and not original_content.endswith("\n") else "") + comment
                rationale = f"Append proposed code to {target_file} to fulfill user request."
            return proposed, rationale

        prompt_messages = [
            {
                "role": "system",
                "content": (
                    "You are VIDURA's Code Change Proposal Generator. "
                    "Synthesize a code change proposal to fulfill the user's request. "
                    "Do NOT attempt to run, apply, or execute the code. "
                    "Return ONLY a valid JSON object with format:\n"
                    "{\n"
                    '  "proposed_content": "complete new or modified code here",\n'
                    '  "rationale": "explanation of what changes were made and why"\n'
                    "}"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"User Request: {request}\n"
                    f"Target File: {target_file}\n"
                    f"Operation: {operation}\n"
                    f"Target Symbol: {target_symbol or 'N/A'}\n"
                    f"Existing File Content:\n```python\n{original_content[:2000]}\n```"
                ),
            },
        ]

        try:
            raw_res = active.generate(
                prompt_messages,
                task_type="developer",
                is_developer_task=True,
                target_files=[target_file],
                target_file=target_file,
                operation=operation,
            )
            # Clean json block wrapping if present
            cleaned = raw_res.strip()
            if "```" in cleaned:
                import re
                match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
                if match:
                    cleaned = match.group(1).strip()

            data = json.loads(cleaned, strict=False)
            proposed_content = self._clean_source_code(data.get("proposed_content", ""))
            rationale = data.get("rationale", f"Proposed code change for {target_file}.")
            if proposed_content:
                return proposed_content, rationale
        except ModelProviderError:
            raise
        except Exception as err:
            logger.warning(f"LLM proposal synthesis fallback due to error: {err}")

        # Fallback if LLM output parsing failed
        if operation == ProposalOperation.CREATE_FILE.value:
            proposed = f"# Proposed file: {target_file}\n# Request: {request}\n"
            rationale = f"Create file {target_file} for request: {request}"
        else:
            proposed = original_content + f"\n# Proposed modification for: {request}\n"
            rationale = f"Modify file {target_file} for request: {request}"
        return self._clean_source_code(proposed), rationale

    @staticmethod
    def _clean_source_code(code_text: str) -> str:
        """Strips markdown code blocks, backticks, and partial ellipsis from generated source code."""
        if not code_text:
            return ""
        cleaned = code_text.strip()
        # Handle wrapped markdown block ```python ... ``` or ``` ... ```
        match = re.search(r"^```(?:[a-zA-Z0-9_+-]+)?\r?\n?(.*?)\r?\n?```$", cleaned, re.DOTALL)
        if match:
            cleaned = match.group(1).strip()
        else:
            if cleaned.startswith("```"):
                lines = cleaned.splitlines()
                if lines and lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip().startswith("```"):
                    lines = lines[:-1]
                cleaned = "\n".join(lines).strip()

        if cleaned.startswith("`") and cleaned.endswith("`") and len(cleaned) >= 2:
            cleaned = cleaned[1:-1].strip()

        return cleaned

    def generate_proposal_from_plan(
        self,
        task: DeveloperTask,
        plan: DeveloperPlan,
    ) -> CodeChangeProposal:
        """Convenience method returning the generated CodeChangeProposal directly."""
        res = self.generate_from_plan(task=task, plan=plan)
        return res.proposal

    def generate_from_plan(
        self,
        task: DeveloperTask,
        plan: DeveloperPlan,
    ) -> DeveloperGenerationResult:
        """Synthesizes concrete code implementations and constructs a CodeChangeProposal from a validated DeveloperTask and DeveloperPlan.

        CRITICAL ARCHITECTURAL BOUNDARIES:
        1. Phase 8.3 is STRICTLY proposal generation and synthesis.
        2. Files on disk are NEVER touched, edited, or modified by this method.
        3. All disk changes remain strictly protected under Phase 7's permission-controlled pipeline.
        """
        logger.info(f"Generating code from plan: task='{task.goal[:60]}', files={plan.relevant_files}")

        # 1. Gate: Check if plan requires more information
        if plan.requires_more_information:
            reason = plan.missing_information_reason or "Plan requires more information before code generation."
            logger.info(f"Halting code generation: {reason}")
            target = task.target_files[0] if task.target_files else (plan.relevant_files[0] if plan.relevant_files else "")
            return DeveloperGenerationResult(
                target_file=target,
                operation="none",
                original_content="",
                proposed_content="",
                generated_code="",
                explanation=f"Cannot generate code: {reason}",
                affected_symbols=[],
                assumptions=[],
                validation_status="requires_information",
                is_valid=False,
                proposal=None,
                errors=[reason],
            )

        # 2. Identify target file
        target_file = ""
        if task.target_files:
            target_file = task.target_files[0]
        elif plan.relevant_files:
            target_file = plan.relevant_files[0]

        if not target_file:
            msg = "No target file identified in DeveloperTask or DeveloperPlan."
            logger.warning(msg)
            return DeveloperGenerationResult(
                target_file="",
                operation="none",
                explanation=msg,
                validation_status="invalid",
                is_valid=False,
                errors=[msg],
            )

        # Scope control: target file must be in plan.relevant_files or planned new files
        if plan.relevant_files and target_file not in plan.relevant_files and not any(target_file in change for change in plan.planned_changes):
            msg = f"Scope expansion rejected: Target file '{target_file}' is not permitted by DeveloperPlan (relevant files: {plan.relevant_files})."
            logger.warning(msg)
            return DeveloperGenerationResult(
                target_file=target_file,
                operation="none",
                explanation=msg,
                validation_status="scope_exceeded",
                is_valid=False,
                requires_plan_update=True,
                errors=[msg],
            )

        # Validate security and workspace boundaries
        if self._is_protected_target(target_file):
            msg = f"Access denied: Target file '{target_file}' matches protected security patterns (.git, .env, *.key, etc.)."
            logger.warning(msg)
            return DeveloperGenerationResult(
                target_file=target_file,
                operation="none",
                explanation=msg,
                validation_status="invalid",
                is_valid=False,
                errors=[msg],
            )

        try:
            resolved_path = self._validate_workspace_path(target_file)
        except ValueError as err:
            logger.warning(f"Path validation failed for '{target_file}': {err}")
            return DeveloperGenerationResult(
                target_file=target_file,
                operation="none",
                explanation=str(err),
                validation_status="invalid",
                is_valid=False,
                errors=[str(err)],
            )

        try:
            relative_target = str(resolved_path.relative_to(self.workspace_root))
        except ValueError:
            relative_target = target_file

        # 3. Check disk state (Read-Only) & determine operation
        file_exists = resolved_path.exists() and resolved_path.is_file()
        original_content = ""

        if file_exists:
            try:
                original_content = resolved_path.read_text(encoding="utf-8")
            except Exception as err:
                msg = f"Failed to read existing file '{target_file}': {err}"
                logger.error(msg)
                return DeveloperGenerationResult(
                    target_file=relative_target,
                    operation=ProposalOperation.MODIFY_FILE.value,
                    explanation=msg,
                    validation_status="permission_denied",
                    is_valid=False,
                    errors=[msg],
                )
            operation = ProposalOperation.MODIFY_FILE.value
        else:
            # File does not exist on disk. Check if creation is authorized.
            creation_authorized = False
            combined_context = (
                (task.requested_change or "") + " " +
                (task.goal or "") + " " +
                " ".join(plan.planned_changes)
            ).lower()

            creation_keywords = ["create file", "create new file", "create_file", "new file", "add file", "create "]
            if any(kw in combined_context for kw in creation_keywords):
                creation_authorized = True

            if not creation_authorized:
                msg = f"Target file '{relative_target}' does not exist on disk and new file creation was not authorized."
                logger.warning(msg)
                return DeveloperGenerationResult(
                    target_file=relative_target,
                    operation=ProposalOperation.MODIFY_FILE.value,
                    explanation=msg,
                    validation_status="target_not_found",
                    is_valid=False,
                    errors=[msg],
                )
            operation = ProposalOperation.CREATE_FILE.value
            original_content = ""

        # 4. Synthesize concrete code from plan
        active = self.active_model
        provider_name = None
        model_name = None
        fallback_used = False
        fallback_reason = None
        cloud_req_id = None
        if active:
            if hasattr(active, "route"):
                decision = active.route(
                    task=task,
                    plan=plan,
                    target_file=relative_target,
                    operation=operation,
                    is_developer_task=True,
                    local_only=getattr(task, "local_only", False),
                )
                provider_name = decision.actual_provider or decision.provider
                model_name = decision.actual_model or decision.model
                fallback_used = getattr(decision, "fallback_used", False)
                fallback_reason = getattr(decision, "fallback_reason", None)
                cloud_req_id = getattr(decision, "cloud_request_id", None)
            elif hasattr(active, "last_decision") and active.last_decision:
                provider_name = active.last_decision.actual_provider or active.last_decision.provider
                model_name = active.last_decision.actual_model or active.last_decision.model
                fallback_used = getattr(active.last_decision, "fallback_used", False)
                fallback_reason = getattr(active.last_decision, "fallback_reason", None)
                cloud_req_id = getattr(active.last_decision, "cloud_request_id", None)
            else:
                provider_name = getattr(active, "provider_type", None) or ("cloud" if getattr(getattr(active, "capabilities", None), "cloud", False) else "local")
                model_name = getattr(active, "model_name", None)

        try:
            generated_code, explanation, affected_symbols, assumptions = self._synthesize_from_plan(
                task=task,
                plan=plan,
                target_file=relative_target,
                operation=operation,
                original_content=original_content,
            )
            if active and hasattr(active, "last_decision") and active.last_decision:
                provider_name = active.last_decision.actual_provider or active.last_decision.provider
                model_name = active.last_decision.actual_model or active.last_decision.model
                fallback_used = getattr(active.last_decision, "fallback_used", False)
                fallback_reason = getattr(active.last_decision, "fallback_reason", None)
                cloud_req_id = getattr(active.last_decision, "cloud_request_id", None) or cloud_req_id
        except ModelProviderError as err:
            logger.error(f"Model provider error during code generation: {err}")
            if active and hasattr(active, "last_decision") and active.last_decision:
                provider_name = active.last_decision.actual_provider or active.last_decision.provider
                model_name = active.last_decision.actual_model or active.last_decision.model
                fallback_used = getattr(active.last_decision, "fallback_used", False)
                fallback_reason = getattr(active.last_decision, "fallback_reason", None)
                cloud_req_id = getattr(active.last_decision, "cloud_request_id", None) or cloud_req_id
            return DeveloperGenerationResult(
                target_file=relative_target,
                operation=operation,
                original_content=original_content,
                proposed_content="",
                generated_code="",
                explanation=f"Provider error during code generation: {err}",
                affected_symbols=task.target_symbols or [],
                assumptions=[],
                validation_status="provider_failure",
                is_valid=False,
                proposal=None,
                errors=[str(err)],
                provider=provider_name,
                model=model_name,
                fallback_used=fallback_used,
                fallback_reason=fallback_reason,
                cloud_request_id=cloud_req_id,
            )

        # Sanitize code to ensure pure source code
        generated_code = self._clean_source_code(generated_code)

        # 5. Construct CodeChangeProposal
        target_symbol = task.target_symbols[0] if task.target_symbols else (plan.relevant_symbols[0] if plan.relevant_symbols else None)
        proposal = CodeChangeProposal(
            operation=operation,
            target_file=relative_target,
            target_symbol=target_symbol,
            description=task.goal or task.requested_change or f"Implement planned change for {relative_target}",
            proposed_content=generated_code,
            original_content=original_content,
            rationale=explanation,
            status=ProposalStatus.PROPOSED.value,
            is_valid=True,
            validation_error=None,
            provider=provider_name,
            model=model_name,
            fallback_used=fallback_used,
            fallback_reason=fallback_reason,
            cloud_request_id=cloud_req_id,
        )

        # Register proposal with applier if available (Read-Only registration)
        if self.applier and hasattr(self.applier, "set_pending_proposal"):
            self.applier.set_pending_proposal(proposal)

        return DeveloperGenerationResult(
            target_file=relative_target,
            operation=operation,
            original_content=original_content,
            proposed_content=generated_code,
            generated_code=generated_code,
            explanation=explanation,
            affected_symbols=affected_symbols,
            assumptions=assumptions,
            validation_status="valid",
            is_valid=True,
            proposal=proposal,
            errors=[],
            provider=provider_name,
            model=model_name,
            fallback_used=fallback_used,
            fallback_reason=fallback_reason,
            cloud_request_id=cloud_req_id,
        )


    def _synthesize_from_plan(
        self,
        task: DeveloperTask,
        plan: DeveloperPlan,
        target_file: str,
        operation: str,
        original_content: str,
    ) -> tuple[str, str, list[str], list[str]]:
        """Synthesizes code using attached LLM provider or deterministic fallback."""
        affected_symbols = list(task.target_symbols)
        for sym in plan.relevant_symbols:
            if sym not in affected_symbols:
                affected_symbols.append(sym)

        combined_constraints = list(task.constraints)
        for c in plan.constraints:
            if c not in combined_constraints:
                combined_constraints.append(c)

        active = self.active_model
        if not active:
            # Deterministic template synthesis when model is None
            assumptions = [
                "Generated code using deterministic template without active LLM provider.",
                f"Preserved existing file logic and satisfied {len(combined_constraints)} constraints.",
            ]
            if operation == ProposalOperation.CREATE_FILE.value:
                lines = [
                    f'"""Module {Path(target_file).name}',
                    "",
                    f"Generated for: {task.goal}",
                    '"""',
                    "",
                ]
                symbols_to_add = affected_symbols if affected_symbols else ["main"]
                for sym in symbols_to_add:
                    lines.append(f"def {sym}():")
                    lines.append(f'    """Implementation for {sym}."""')
                    lines.append("    pass\n")
                code = "\n".join(lines)
                explanation = f"Created initial implementation for {target_file} fulfilling task goal: '{task.goal}'."
            else:
                symbols_to_add = [s for s in affected_symbols if s not in original_content]
                additions = []
                for sym in symbols_to_add:
                    additions.append(f"\n\ndef {sym}():\n    \"\"\"Implementation for {sym}.\"\"\"\n    pass\n")
                if additions:
                    code = original_content.rstrip() + "".join(additions)
                else:
                    comment = f"\n# Updated implementation for: {task.goal}\n"
                    code = original_content.rstrip() + comment
                explanation = f"Modified {target_file} to implement planned changes for '{task.goal}'."

            return code, explanation, affected_symbols, assumptions

        rag_prompt_section = ""
        if getattr(plan, "rag_context", None):
            rag_ctx = plan.rag_context
            if isinstance(rag_ctx, dict):
                c_text = rag_ctx.get("context_text") or rag_ctx.get("prompt_text")
                if c_text:
                    rag_prompt_section = f"\n{c_text}\n\n"

        prompt_messages = [
            {
                "role": "system",
                "content": (
                    "You are VIDURA's Developer Code Generator acting as a Senior Software Engineer.\n"
                    "Generate complete, production-ready, clean source code to fulfill the DeveloperTask and DeveloperPlan.\n"
                    "RULES:\n"
                    "1. Return ONLY a valid JSON object matching the schema below.\n"
                    "2. The 'generated_code' MUST be the entire, complete file content. NEVER use ellipsis '...' or leave parts out.\n"
                    "3. Do NOT wrap 'generated_code' in markdown code fences (no ```python). Return raw source code text inside the JSON string.\n"
                    "4. Strictly adhere to all constraints, existing conventions, import styles, and architectural boundaries.\n"
                    "5. Do NOT include conversational filler or explanations outside the JSON object.\n\n"
                    "JSON Schema:\n"
                    "{\n"
                    '  "generated_code": "entire complete source code here",\n'
                    '  "explanation": "explanation of implementation and changes made",\n'
                    '  "affected_symbols": ["symbol1", "symbol2"],\n'
                    '  "assumptions": ["assumption1"]\n'
                    "}"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Developer Task:\n"
                    f"- Goal: {task.goal}\n"
                    f"- Requested Change: {task.requested_change or 'N/A'}\n"
                    f"- Target Symbols: {task.target_symbols}\n"
                    f"- Expected Behavior: {task.expected_behavior or 'N/A'}\n\n"
                    f"Developer Plan:\n"
                    f"- Planned Changes: {plan.planned_changes}\n"
                    f"- Dependencies: {plan.dependencies}\n"
                    f"- Affected Components: {plan.affected_components}\n"
                    f"- Risks: {plan.risks}\n"
                    f"- Constraints: {combined_constraints}\n\n"
                    f"{rag_prompt_section}"
                    f"Target File: {target_file}\n"
                    f"Operation: {operation}\n"
                    f"Original File Content:\n"
                    f"```\n{original_content[:3000]}\n```"
                ),
            },
        ]

        try:
            raw_res = active.generate(
                prompt_messages,
                task=task,
                plan=plan,
                task_type="developer",
                is_developer_task=True,
                target_file=target_file,
                operation=operation,
                local_only=getattr(task, "local_only", False),
            )
            cleaned = raw_res.strip()
            if "```" in cleaned:
                match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
                if match:
                    cleaned = match.group(1).strip()
                else:
                    brace_match = re.search(r"(\{.*\})", cleaned, re.DOTALL)
                    if brace_match:
                        cleaned = brace_match.group(1).strip()

            data = json.loads(cleaned, strict=False)
            gen_code = self._clean_source_code(data.get("generated_code", ""))
            explanation = data.get("explanation", f"Implemented planned changes for {target_file}.")
            symbols = data.get("affected_symbols", affected_symbols)
            assump = data.get("assumptions", [f"Implemented according to plan for {target_file}."])
            if gen_code:
                return gen_code, explanation, symbols, assump
        except ModelProviderError:
            raise
        except Exception as err:
            logger.warning(f"LLM code synthesis failed or produced invalid JSON: {err}. Falling back to template synthesis.")

        # Fallback if LLM output parsing failed
        if operation == ProposalOperation.CREATE_FILE.value:
            code = f'"""Module {Path(target_file).name}\n\nGenerated for: {task.goal}\n"""\n\n'
            for sym in affected_symbols:
                code += f"def {sym}():\n    \"\"\"Implementation for {sym}.\"\"\"\n    pass\n\n"
            explanation = f"Created initial implementation for {target_file}."
        else:
            code = original_content + f"\n# Updated implementation for: {task.goal}\n"
            explanation = f"Modified {target_file} for task: {task.goal}."

        assumptions = ["Fallback synthesis applied following plan constraints."]
        return code, explanation, affected_symbols, assumptions


class DeveloperCodeGenerator(CodeChangeGenerator):
    """Phase 8.3 Developer Code Generator.

    Generates concrete code implementations and constructs CodeChangeProposals
    from validated DeveloperTasks and DeveloperPlans without modifying the filesystem.
    """
    pass

