import json
import logging
from pathlib import Path
from typing import Any
from config import load_config
from models.base import BaseLLMProvider
from codebase.manager import CodebaseManager
from developer.models import CodeChangeProposal, ProposalOperation, ProposalStatus

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

        # 4. Synthesize proposed content using LLM provider or fallback synthesis
        proposed_content, rationale = self._synthesize_proposal_content(
            request=request,
            target_file=relative_target,
            operation=normalized_op,
            target_symbol=target_symbol,
            original_content=original_content,
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
        if not self.model:
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
            raw_res = self.model.generate(prompt_messages)
            # Clean json block wrapping if present
            cleaned = raw_res.strip()
            if "```" in cleaned:
                import re
                match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
                if match:
                    cleaned = match.group(1).strip()

            data = json.loads(cleaned, strict=False)
            proposed_content = data.get("proposed_content", "")
            rationale = data.get("rationale", f"Proposed code change for {target_file}.")
            if proposed_content:
                return proposed_content, rationale
        except Exception as err:
            logger.warning(f"LLM proposal synthesis fallback due to error: {err}")

        # Fallback if LLM output parsing failed
        if operation == ProposalOperation.CREATE_FILE.value:
            proposed = f"# Proposed file: {target_file}\n# Request: {request}\n"
            rationale = f"Create file {target_file} for request: {request}"
        else:
            proposed = original_content + f"\n# Proposed modification for: {request}\n"
            rationale = f"Modify file {target_file} for request: {request}"
        return proposed, rationale
