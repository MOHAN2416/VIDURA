from typing import Any
from tools.base import BaseTool, success_result, error_result
from developer.generator import CodeChangeGenerator
from developer.applier import CodeChangeApplier, UNSUPPORTED_OPERATIONS
from developer.models import CodeChangeProposal


class ProposeCodeChangeTool(BaseTool):
    """Tool to generate structured code change proposals without writing or modifying files on disk."""

    def __init__(self, generator: CodeChangeGenerator) -> None:
        self.generator = generator

    @property
    def name(self) -> str:
        return "propose_code_change"

    @property
    def description(self) -> str:
        return (
            "Proposes a structured code change ('create_file' or 'modify_file') without modifying any file on disk. "
            "Use when asked to create a function, add a class, modify a module, or propose code changes."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "request": {
                    "type": "string",
                    "description": "The requested code change (e.g. 'Add greet_user function to utils.py').",
                },
                "target_file": {
                    "type": "string",
                    "description": "Target relative file path (e.g. 'utils.py' or 'agent/loop.py').",
                },
                "operation": {
                    "type": "string",
                    "enum": ["create_file", "modify_file"],
                    "default": "modify_file",
                    "description": "Operation type: 'create_file' or 'modify_file'.",
                },
                "target_symbol": {
                    "type": "string",
                    "description": "Optional class or function symbol name.",
                },
            },
            "required": ["request", "target_file"],
        }

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        request = str(kwargs.get("request", "")).strip()
        target_file = str(kwargs.get("target_file", "")).strip()
        operation = str(kwargs.get("operation", "modify_file")).strip()
        target_symbol = kwargs.get("target_symbol")

        if not request:
            return error_result(self.name, "Parameter 'request' is required.")
        if not target_file:
            return error_result(self.name, "Parameter 'target_file' is required.")

        op_lower = operation.lower()
        if op_lower in UNSUPPORTED_OPERATIONS or op_lower not in ("create_file", "modify_file"):
            return error_result(
                self.name,
                f"Operation '{operation}' is strictly unsupported in VIDURA. Only 'create_file' and 'modify_file' are supported."
            )

        try:
            proposal = self.generator.generate_proposal(
                request=request,
                target_file=target_file,
                operation=operation,
                target_symbol=target_symbol,
            )
            if not proposal.is_valid:
                return error_result(self.name, proposal.validation_error or "Invalid code change proposal request.")

            return success_result(self.name, data=proposal.to_dict())
        except Exception as err:
            return error_result(self.name, f"Error generating code change proposal: {err}")


class ApplyCodeChangeTool(BaseTool):
    """Tool to apply a validated code change proposal on disk when explicit permission is granted."""

    def __init__(self, applier: CodeChangeApplier) -> None:
        self.applier = applier

    @property
    def name(self) -> str:
        return "apply_code_change"

    @property
    def description(self) -> str:
        return (
            "Applies a pending validated code change proposal to disk strictly when explicit write permission has been granted."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "confirm_permission": {
                    "type": "boolean",
                    "default": False,
                    "description": "Explicit permission confirmation flag.",
                },
                "target_file": {
                    "type": "string",
                    "description": "Optional target file name of the pending proposal.",
                },
            },
            "required": [],
        }

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        confirm_permission = bool(kwargs.get("confirm_permission", False))

        try:
            # TOOL ARGUMENT HARDENING: Strictly rely on applier's active pending proposal.
            # Do NOT allow LLM-supplied kwargs to inject or override proposal data.
            pending_proposal = self.applier.get_pending_proposal()

            if not pending_proposal:
                return error_result(self.name, "No active pending code change proposal found. Generate a proposal first.")

            result = self.applier.apply_proposal(proposal=pending_proposal, explicit_permission=confirm_permission)
            if not result.success:
                return error_result(self.name, result.reason)

            return success_result(self.name, data=result.to_dict())
        except Exception as err:
            return error_result(self.name, f"Error applying code change: {err}")
