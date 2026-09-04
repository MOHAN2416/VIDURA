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
        allowed_args = {"request", "target_file", "operation", "target_symbol"}
        unexpected = set(kwargs.keys()) - allowed_args
        if unexpected:
            return error_result(self.name, f"Unexpected argument(s) provided: {sorted(unexpected)}")

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

    def __init__(self, applier: CodeChangeApplier, executor: Any = None) -> None:
        self.applier = applier
        from developer.executor import DeveloperExecutor
        self.executor = executor or DeveloperExecutor(applier=applier)

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
        allowed_args = {"confirm_permission", "target_file", "proposal"}
        unexpected = set(kwargs.keys()) - allowed_args
        if unexpected:
            return error_result(self.name, f"Unexpected argument(s) provided: {sorted(unexpected)}")

        confirm_permission = bool(kwargs.get("confirm_permission", False))

        try:
            pending_proposal = self.applier.get_pending_proposal()

            if not pending_proposal:
                return error_result(self.name, "No active pending code change proposal found. Generate a proposal first.")

            exec_res = self.executor.execute_proposal(explicit_permission=confirm_permission)
            if not exec_res.success:
                err_msg = exec_res.error or exec_res.summary or "Application failed."
                if exec_res.application_result and exec_res.application_result.reason:
                    err_msg = exec_res.application_result.reason
                return error_result(self.name, err_msg)

            data = exec_res.application_result.to_dict() if exec_res.application_result else exec_res.to_dict()
            data["execution_result"] = exec_res.to_dict()
            return success_result(self.name, data=data)
        except Exception as err:
            return error_result(self.name, f"Error applying code change: {err}")


class RunTestsTool(BaseTool):
    """Tool to execute controlled project tests strictly through the secure test runner."""

    def __init__(self, test_runner: Any = None, permission_manager: Any = None, workspace_root: Any = None) -> None:
        if test_runner is None:
            from developer.testing import TestRunner
            self.test_runner = TestRunner(workspace_root=workspace_root, permission_manager=permission_manager)
        else:
            self.test_runner = test_runner
        self.permission_manager = permission_manager

    @property
    def name(self) -> str:
        return "run_tests"

    @property
    def description(self) -> str:
        return (
            "Runs project tests on a validated target file inside the workspace using the controlled test runner. "
            "Arbitrary commands and external paths are strictly prohibited."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "description": "The test file path (e.g. 'tests/test_agent.py') or node identifier to execute.",
                },
            },
            "required": ["target"],
        }

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        allowed_args = {"target", "extra_args"}
        unexpected = set(kwargs.keys()) - allowed_args
        if unexpected:
            return error_result(self.name, f"Unexpected argument(s) provided: {sorted(unexpected)}")

        target = str(kwargs.get("target", "")).strip()
        extra_args = kwargs.get("extra_args")
        if not target:
            return error_result(self.name, "Missing required parameter 'target'.")

        try:
            res = self.test_runner.run_test(target=target, extra_args=extra_args)
            if not res.passed:
                return error_result(self.name, res.error or f"Tests failed with exit code {res.exit_code}", data=res.to_dict())
            return success_result(self.name, data=res.to_dict())
        except Exception as err:
            return error_result(self.name, f"Test execution failed: {err}")
