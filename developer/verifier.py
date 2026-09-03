import os
import logging
from pathlib import Path
from typing import Any
from config import load_config
from developer.models import CodeChangeProposal, VerificationResult

logger = logging.getLogger("VIDURA.developer.verifier")


class CodeChangeVerifier:
    """Verifies actual filesystem state against trusted CodeChangeProposal specifications."""

    def __init__(self, workspace_root: str | Path | None = None) -> None:
        cfg = load_config()
        self.workspace_root = Path(workspace_root).resolve() if workspace_root else cfg.workspace_root.resolve()

    def _validate_workspace_path(self, target_path: str | Path) -> Path:
        """Resolves target_path and verifies it lies strictly within self.workspace_root.
        
        Performs realpath symlink resolution to prevent symlink traversal escapes.
        """
        root = self.workspace_root.resolve()
        root_real = Path(os.path.realpath(root))
        path_obj = Path(target_path)
        
        if path_obj.is_absolute():
            candidate = path_obj
        else:
            candidate = root / path_obj

        try:
            resolved = candidate.resolve()
            real_target = Path(os.path.realpath(resolved))
        except Exception as err:
            raise ValueError(f"Access denied: Failed to resolve path '{target_path}': {err}")

        if not resolved.is_relative_to(root) or not real_target.is_relative_to(root_real):
            raise ValueError(
                f"Access denied: Requested path '{target_path}' (resolved: '{resolved}', real: '{real_target}') "
                f"is outside the allowed workspace boundary '{root}'."
            )
            
        return resolved

    def verify(self, proposal: CodeChangeProposal) -> VerificationResult:
        """Verifies actual physical disk state against proposal specifications.
        
        CRITICAL VERIFICATION RULES:
        1. Authoritative Filesystem Inspection: Reads actual disk file.
        2. Exact Content Comparison: Compares disk content against proposal.proposed_content.
        3. Fail-Closed Safety: Returns success=False on any mismatch, missing file, or exception.
        """
        logger.info(f"Verifying filesystem state for target: '{proposal.target_file}', op: '{proposal.operation}'")

        try:
            if not proposal or not isinstance(proposal, CodeChangeProposal):
                return VerificationResult(
                    success=False,
                    target_file=str(proposal.target_file if proposal else ""),
                    operation=str(proposal.operation if proposal else ""),
                    verified=False,
                    expected_state="Valid proposal object",
                    actual_state="Invalid or missing proposal",
                    reason="Verification failed: Invalid proposal object.",
                    error="Invalid proposal.",
                )

            try:
                resolved_path = self._validate_workspace_path(proposal.target_file)
            except ValueError as err:
                logger.warning(f"Verification path validation failed for '{proposal.target_file}': {err}")
                return VerificationResult(
                    success=False,
                    target_file=proposal.target_file,
                    operation=proposal.operation,
                    verified=False,
                    expected_state="Valid workspace path",
                    actual_state="Invalid or out-of-boundary path",
                    reason=str(err),
                    error=str(err),
                )

            if not resolved_path.exists() or not resolved_path.is_file():
                logger.warning(f"Verification failed: File '{proposal.target_file}' does not exist on disk.")
                return VerificationResult(
                    success=False,
                    target_file=proposal.target_file,
                    operation=proposal.operation,
                    verified=False,
                    expected_state="Target file exists on disk",
                    actual_state="File missing",
                    reason=f"Verification failed: Target file '{proposal.target_file}' does not exist on disk.",
                    error="File missing.",
                )

            try:
                actual_content = resolved_path.read_text(encoding="utf-8")
            except Exception as err:
                logger.error(f"Failed to read disk content for verification of '{proposal.target_file}': {err}")
                return VerificationResult(
                    success=False,
                    target_file=proposal.target_file,
                    operation=proposal.operation,
                    verified=False,
                    expected_state="Readable disk file",
                    actual_state=f"Read error: {err}",
                    reason=f"Verification failed: Failed to read file content from disk: {err}",
                    error=str(err),
                )

            if actual_content != proposal.proposed_content:
                logger.warning(f"Verification failed for '{proposal.target_file}': Disk content differs from proposed content.")
                return VerificationResult(
                    success=False,
                    target_file=proposal.target_file,
                    operation=proposal.operation,
                    verified=False,
                    expected_state=f"Proposed content ({len(proposal.proposed_content)} bytes)",
                    actual_state=f"Actual disk content ({len(actual_content)} bytes)",
                    reason=f"Verification failed: Disk content of '{proposal.target_file}' does not match expected proposed content.",
                    error="Content mismatch.",
                )

            logger.info(f"Verification SUCCEEDED for '{proposal.target_file}': Disk content matches proposed content byte-for-byte.")
            return VerificationResult(
                success=True,
                target_file=proposal.target_file,
                operation=proposal.operation,
                verified=True,
                expected_state=f"Proposed content ({len(proposal.proposed_content)} bytes)",
                actual_state=f"Disk content verified ({len(actual_content)} bytes)",
                reason=f"Post-change verification succeeded: '{proposal.target_file}' content matches proposed content byte-for-byte.",
                error=None,
            )

        except Exception as err:
            logger.error(f"Unexpected exception during verification: {err}", exc_info=True)
            target_str = proposal.target_file if proposal else ""
            op_str = proposal.operation if proposal else ""
            return VerificationResult(
                success=False,
                target_file=target_str,
                operation=op_str,
                verified=False,
                expected_state="Successful verification",
                actual_state=f"Exception: {err}",
                reason=f"Verification failed due to unexpected error: {err}",
                error=str(err),
            )
