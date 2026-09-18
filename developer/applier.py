import os
import logging
from pathlib import Path
from config import load_config
from permissions.manager import PermissionManager
from developer.models import (
    CodeChangeProposal,
    CodeChangeResult,
    ProposalOperation,
    ApplicationStatus,
)
from developer.verifier import CodeChangeVerifier

logger = logging.getLogger("VIDURA.developer.applier")

PROTECTED_PATTERNS = [".git", ".env", ".key", ".pem", ".secret"]
UNSUPPORTED_OPERATIONS = [
    "delete", "delete_file", "remove", "destroy", "unlink",
    "rename", "rename_file", "move", "move_file", "copy", "copy_file",
    "chmod", "chown", "execute", "shell", "command", "run", "exec"
]


class CodeChangeApplier:
    """Applies validated CodeChangeProposal instances to the filesystem strictly when explicit permission is granted."""

    def __init__(
        self,
        permission_manager: PermissionManager | None = None,
        workspace_root: str | Path | None = None,
        verifier: CodeChangeVerifier | None = None,
    ) -> None:
        cfg = load_config()
        self.permission_manager = permission_manager
        self.workspace_root = Path(workspace_root).resolve() if workspace_root else cfg.workspace_root.resolve()
        self.verifier = verifier or CodeChangeVerifier(workspace_root=self.workspace_root)
        self._pending_proposal: CodeChangeProposal | None = None

    def set_pending_proposal(self, proposal: CodeChangeProposal) -> None:
        """Stores a complete, trusted CodeChangeProposal as the active pending proposal.
        
        Auto-revokes write permission to enforce single-proposal authorization scope.
        """
        if proposal and isinstance(proposal, CodeChangeProposal) and proposal.is_valid:
            proposal.status = "pending"
            self._pending_proposal = proposal
            if self.permission_manager:
                self.permission_manager.revoke_write_permission()
            logger.info(f"Registered pending proposal: target='{proposal.target_file}', op='{proposal.operation}'")

    def get_pending_proposal(self) -> CodeChangeProposal | None:
        """Returns the currently active pending proposal, if any."""
        return self._pending_proposal

    def clear_pending_proposal(self) -> None:
        """Clears/invalidates the currently active pending proposal and revokes write permission."""
        if self._pending_proposal:
            self._pending_proposal.status = "invalidated"
            self._pending_proposal = None
            logger.info("Cleared active pending proposal.")
        if self.permission_manager:
            self.permission_manager.revoke_write_permission()

    def _is_protected_target(self, target_path: str | Path) -> bool:
        """Checks if target path matches protected security patterns (.git, .env, *.key, *.pem, *.secret)."""
        if not target_path:
            return True
        path_str = str(target_path).lower().strip()
        parts = Path(target_path).parts
        for part in parts:
            part_lower = part.lower()
            if part_lower.startswith(".git") or part_lower == ".env":
                return True
        for pattern in PROTECTED_PATTERNS:
            if path_str.endswith(pattern) or f"/{pattern}" in path_str or pattern in parts:
                return True
        return False

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

    def apply_proposal(
        self,
        proposal: CodeChangeProposal | None = None,
        explicit_permission: bool = False,
    ) -> CodeChangeResult:
        """Applies a CodeChangeProposal on disk strictly when explicit permission is granted."""
        try:
            # Determine trusted proposal object
            trusted_proposal = self._pending_proposal

            # If caller passed a proposal object and no pending proposal exists, fall back if valid
            if not trusted_proposal:
                if proposal and isinstance(proposal, CodeChangeProposal) and proposal.is_valid:
                    trusted_proposal = proposal
                else:
                    target_str = proposal.target_file if proposal else ""
                    op_str = proposal.operation if proposal else ""
                    return CodeChangeResult(
                        success=False,
                        target_file=target_str,
                        operation=op_str,
                        permission_granted=False,
                        reason="No active pending code change proposal found. Generate a proposal first.",
                        verification_success=False,
                        status_code=ApplicationStatus.APPLICATION_FAILED.value,
                        error="No pending proposal.",
                    )

            # Check if proposal status is applied, denied, or invalidated
            if trusted_proposal.status in ("applied", "denied", "invalidated"):
                return CodeChangeResult(
                    success=False,
                    target_file=trusted_proposal.target_file,
                    operation=trusted_proposal.operation,
                    permission_granted=False,
                    reason=f"Proposal has already been {trusted_proposal.status}.",
                    verification_success=False,
                    status_code=ApplicationStatus.APPLICATION_FAILED.value if trusted_proposal.status == "applied" else ApplicationStatus.PERMISSION_DENIED.value,
                    error=f"Proposal {trusted_proposal.status}.",
                )

            logger.info(f"Attempting to apply trusted proposal: op='{trusted_proposal.operation}', target='{trusted_proposal.target_file}'")

            # 1. PERMISSION CHECK: Default NO PERMISSION = NO MODIFICATION
            if self.permission_manager:
                permission_granted = self.permission_manager.is_write_allowed(trusted_proposal.target_file, trusted_proposal.operation)
            else:
                permission_granted = explicit_permission

            if not permission_granted:
                logger.warning(f"Permission DENIED for modifying '{trusted_proposal.target_file}'.")
                return CodeChangeResult(
                    success=False,
                    target_file=trusted_proposal.target_file,
                    operation=trusted_proposal.operation,
                    permission_granted=False,
                    reason="Permission denied: Explicit authorization is required to modify or create files on disk.",
                    verification_success=False,
                    status_code=ApplicationStatus.PERMISSION_DENIED.value,
                    error="Permission denied.",
                )

            # 2. PROPOSAL VALIDITY CHECK
            if not trusted_proposal.is_valid:
                logger.warning(f"Cannot apply invalid proposal for '{trusted_proposal.target_file}': {trusted_proposal.validation_error}")
                return CodeChangeResult(
                    success=False,
                    target_file=trusted_proposal.target_file,
                    operation=trusted_proposal.operation,
                    permission_granted=True,
                    reason=trusted_proposal.validation_error or "Invalid proposal.",
                    verification_success=False,
                    status_code=ApplicationStatus.APPLICATION_FAILED.value,
                    error=trusted_proposal.validation_error or "Invalid proposal.",
                )

            # 3. WORKSPACE BOUNDARY AND PROTECTED FILE CHECK
            if self._is_protected_target(trusted_proposal.target_file):
                logger.warning(f"Access denied for protected file '{trusted_proposal.target_file}'.")
                return CodeChangeResult(
                    success=False,
                    target_file=trusted_proposal.target_file,
                    operation=trusted_proposal.operation,
                    permission_granted=True,
                    reason=f"Access denied: Target path '{trusted_proposal.target_file}' is protected (.git, .env, *.key, *.pem, *.secret).",
                    verification_success=False,
                    status_code=ApplicationStatus.PERMISSION_DENIED.value,
                    error="Access denied: Protected file.",
                )

            try:
                resolved_path = self._validate_workspace_path(trusted_proposal.target_file)
            except ValueError as err:
                logger.warning(f"Path validation failed for application: {err}")
                return CodeChangeResult(
                    success=False,
                    target_file=trusted_proposal.target_file,
                    operation=trusted_proposal.operation,
                    permission_granted=True,
                    reason=str(err),
                    verification_success=False,
                    status_code=ApplicationStatus.APPLICATION_FAILED.value,
                    error=str(err),
                )

            # 4. SUPPORTED OPERATION CHECK
            op = trusted_proposal.operation.lower().strip() if trusted_proposal.operation else ""
            if op in UNSUPPORTED_OPERATIONS or op not in (ProposalOperation.CREATE_FILE.value, ProposalOperation.MODIFY_FILE.value):
                logger.warning(f"Rejecting unsupported operation '{trusted_proposal.operation}' for '{trusted_proposal.target_file}'.")
                return CodeChangeResult(
                    success=False,
                    target_file=trusted_proposal.target_file,
                    operation=trusted_proposal.operation,
                    permission_granted=True,
                    reason=f"Operation '{trusted_proposal.operation}' is strictly unsupported. Only 'create_file' and 'modify_file' are supported.",
                    verification_success=False,
                    status_code=ApplicationStatus.APPLICATION_FAILED.value,
                    error="Unsupported operation.",
                )

            # 5. ATOMICITY & STALE PROPOSAL CHECK
            if op == ProposalOperation.MODIFY_FILE.value:
                if not resolved_path.exists() or not resolved_path.is_file():
                    return CodeChangeResult(
                        success=False,
                        target_file=trusted_proposal.target_file,
                        operation=op,
                        permission_granted=True,
                        reason=f"Target file '{trusted_proposal.target_file}' no longer exists on disk.",
                        verification_success=False,
                        status_code=ApplicationStatus.APPLICATION_FAILED.value,
                        error="Target file missing.",
                    )

                try:
                    current_disk_content = resolved_path.read_text(encoding="utf-8")
                except Exception as err:
                    return CodeChangeResult(
                        success=False,
                        target_file=trusted_proposal.target_file,
                        operation=op,
                        permission_granted=True,
                        reason=f"Failed to read existing file: {err}",
                        verification_success=False,
                        status_code=ApplicationStatus.APPLICATION_FAILED.value,
                        error=str(err),
                    )

                if current_disk_content != trusted_proposal.original_content:
                    logger.warning(f"Stale proposal detected for '{trusted_proposal.target_file}'. File changed on disk.")
                    return CodeChangeResult(
                        success=False,
                        target_file=trusted_proposal.target_file,
                        operation=op,
                        permission_granted=True,
                        reason="Stale proposal rejected: Target file has been modified on disk since proposal generation.",
                        verification_success=False,
                        status_code=ApplicationStatus.STALE_PROPOSAL.value,
                        error="Stale proposal.",
                    )
            elif op == ProposalOperation.CREATE_FILE.value:
                if resolved_path.exists() and resolved_path.is_file():
                    try:
                        existing_content = resolved_path.read_text(encoding="utf-8")
                        if existing_content and existing_content != trusted_proposal.proposed_content and existing_content != trusted_proposal.original_content:
                            logger.warning(f"Create file target '{trusted_proposal.target_file}' already exists with different content.")
                    except Exception:
                        pass

            # 6. SAFE FILE WRITING
            try:
                resolved_path.parent.mkdir(parents=True, exist_ok=True)
                resolved_path.write_text(trusted_proposal.proposed_content, encoding="utf-8")
            except Exception as err:
                logger.error(f"Filesystem write failed for '{trusted_proposal.target_file}': {err}")
                return CodeChangeResult(
                    success=False,
                    target_file=trusted_proposal.target_file,
                    operation=op,
                    permission_granted=True,
                    reason=f"Filesystem write failed: {err}",
                    verification_success=False,
                    status_code=ApplicationStatus.APPLICATION_FAILED.value,
                    error=str(err),
                )

            # 7. POST-CHANGE VERIFICATION VIA CodeChangeVerifier
            ver_res = self.verifier.verify(trusted_proposal)
            if not ver_res.success:
                logger.error(f"Post-change verification failed for '{trusted_proposal.target_file}': {ver_res.reason}")
                return CodeChangeResult(
                    success=False,
                    target_file=trusted_proposal.target_file,
                    operation=op,
                    permission_granted=True,
                    reason=ver_res.reason,
                    verification_success=False,
                    status_code=ApplicationStatus.VERIFICATION_FAILED.value,
                    error=ver_res.error or "Verification failed.",
                    verification_details=ver_res.to_dict(),
                )

            # Mark proposal as applied and clear pending proposal
            trusted_proposal.status = "applied"
            self._pending_proposal = None

            logger.info(f"Successfully applied and verified {op} for '{trusted_proposal.target_file}'.")
            return CodeChangeResult(
                success=True,
                target_file=trusted_proposal.target_file,
                operation=op,
                permission_granted=True,
                reason=f"Successfully applied {op} to '{trusted_proposal.target_file}'. File verified on disk.",
                verification_success=True,
                status_code=ApplicationStatus.APPLIED_AND_VERIFIED.value,
                error=None,
                verification_details=ver_res.to_dict(),
            )

        except Exception as err:
            logger.error(f"Unexpected exception during apply_proposal: {err}", exc_info=True)
            target_str = proposal.target_file if proposal else ""
            op_str = proposal.operation if proposal else ""
            return CodeChangeResult(
                success=False,
                target_file=target_str,
                operation=op_str,
                permission_granted=False,
                reason=f"Unexpected error during application: {err}",
                verification_success=False,
                status_code=ApplicationStatus.APPLICATION_FAILED.value,
                error=str(err),
            )

    # Alias for apply_proposal
    apply = apply_proposal
