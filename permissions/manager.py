import logging

logger = logging.getLogger("VIDURA.permissions.manager")


class PermissionManager:
    """Security boundary managing file write permissions and execution approvals.
    
    CRITICAL SECURITY MANDATE:
    - Default state is strictly NO WRITE PERMISSION (write_allowed = False).
    - Model outputs, proposal validity, or prompt text can NEVER grant permission automatically.
    - Write permission must be explicitly granted by application controller or user authorization (/approve).
    """

    def __init__(self, default_write_allowed: bool = False) -> None:
        self._write_allowed = bool(default_write_allowed)

    @property
    def write_allowed(self) -> bool:
        return self._write_allowed

    def is_write_allowed(self, target_file: str | None = None, operation: str | None = None) -> bool:
        """Checks if file modification is currently authorized.
        
        Performs strict input validation:
        - If write_allowed is False, returns False immediately.
        - If operation is provided and is an unsupported operation (e.g. deletion, execution, shell), returns False.
        """
        if not self._write_allowed:
            return False

        if operation:
            op_lower = str(operation).lower().strip()
            if op_lower in ("delete", "delete_file", "remove", "destroy", "unlink", "rename_file", "move_file", "chmod", "chown", "execute", "shell"):
                logger.warning(f"Write permission check DENIED for unsupported operation: '{operation}'")
                return False

        return self._write_allowed

    def grant_write_permission(self, target_file: str | None = None) -> None:
        """Explicitly grants file write permission."""
        logger.info(f"Write permission EXPLICITLY GRANTED for target: {target_file or 'all files'}")
        self._write_allowed = True

    def revoke_write_permission(self) -> None:
        """Revokes file write permission."""
        logger.info("Write permission REVOKED.")
        self._write_allowed = False
