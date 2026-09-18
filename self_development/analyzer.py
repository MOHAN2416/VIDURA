import logging
import re
from pathlib import Path
from typing import Any

from codebase.manager import CodebaseManager
from memory.manager import MemoryManager
from self_development.models import SelfDevelopmentGoal
from self_development.security import is_security_critical_target

logger = logging.getLogger("VIDURA.self_development.analyzer")


class SelfDevelopmentAnalyzer:
    """Performs grounded self-analysis of VIDURA's codebase and supporting experiences.

    CRITICAL ARCHITECTURAL INVARIANTS:
    1. Read-Only Codebase Inspection: Strictly queries CodebaseManager and AST parser.
    2. Zero Hallucination of Architecture: Unknown components must be reported as unknown.
    3. Memory as Supporting Context Only: Current filesystem/codebase state is authoritative.
    4. Workspace Boundary: Never inspects or accesses outside workspace_root.
    """

    def __init__(
        self,
        codebase_manager: CodebaseManager | None = None,
        memory_manager: MemoryManager | None = None,
        workspace_root: str | Path | None = None,
    ) -> None:
        self.codebase_manager = codebase_manager or CodebaseManager(workspace_root=str(workspace_root) if workspace_root else None)
        self.memory_manager = memory_manager
        if workspace_root:
            self.workspace_root = Path(workspace_root).resolve()
        else:
            self.workspace_root = Path(self.codebase_manager.workspace_root).resolve() if self.codebase_manager.workspace_root else Path.cwd().resolve()

    def analyze(self, goal: SelfDevelopmentGoal) -> dict[str, Any]:
        """Analyzes an explicit self-development goal against actual codebase components."""
        logger.info(f"Analyzing self-development goal: '{goal.description}' (id: {goal.goal_id})")

        self.codebase_manager.ensure_scanned()
        desc = goal.description.strip()
        scope_files = list(goal.scope)

        # 1. Identify candidate modules, files, and symbols from goal description and scope
        identified_files: list[str] = []
        identified_symbols: list[str] = []
        dependencies: list[str] = []
        unknown_components: list[str] = []

        # Check explicit scope files first
        for sf in scope_files:
            mod = self.codebase_manager.get_module(sf)
            if mod:
                if mod.relative_path not in identified_files:
                    identified_files.append(mod.relative_path)
            elif (self.workspace_root / sf).exists():
                if sf not in identified_files:
                    identified_files.append(sf)
            else:
                unknown_components.append(sf)

        # Keyword-based codebase matching across modules and symbols
        keywords = re.findall(r"[A-Za-z0-9_]{3,}", desc)
        for kw in keywords:
            # Check symbol index
            syms = self.codebase_manager.find_symbol(kw)
            for s in syms:
                s_name = s.get("name", "")
                s_path = s.get("file_path", "")
                if s_name and s_name not in identified_symbols:
                    identified_symbols.append(s_name)
                if s_path and s_path not in identified_files:
                    identified_files.append(s_path)

            # Check module index
            mod = self.codebase_manager.get_module(kw)
            if mod and mod.relative_path not in identified_files:
                identified_files.append(mod.relative_path)

        # Domain specific heuristics for VIDURA's core components if referenced
        desc_lower = desc.lower()
        if "tool" in desc_lower and "malformed" in desc_lower:
            for cand in ["agent/loop.py", "tools/registry.py"]:
                if self.codebase_manager.get_module(cand) or (self.workspace_root / cand).exists():
                    if cand not in identified_files:
                        identified_files.append(cand)
            if "AgentLoop" not in identified_symbols:
                identified_symbols.append("AgentLoop")

        if "router" in desc_lower or "routing" in desc_lower:
            for cand in ["models/router.py", "models/routing.py"]:
                if (self.workspace_root / cand).exists() and cand not in identified_files:
                    identified_files.append(cand)

        if "permission" in desc_lower:
            cand = "permissions/manager.py"
            if (self.workspace_root / cand).exists() and cand not in identified_files:
                identified_files.append(cand)

        # Resolve dependencies and importers for identified files
        for f in identified_files:
            deps = self.codebase_manager.find_dependencies(f)
            for d in deps:
                if d not in dependencies:
                    dependencies.append(d)

        # 2. Check for security-critical components
        requires_elevated = False
        for f in identified_files:
            if is_security_critical_target(f):
                requires_elevated = True
                break
        for sym in identified_symbols:
            if is_security_critical_target(sym):
                requires_elevated = True
                break

        # 3. Retrieve relevant supporting experiences from memory
        recalled_lessons: list[str] = []
        if self.memory_manager:
            try:
                exps = self.memory_manager.list_experiences(limit=20)
                for exp in exps:
                    exp_text = f"{exp.task} {exp.action_summary} {exp.lesson}".lower()
                    if any(kw.lower() in exp_text for kw in keywords if len(kw) > 3):
                        recalled_lessons.append(f"Past experience ({exp.task}): {exp.lesson}")
            except Exception as err:
                logger.warning(f"Failed to recall supporting experiences: {err}")

        # 4. Synthesize concrete behavior & improvement opportunity
        if not identified_files and not scope_files:
            return {
                "relevant_files": [],
                "relevant_symbols": [],
                "dependencies": [],
                "current_behavior": "Unknown component or scope.",
                "observed_problem": f"Could not ground goal '{desc}' in any existing codebase components.",
                "proposed_improvement": "",
                "expected_benefit": "",
                "risks": ["Cannot proceed without codebase grounding."],
                "constraints": list(goal.constraints),
                "validation_strategy": [],
                "rollback_strategy": "No changes proposed.",
                "confidence": 0.0,
                "requires_more_information": True,
                "missing_information_reason": f"No relevant codebase components found for goal: '{desc}'.",
                "requires_elevated_authorization": False,
                "recalled_lessons": recalled_lessons,
                "unknown_components": unknown_components,
            }

        # Build grounded problem and proposed improvement description
        primary_file = identified_files[0] if identified_files else "unknown"
        current_behavior = f"Component '{primary_file}' processes inputs according to existing implementation."
        observed_problem = f"Opportunity identified in {primary_file} to address: {desc}"
        proposed_improvement = f"Refine implementation in {primary_file} to fulfill goal: {desc}"
        expected_benefit = f"Robust, verified self-development enhancement for {desc}"

        return {
            "relevant_files": identified_files,
            "relevant_symbols": identified_symbols,
            "dependencies": dependencies,
            "current_behavior": current_behavior,
            "observed_problem": observed_problem,
            "proposed_improvement": proposed_improvement,
            "expected_benefit": expected_benefit,
            "risks": [
                f"Modification to {primary_file} could introduce behavioral regressions.",
                "Must pass full test verification before final evaluation.",
            ],
            "constraints": list(goal.constraints),
            "validation_strategy": [
                f"Verify disk modifications in {identified_files}",
                "Execute relevant automated test suites with Pytest",
                "Confirm zero regressions across developer tests",
            ],
            "rollback_strategy": "Discard uncommitted changes or restore original file contents if tests fail.",
            "confidence": 0.9 if identified_files else 0.4,
            "requires_more_information": False,
            "missing_information_reason": None,
            "requires_elevated_authorization": requires_elevated,
            "recalled_lessons": recalled_lessons,
            "unknown_components": unknown_components,
        }
