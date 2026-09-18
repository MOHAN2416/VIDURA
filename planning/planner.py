import logging
import os
import re
from pathlib import Path
from typing import Any
from config import load_config
from codebase import CodebaseManager
from codebase.models import ModuleInfo
from models.base import BaseLLMProvider
from task_understanding.models import DeveloperTask, TaskType
from planning.models import DeveloperPlan, PlannedComponent

logger = logging.getLogger("VIDURA.planning.planner")

AMBIGUOUS_PATTERNS = [
    r"^(?:please\s+)?fix(?:\s+the)?\s+bug\.?$",
    r"^(?:please\s+)?fix(?:\s+the)?\s+issue\.?$",
    r"^(?:please\s+)?improve(?:\s+the)?\s+database\.?$",
    r"^(?:please\s+)?improve(?:\s+the)?\s+code(?:base)?\.?$",
    r"^(?:please\s+)?update(?:\s+the)?\s+code\.?$",
    r"^(?:please\s+)?make(?:\s+some)?\s+changes\.?$",
    r"^(?:please\s+)?do(?:\s+the)?\s+work\.?$",
    r"^(?:please\s+)?refactor(?:\s+the)?\s+code\.?$",
    r"^(?:please\s+)?do\s+something\.?$",
]


class DeveloperPlanner:
    """Codebase-aware development planner for VIDURA.
    
    Inspects actual workspace codebase using CodebaseManager, resolves symbols,
    analyzes dependency and importer graphs, and produces a validated DeveloperPlan
    without side effects (strictly read-only).
    """

    def __init__(
        self,
        codebase_manager: CodebaseManager | None = None,
        model: BaseLLMProvider | None = None,
        workspace_root: str | Path | None = None,
        experience_manager: Any = None,
        rag_manager: Any = None,
    ) -> None:
        self.codebase_manager = codebase_manager or CodebaseManager(workspace_root=str(workspace_root) if workspace_root else None)
        self.model = model
        self.experience_manager = experience_manager
        self.rag_manager = rag_manager
        if workspace_root:
            self.workspace_root = Path(workspace_root).resolve()
        else:
            self.workspace_root = self.codebase_manager.scanner.workspace_root.resolve()

    def create_plan(self, task: DeveloperTask) -> DeveloperPlan:
        """Alias for plan()."""
        return self.plan(task)

    def create_developer_plan(self, task: DeveloperTask) -> DeveloperPlan:
        """Alias for plan()."""
        return self.plan(task)

    def plan(self, task: DeveloperTask) -> DeveloperPlan:
        """Analyzes a DeveloperTask against the actual codebase and returns a DeveloperPlan.

        Args:
            task: The understood DeveloperTask to plan.

        Returns:
            A strongly typed, validated DeveloperPlan.
        """
        logger.info(f"Generating codebase-aware plan for task: '{task.goal}' (type={task.task_type.value})")

        # 1. Non-development or general conversation queries
        if not task.is_development_task or task.task_type == TaskType.GENERAL:
            return DeveloperPlan(
                original_task=task,
                relevant_files=[],
                relevant_symbols=[],
                dependencies=[],
                affected_components=[],
                planned_changes=[f"Provide direct response for general query: '{task.goal}'"],
                risks=[],
                constraints=list(task.constraints),
                requires_more_information=False,
                missing_information_reason=None,
                confidence=1.0,
            )

        # Ambiguity check: if goal is underspecified without explicit targets
        clean_goal = task.goal.lower().strip()
        is_ambiguous = any(re.search(pat, clean_goal) for pat in AMBIGUOUS_PATTERNS)
        if is_ambiguous and not task.target_files and not task.target_symbols:
            logger.info(f"Ambiguous request detected: '{task.goal}'. Halting with requires_more_information=True.")
            return DeveloperPlan(
                original_task=task,
                relevant_files=[],
                relevant_symbols=[],
                dependencies=[],
                affected_components=[],
                planned_changes=[],
                risks=[],
                constraints=list(task.constraints),
                requires_more_information=True,
                missing_information_reason=f"Ambiguous request '{task.goal}': Insufficient context to determine target file, symbol, or intended behavior.",
                confidence=0.1,
            )

        self.codebase_manager.ensure_scanned()

        relevant_files: list[str] = []
        relevant_symbols: list[str] = []
        dependencies: list[str] = []
        affected_components: list[str] = []
        planned_changes: list[str] = []
        risks: list[str] = []
        requires_more_info = False
        missing_reasons: list[str] = []

        # 2. Analyze Explicit Target Files
        for raw_file in task.target_files:
            file_clean = raw_file.strip().lstrip("./")
            abs_path = (self.workspace_root / file_clean).resolve()

            if not abs_path.is_relative_to(self.workspace_root):
                missing_reasons.append(f"Target path '{raw_file}' violates workspace boundaries.")
                requires_more_info = True
                continue

            rel_path = str(abs_path.relative_to(self.workspace_root))

            if not abs_path.exists():
                # Check if file_clean matches an indexed module relative path (e.g. "agent.py" -> "agent/agent.py")
                for mod in self.codebase_manager.list_modules():
                    if mod.relative_path == file_clean or mod.relative_path.endswith(f"/{file_clean}"):
                        cand_abs = (self.workspace_root / mod.relative_path).resolve()
                        if cand_abs.exists():
                            abs_path = cand_abs
                            rel_path = mod.relative_path
                            break

            if abs_path.exists() and abs_path.is_file():
                if rel_path not in relevant_files:
                    relevant_files.append(rel_path)

                mod_info = self.codebase_manager.get_module(rel_path)
                if mod_info:
                    # Collect symbols defined in this file
                    for cls in mod_info.classes:
                        if cls.name not in relevant_symbols:
                            relevant_symbols.append(cls.name)
                    for fn in mod_info.functions:
                        if fn.name not in relevant_symbols:
                            relevant_symbols.append(fn.name)

                    # Collect dependencies and importers
                    for dep in self.codebase_manager.find_dependencies(mod_info.name):
                        if dep not in dependencies:
                            dependencies.append(dep)
                    for imp in self.codebase_manager.find_importers(mod_info.name):
                        if imp not in affected_components:
                            affected_components.append(imp)
            else:
                # File does not exist on disk
                if task.task_type in (TaskType.CODE_DEBUG, TaskType.CODE_REVIEW, TaskType.CODE_EXPLANATION) or \
                   any(kw in task.goal.lower() for kw in ("modify ", "update ", "fix ", "review ", "edit ", "change ")):
                    missing_reasons.append(f"Target file '{raw_file}' does not exist in workspace.")
                    requires_more_info = True
                else:
                    # New file planned for creation in code_change
                    planned_changes.append(f"Planned new file to be created: '{rel_path}'.")
                    if rel_path not in relevant_files:
                        relevant_files.append(rel_path)

        # 3. Analyze Explicit Target Symbols
        for raw_sym in task.target_symbols:
            sym_clean = raw_sym.strip()
            sym_res = self.codebase_manager.resolve_symbol_to_module(sym_clean)

            if sym_res and sym_res.get("module"):
                actual_sym = sym_res.get("symbol", sym_clean)
                if actual_sym not in relevant_symbols:
                    relevant_symbols.append(actual_sym)

                mod_name = sym_res["module"]
                mod_info = self.codebase_manager.get_module(mod_name)
                if mod_info:
                    if mod_info.relative_path not in relevant_files:
                        relevant_files.append(mod_info.relative_path)

                for dep in self.codebase_manager.find_dependencies(mod_name):
                    if dep not in dependencies:
                        dependencies.append(dep)
                for imp in self.codebase_manager.find_importers(mod_name):
                    if imp not in affected_components:
                        affected_components.append(imp)
            else:
                # Symbol cannot be resolved in codebase index
                missing_reasons.append(f"Symbol '{sym_clean}' could not be resolved in the codebase index.")
                requires_more_info = True

        # 4. Handle Unknown Targets (No explicit files or symbols identified)
        if not relevant_files and not relevant_symbols:
            # Safely check if any module or symbol in the index strongly matches terms in goal
            goal_lower = task.goal.lower()
            discovered = False

            # Search indexed modules for keyword matches
            for mod in self.codebase_manager.list_modules():
                mod_base = mod.name.split(".")[-1].lower()
                if len(mod_base) > 3 and mod_base in goal_lower:
                    if mod.relative_path not in relevant_files:
                        relevant_files.append(mod.relative_path)
                    discovered = True
                    for dep in self.codebase_manager.find_dependencies(mod.name):
                        if dep not in dependencies:
                            dependencies.append(dep)
                    for imp in self.codebase_manager.find_importers(mod.name):
                        if imp not in affected_components:
                            affected_components.append(imp)

            # If not discovered via direct keyword, query RAG hybrid retrieval if available
            if not discovered and self.rag_manager:
                try:
                    from rag.models import RAGQuery
                    rag_query = RAGQuery(query=task.goal, top_k=3, min_score=0.2)
                    rag_candidates = self.rag_manager.retrieve(rag_query)
                    for rc in rag_candidates:
                        cand_rel = rc.chunk.relative_path
                        # Authoritative filesystem check: target file must exist on disk
                        cand_abs = (self.workspace_root / cand_rel).resolve()
                        if cand_abs.exists() and cand_abs.is_file():
                            if cand_rel not in relevant_files:
                                relevant_files.append(cand_rel)
                            discovered = True
                            cand_mod = cand_rel.replace(".py", "").replace("/", ".")
                            for dep in self.codebase_manager.find_dependencies(cand_mod):
                                if dep not in dependencies:
                                    dependencies.append(dep)
                            for imp in self.codebase_manager.find_importers(cand_mod):
                                if imp not in affected_components:
                                    affected_components.append(imp)
                except Exception as r_err:
                    logger.warning(f"RAG target discovery encountered error: {r_err}")

            if not discovered:
                requires_more_info = True
                missing_reasons.append(
                    f"No target files or symbols could be identified from request '{task.goal}'. "
                    "Insufficient codebase evidence to determine target component."
                )

        # 5. Read relevant file contents safely (Read-Only inspection)
        inspected_count = 0
        for rf in list(relevant_files):
            abs_p = (self.workspace_root / rf).resolve()
            if abs_p.exists() and abs_p.is_file():
                inspected_count += 1
                try:
                    with open(abs_p, "r", encoding="utf-8", errors="replace") as f:
                        _ = f.read(50000) # Capped read to avoid memory exhaustion
                except Exception as err:
                    logger.warning(f"Failed to inspect file '{rf}': {err}")

        # 6. Formulate Grounded Risks based on actual dependencies and importers
        for comp in affected_components:
            importers = self.codebase_manager.find_importers(comp)
            if importers:
                risks.append(f"Modifying '{comp}' may affect downstream importers: {', '.join(sorted(importers))}.")
            else:
                risks.append(f"Component '{comp}' is imported by target module.")

        for rf in relevant_files:
            if "agent/loop" in rf:
                risks.append("Modifications to AgentLoop impact the core agent execution lifecycle.")
            elif "memory/store" in rf:
                risks.append("Modifications to MemoryStore impact persistent SQLite database records.")
            elif "tools/registry" in rf:
                risks.append("Modifications to ToolRegistry impact registration and execution of all tools.")

        # Deduplicate risks
        unique_risks = []
        for r in risks:
            if r not in unique_risks:
                unique_risks.append(r)

        # 6b. Retrieve Relevant Experiences & Advisory Warnings (Phase 11)
        plan_experiences: list[dict[str, Any]] = []
        if self.experience_manager:
            try:
                from experience.models import ExperienceQuery
                exp_query = ExperienceQuery(
                    query=task.goal,
                    file_path=relevant_files[0] if relevant_files else None,
                    symbol=relevant_symbols[0] if relevant_symbols else None,
                    limit=5,
                    min_score=0.15,
                )
                scored_exps = self.experience_manager.retrieve_experiences(exp_query)
                for se in scored_exps:
                    exp = se.experience
                    plan_experiences.append(se.to_dict())
                    if se.is_warning or not exp.success:
                        warn_msg = (
                            f"Advisory Warning from past experience: {exp.failure_reason or exp.result}. "
                            f"Caution: {exp.recommendation or exp.lesson or 'Check regression impact.'}"
                        )
                        if warn_msg not in unique_risks:
                            unique_risks.append(warn_msg)
            except Exception as exp_err:
                logger.warning(f"Error retrieving past experiences during planning: {exp_err}")

        # 6c. Assemble Structured RAG Context (Phase 12)
        assembled_rag_context = None
        plan_chunks = []
        if self.rag_manager:
            try:
                from rag.models import RAGQuery
                rag_query = RAGQuery(
                    query=task.goal,
                    target_files=relevant_files,
                    target_symbols=relevant_symbols,
                    top_k=5,
                    min_score=0.15,
                )
                ctx_obj = self.rag_manager.assemble_context(rag_query)
                assembled_rag_context = ctx_obj.to_dict()
                plan_chunks = [c.to_dict() for c in ctx_obj.retrieved_chunks]
            except Exception as rag_err:
                logger.warning(f"Error assembling RAG context during planning: {rag_err}")

        # 7. Formulate Declarative Planned Changes (Non-executable)
        if task.task_type == TaskType.CODE_CHANGE:
            for rf in relevant_files:
                planned_changes.append(f"Inspect existing implementation in '{rf}'.")
                planned_changes.append(f"Draft changes to satisfy: {task.goal}.")
        elif task.task_type == TaskType.CODE_DEBUG:
            for rf in relevant_files:
                planned_changes.append(f"Analyze bug location in '{rf}'.")
                planned_changes.append(f"Develop regression fix for reported issue in '{rf}'.")
        elif task.task_type == TaskType.CODE_EXPLANATION:
            for sym in relevant_symbols:
                planned_changes.append(f"Explain architecture and role of '{sym}'.")
            for rf in relevant_files:
                planned_changes.append(f"Detail component behavior in '{rf}'.")
        elif task.task_type == TaskType.CODE_REVIEW:
            for rf in relevant_files:
                planned_changes.append(f"Review '{rf}' for code quality, potential defects, and test coverage.")
        elif task.task_type == TaskType.TEST_REQUEST:
            for rf in relevant_files:
                planned_changes.append(f"Design unit tests covering functionality in '{rf}'.")

        if not planned_changes and not requires_more_info:
            planned_changes.append(f"Analyze codebase components related to '{task.goal}'.")

        # 8. Preserve Constraints from Task
        constraints = list(task.constraints)

        # 9. Compute Confidence
        if requires_more_info:
            confidence = 0.3
        elif relevant_files:
            confidence = 0.95
        else:
            confidence = 0.7

        reason_str = "; ".join(missing_reasons) if missing_reasons else None

        return DeveloperPlan(
            original_task=task,
            relevant_files=relevant_files,
            relevant_symbols=relevant_symbols,
            dependencies=dependencies,
            affected_components=affected_components,
            planned_changes=planned_changes,
            risks=unique_risks,
            constraints=constraints,
            requires_more_information=requires_more_info,
            missing_information_reason=reason_str,
            confidence=confidence,
            relevant_experiences=plan_experiences,
            rag_context=assembled_rag_context,
            retrieved_chunks=plan_chunks,
        )
