import json
import logging
import re
from models.base import BaseLLMProvider
from task_understanding.models import DeveloperTask, TaskType

logger = logging.getLogger("VIDURA.task_understanding.analyzer")

# Regex to match explicit python files, relative paths, and common filenames (e.g. agent/loop.py, math_utils.py, agent.py)
FILE_PATH_PATTERN = re.compile(
    r"\b([a-zA-Z0-9_\-./]+\.(?:py|json|md|txt|toml|yaml|yml|sh))\b",
    re.IGNORECASE,
)

# Regex to match PascalCase symbol names (e.g. AgentLoop, MemoryStore, ToolRegistry)
SYMBOL_PATTERN = re.compile(r"\b([A-Z][a-zA-Z0-9_]+)\b")

CONSTRAINT_PATTERNS = [
    (r"\bwithout changing (?:existing|current) behavior\b", "preserve existing behavior"),
    (r"\bwithout breaking (?:existing|current) (?:tests|behavior|functionality)\b", "preserve existing tests and behavior"),
    (r"\bwithout modifying (?:existing|current) (?:behavior|code)\b", "preserve existing behavior"),
    (r"\bpreserve (?:existing|current) behavior\b", "preserve existing behavior"),
    (r"\bpreserve backward compatibility\b", "preserve backward compatibility"),
]

COMMON_NON_SYMBOLS = {
    # Tech / General non-code nouns
    "Python", "Java", "JavaScript", "TypeScript", "Rust", "Golang", "Docker",
    "VIDURA", "JSON", "AST", "LLM", "CLI", "API", "REST", "SQL", "HTML", "CSS",
    "URL", "UTF", "Git", "Linux", "Unix", "Windows", "MacOS",
    # Question words / Pronouns / Conjunctions / Prepositions
    "What", "How", "Why", "Where", "When", "Who", "Which", "Whom", "Whose",
    "This", "That", "These", "Those", "There", "Here", "It", "Its",
    "The", "A", "An", "In", "On", "At", "To", "For", "Of", "With", "By",
    "From", "About", "Into", "Through", "During", "Before", "After", "Above",
    "Below", "Between", "Under", "And", "Or", "But", "So", "If", "Because",
    "As", "Until", "While", "All", "Any", "Some", "Every", "Each", "No", "Not",
    # Verbs / Actions
    "Add", "Create", "Update", "Modify", "Change", "Fix", "Debug", "Review",
    "Audit", "Explain", "Describe", "Show", "Display", "Find", "Locate",
    "Search", "Write", "Read", "Check", "Verify", "Validate", "Test", "Tests",
    "Run", "Execute", "Apply", "Propose", "Make", "Ensure", "Preserve", "Do",
    "Does", "Did", "Can", "Could", "Would", "Should", "Shall", "Will", "May",
    "Might", "Must", "Is", "Are", "Was", "Were", "Be", "Been", "Being", "Have",
    "Has", "Had", "Having", "Get", "Set", "Please", "Thanks", "Thank", "Hello",
    "Hi", "Hey", "Good", "Morning", "Evening", "Night", "Help",
    "Improve", "Enhance", "Optimize", "Refactor", "Implement", "Extend", "Support"
}


class DeveloperTaskAnalyzer:
    """Analyzes and structures user prompts into DeveloperTask instances without side effects."""

    def __init__(self, model: BaseLLMProvider | None = None) -> None:
        self.model = model

    def analyze(self, user_request: str) -> DeveloperTask:
        """Analyzes a user request string and returns a validated DeveloperTask instance."""
        if not user_request or not user_request.strip():
            return DeveloperTask(task_type=TaskType.GENERAL, is_development_task=False, goal="")

        req_clean = user_request.strip()

        # 1. Target Files & Symbols Extraction strictly from explicit request text
        extracted_files = self._extract_explicit_files(req_clean)
        extracted_symbols = self._extract_explicit_symbols(req_clean)
        extracted_constraints = self._extract_explicit_constraints(req_clean)

        # 2. Deterministic rule-based classification
        rule_task = self._rule_based_classify(req_clean, extracted_files, extracted_symbols, extracted_constraints)

        # 3. LLM-assisted classification & refinement if model is available
        if self.model:
            try:
                llm_task = self._llm_classify(req_clean, extracted_files, extracted_symbols, extracted_constraints)
                if llm_task and llm_task.task_type != TaskType.UNABLE_TO_CLASSIFY:
                    # Sanitize target files to strictly prohibit hallucinated files not in raw request text
                    llm_task.target_files = [f for f in llm_task.target_files if f in req_clean]
                    if not llm_task.target_files and ("bug" in req_clean.lower() or "fix" in req_clean.lower() or "add" in req_clean.lower()):
                        if llm_task.is_development_task:
                            llm_task.requires_codebase_analysis = True
                    return llm_task
            except Exception as err:
                logger.warning(f"LLM task analysis failed, falling back to rule-based: {err}")

        return rule_task

    def _extract_explicit_files(self, text: str) -> list[str]:
        """Extracts file names/paths explicitly mentioned in the input text."""
        matches = FILE_PATH_PATTERN.findall(text)
        files = []
        for m in matches:
            cleaned = m.rstrip(".,;:!?")
            if cleaned and cleaned in text and cleaned not in files:
                files.append(cleaned)
        return files

    def _extract_explicit_symbols(self, text: str) -> list[str]:
        """Extracts candidate PascalCase class/type symbols explicitly mentioned in input text."""
        matches = SYMBOL_PATTERN.findall(text)
        symbols = []
        for m in matches:
            cleaned = m.rstrip(".,;:!?")
            if cleaned and cleaned not in COMMON_NON_SYMBOLS and cleaned in text and cleaned not in symbols:
                symbols.append(cleaned)
        return symbols

    def _extract_explicit_constraints(self, text: str) -> list[str]:
        """Extracts explicit user constraints from text."""
        constraints = []
        text_lower = text.lower()
        for pattern, label in CONSTRAINT_PATTERNS:
            if re.search(pattern, text_lower):
                if label not in constraints:
                    constraints.append(label)
        return constraints

    def _rule_based_classify(
        self,
        text: str,
        files: list[str],
        symbols: list[str],
        constraints: list[str],
    ) -> DeveloperTask:
        """Determines DeveloperTask using deterministic pattern matching."""
        lower_req = text.lower().strip()

        # General conversation patterns (checked first)
        if lower_req.startswith(("what is", "who is", "explain what is", "how to install", "tell me about", "hello", "hi ", "hey")) and not files:
            return DeveloperTask(
                task_type=TaskType.GENERAL,
                is_development_task=False,
                goal=text,
                confidence=0.95,
            )

        # Test request
        if "test" in lower_req and ("add test" in lower_req or "write test" in lower_req or "tests for" in lower_req):
            return DeveloperTask(
                task_type=TaskType.TEST_REQUEST,
                is_development_task=True,
                goal=text,
                requested_change=f"Add unit tests for {', '.join(files) if files else 'specified code'}",
                target_files=files,
                target_symbols=symbols,
                constraints=constraints,
                requires_codebase_analysis=len(files) == 0,
                confidence=0.9,
            )

        # Code review
        if "review" in lower_req or "audit" in lower_req or "potential problems" in lower_req:
            return DeveloperTask(
                task_type=TaskType.CODE_REVIEW,
                is_development_task=True,
                goal=text,
                target_files=files,
                target_symbols=symbols,
                constraints=constraints,
                requires_codebase_analysis=len(files) == 0,
                confidence=0.9,
            )

        # Code explanation
        if lower_req.startswith("explain ") or "how does" in lower_req or "explain how" in lower_req:
            return DeveloperTask(
                task_type=TaskType.CODE_EXPLANATION,
                is_development_task=True,
                goal=text,
                target_files=files,
                target_symbols=symbols,
                constraints=constraints,
                requires_codebase_analysis=len(files) == 0 and len(symbols) == 0,
                confidence=0.9,
            )

        # Code debug
        if "fix " in lower_req or "bug" in lower_req or "debug" in lower_req or "error in" in lower_req or "issue in" in lower_req:
            return DeveloperTask(
                task_type=TaskType.CODE_DEBUG,
                is_development_task=True,
                goal=text,
                requested_change=f"Fix bug/issue described in prompt",
                target_files=files,
                target_symbols=symbols,
                constraints=constraints,
                requires_codebase_analysis=len(files) == 0,
                confidence=0.9,
            )

        # Code change
        if "add " in lower_req or "create " in lower_req or "modify " in lower_req or "update " in lower_req or "implement " in lower_req or "refactor " in lower_req or "improve " in lower_req or "enhance " in lower_req:
            return DeveloperTask(
                task_type=TaskType.CODE_CHANGE,
                is_development_task=True,
                goal=text,
                requested_change=text,
                target_files=files,
                target_symbols=symbols,
                constraints=constraints,
                requires_codebase_analysis=len(files) == 0,
                confidence=0.9,
            )

        # Default fallback if files or symbols are explicitly mentioned
        if files or symbols:
            return DeveloperTask(
                task_type=TaskType.CODE_CHANGE if "add" in lower_req or "modify" in lower_req else TaskType.CODE_EXPLANATION,
                is_development_task=True,
                goal=text,
                target_files=files,
                target_symbols=symbols,
                constraints=constraints,
                requires_codebase_analysis=False,
                confidence=0.7,
            )

        # General conversation default
        return DeveloperTask(
            task_type=TaskType.GENERAL,
            is_development_task=False,
            goal=text,
            confidence=0.8,
        )

    def _llm_classify(
        self,
        user_request: str,
        files: list[str],
        symbols: list[str],
        constraints: list[str],
    ) -> DeveloperTask | None:
        """Queries LLM provider to classify task and returns validated DeveloperTask."""
        if not self.model:
            return None

        system_instruction = (
            "You are VIDURA's Developer Task Analyzer.\n"
            "Analyze the user's input prompt and classify it into a structured task representation.\n"
            "Supported Task Types: 'general', 'code_change', 'code_debug', 'code_explanation', 'code_review', 'test_request'.\n\n"
            "STRICT RULES:\n"
            "1. Output ONLY a valid JSON object matching the schema below.\n"
            "2. Do NOT invent or hallucinate target_files. If no file is explicitly named in the prompt, set target_files to [].\n"
            "3. Extract constraints ONLY if explicitly stated in the prompt text.\n\n"
            "JSON Schema:\n"
            "{\n"
            '  "task_type": "general" | "code_change" | "code_debug" | "code_explanation" | "code_review" | "test_request",\n'
            '  "is_development_task": boolean,\n'
            '  "goal": "summary of user goal",\n'
            '  "requested_change": "description of change or null",\n'
            '  "target_files": ["explicit_file_paths_from_prompt"],\n'
            '  "target_symbols": ["explicit_symbol_names_from_prompt"],\n'
            '  "constraints": ["explicit_constraints_from_prompt"],\n'
            '  "requires_codebase_analysis": boolean,\n'
            '  "confidence": float_between_0_and_1\n'
            "}"
        )

        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": f"User Prompt: {user_request}"},
        ]

        raw_response = self.model.generate(messages)
        if not raw_response:
            return None

        cleaned = raw_response.strip()
        if "```" in cleaned:
            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
            if match:
                cleaned = match.group(1).strip()
            else:
                brace = re.search(r"(\{.*\})", cleaned, re.DOTALL)
                if brace:
                    cleaned = brace.group(1).strip()

        try:
            data = json.loads(cleaned, strict=False)
            task = DeveloperTask.from_dict(data)
            # Ensure target_files strictly contains only filenames explicitly in prompt text
            task.target_files = [f for f in task.target_files if f in user_request]
            if files:
                for f in files:
                    if f not in task.target_files:
                        task.target_files.append(f)
            if symbols:
                for s in symbols:
                    if s not in task.target_symbols:
                        task.target_symbols.append(s)
            if constraints:
                for c in constraints:
                    if c not in task.constraints:
                        task.constraints.append(c)
            return task
        except Exception as err:
            logger.warning(f"Failed to parse LLM JSON task classification response: {err}")
            return None
