import re
from memory.models import MemoryCategory


def extract_explicit_memory_request(text: str) -> tuple[str | None, str]:
    """Detects explicit user requests to remember information and returns (statement, category).

    Examples:
        "Remember that I prefer Python." -> ("I prefer Python.", "user")
        "Remember VIDURA must stay offline." -> ("VIDURA must stay offline.", "project")

    Returns:
        Tuple of (statement: str | None, category: str). Returns (None, "") if no explicit request found.
    """
    if not text:
        return None, ""

    cleaned = text.strip()

    # Regex patterns for explicit memory requests
    patterns = [
        r"^(?:please\s+)?remember\s+that\s+(.+)$",
        r"^(?:please\s+)?remember\s+(.+)$",
    ]

    extracted_statement: str | None = None
    for pattern in patterns:
        match = re.search(pattern, cleaned, re.IGNORECASE)
        if match:
            extracted_statement = match.group(1).strip()
            # Clean up trailing punctuation if present
            if extracted_statement.endswith("."):
                extracted_statement = extracted_statement[:-1].strip() + "."
            break

    if not extracted_statement:
        return None, ""

    # Categorize into PROJECT vs USER
    lower_stmt = extracted_statement.lower()
    project_keywords = {"vidura", "project", "architecture", "config", "system", "database", "model"}
    
    category = MemoryCategory.USER
    if any(k in lower_stmt for k in project_keywords):
        category = MemoryCategory.PROJECT

    return extracted_statement, category
