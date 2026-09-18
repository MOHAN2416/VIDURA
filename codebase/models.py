from dataclasses import dataclass, field


@dataclass
class CodeFile:
    """Represents metadata of a file within the project workspace."""
    path: str
    relative_path: str
    size: int
    is_python: bool


@dataclass
class ImportInfo:
    """Represents an import statement in a Python module."""
    module: str
    names: list[str] = field(default_factory=list)
    is_local: bool = False
    line_number: int = 1


@dataclass
class FunctionInfo:
    """Represents a Python function or class method."""
    name: str
    module_path: str
    parameters: list[dict[str, str]] = field(default_factory=list)
    return_annotation: str | None = None
    docstring: str | None = None
    line_number: int = 1
    is_async: bool = False
    is_method: bool = False


@dataclass
class ClassInfo:
    """Represents a Python class definition."""
    name: str
    module_path: str
    bases: list[str] = field(default_factory=list)
    methods: list[FunctionInfo] = field(default_factory=list)
    docstring: str | None = None
    line_number: int = 1


@dataclass
class ModuleInfo:
    """Represents structured analysis of a Python module."""
    name: str
    file_path: str
    relative_path: str
    docstring: str | None = None
    imports: list[ImportInfo] = field(default_factory=list)
    local_imports: list[str] = field(default_factory=list)
    classes: list[ClassInfo] = field(default_factory=list)
    functions: list[FunctionInfo] = field(default_factory=list)
    parse_error: str | None = None
