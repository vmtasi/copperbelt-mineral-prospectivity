from pathlib import Path
import subprocess
import sys
import os

from mcp.server import MCPServer


# =============================================================================
# PROJECT CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(
    r"C:\Users\vanmu\copperbelt-mineral-prospectivity"
).resolve()

mcp = MCPServer(
    "Copperbelt Mineral Prospectivity",
    instructions=(
        "MCP server for the Copperbelt Mineral Prospectivity research project. "
        "Provides controlled access to project files, source code, Git state, "
        "analysis outputs, tests, and project-local execution."
    ),
)


# =============================================================================
# SECURITY / PATH HELPERS
# =============================================================================

def safe_path(relative_path: str = "") -> Path:
    """
    Resolve a path relative to the project root and prevent path traversal.
    """
    target = (PROJECT_ROOT / relative_path).resolve()

    if target != PROJECT_ROOT and PROJECT_ROOT not in target.parents:
        raise ValueError(
            "Requested path is outside the Copperbelt project."
        )

    return target


def relative_to_project(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT))


def run_subprocess(
    command: list[str],
    timeout: int = 120,
) -> str:
    """
    Run a command from the project root.

    stdout and stderr are combined so the MCP client receives the complete
    execution record.
    """
    try:
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )

        output = result.stdout

        if result.stderr:
            output += "\n--- STDERR ---\n"
            output += result.stderr

        output += f"\n--- EXIT CODE: {result.returncode} ---"

        return output.strip()

    except subprocess.TimeoutExpired:
        return f"ERROR: Command timed out after {timeout} seconds."

    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"


# =============================================================================
# 1. PROJECT INFORMATION
# =============================================================================

@mcp.tool()
def project_info() -> str:
    """
    Return basic information about the connected Copperbelt project.
    """
    return (
        f"Project: Copperbelt Mineral Prospectivity\n"
        f"Root: {PROJECT_ROOT}\n"
        f"Exists: {PROJECT_ROOT.exists()}\n"
        f"Python: {sys.executable}\n"
    )


# =============================================================================
# 2. PROJECT CONTEXT
# =============================================================================

@mcp.tool()
def get_project_context() -> str:
    """
    Return the high-level project structure and important research files.
    """
    important_dirs = [
        "src",
        "scripts",
        "data",
        "figures",
        "figures/audit",
        "notebooks",
        "tests",
        "docs",
    ]

    lines = [
        "COPPERBELT MINERAL PROSPECTIVITY PROJECT",
        f"Root: {PROJECT_ROOT}",
        "",
        "IMPORTANT DIRECTORIES:",
    ]

    for directory in important_dirs:
        path = safe_path(directory)
        status = "exists" if path.exists() else "missing"
        lines.append(f"  {directory}/ [{status}]")

    lines.extend(
        [
            "",
            "ROOT FILES:",
        ]
    )

    try:
        for item in sorted(PROJECT_ROOT.iterdir()):
            if item.name.startswith("."):
                continue

            suffix = "/" if item.is_dir() else ""
            lines.append(f"  {item.name}{suffix}")

    except OSError as exc:
        lines.append(f"ERROR reading project root: {exc}")

    return "\n".join(lines)


# =============================================================================
# 3. LIST PROJECT FILES
# =============================================================================

@mcp.tool()
def list_project_files(
    relative_path: str = "",
    pattern: str = "*",
    recursive: bool = False,
) -> str:
    """
    List files and directories inside the project.

    relative_path:
        Directory relative to project root.

    pattern:
        Glob pattern such as *.py, *.csv, *.md, or *.

    recursive:
        If True, search recursively.
    """
    try:
        target = safe_path(relative_path)
    except ValueError as exc:
        return f"ERROR: {exc}"

    if not target.exists():
        return f"ERROR: Path does not exist: {relative_path}"

    if not target.is_dir():
        return f"ERROR: Not a directory: {relative_path}"

    try:
        iterator = (
            target.rglob(pattern)
            if recursive
            else target.glob(pattern)
        )

        results = []

        for item in sorted(iterator):
            try:
                relative = item.relative_to(PROJECT_ROOT)
                suffix = "/" if item.is_dir() else ""
                results.append(f"{relative}{suffix}")
            except ValueError:
                continue

        if not results:
            return "No matching files found."

        return "\n".join(results)

    except OSError as exc:
        return f"ERROR: {exc}"


# =============================================================================
# 4. READ PROJECT FILE
# =============================================================================

@mcp.tool()
def read_project_file(
    relative_path: str,
    max_chars: int = 200000,
) -> str:
    """
    Read a text file inside the Copperbelt project.

    Large files are truncated at max_chars.
    """
    try:
        target = safe_path(relative_path)
    except ValueError as exc:
        return f"ERROR: {exc}"

    if not target.exists():
        return f"ERROR: File does not exist: {relative_path}"

    if not target.is_file():
        return f"ERROR: Not a file: {relative_path}"

    allowed_extensions = {
        ".py",
        ".md",
        ".txt",
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".csv",
        ".tsv",
        ".log",
        ".ini",
        ".cfg",
        ".xml",
    }

    if target.suffix.lower() not in allowed_extensions:
        return (
            f"ERROR: File type '{target.suffix}' is not enabled "
            "for text reading."
        )

    try:
        text = target.read_text(
            encoding="utf-8",
            errors="replace",
        )
    except OSError as exc:
        return f"ERROR reading file: {exc}"

    if len(text) > max_chars:
        return (
            text[:max_chars]
            + "\n\n--- FILE TRUNCATED ---\n"
            + f"Maximum returned characters: {max_chars}"
        )

    return text


# =============================================================================
# 5. SEARCH PROJECT
# =============================================================================

@mcp.tool()
def search_project(
    query: str,
    relative_path: str = "",
    pattern: str = "*.py",
    max_results: int = 200,
) -> str:
    """
    Search project text files for a case-insensitive string.

    Examples:

        search_project("LogisticRegression")
        search_project("deposit_present", "src", "*.py")
        search_project("Wasserstein", "figures/audit", "*.csv")
    """
    try:
        target = safe_path(relative_path)
    except ValueError as exc:
        return f"ERROR: {exc}"

    if not target.exists() or not target.is_dir():
        return f"ERROR: Directory does not exist: {relative_path}"

    matches = []
    query_lower = query.lower()

    try:
        for file in target.rglob(pattern):

            if not file.is_file():
                continue

            try:
                text = file.read_text(
                    encoding="utf-8",
                    errors="replace",
                )
            except OSError:
                continue

            for line_number, line in enumerate(
                text.splitlines(),
                start=1,
            ):
                if query_lower in line.lower():
                    rel = file.relative_to(PROJECT_ROOT)

                    matches.append(
                        f"{rel}:{line_number}: {line.strip()}"
                    )

                    if len(matches) >= max_results:
                        return (
                            "\n".join(matches)
                            + f"\n\n--- RESULT LIMIT ({max_results}) REACHED ---"
                        )

    except OSError as exc:
        return f"ERROR: {exc}"

    if not matches:
        return "No matches found."

    return "\n".join(matches)


# =============================================================================
# 6. INSPECT PYTHON MODULE
# =============================================================================

@mcp.tool()
def inspect_python_module(
    relative_path: str,
) -> str:
    """
    Inspect a Python source file and return structural information:
    classes, functions, imports, decorators and line counts.
    """
    try:
        target = safe_path(relative_path)
    except ValueError as exc:
        return f"ERROR: {exc}"

    if not target.exists():
        return f"ERROR: File does not exist: {relative_path}"

    if target.suffix.lower() != ".py":
        return "ERROR: inspect_python_module requires a .py file."

    command = [
        sys.executable,
        "-c",
        """
import ast
import sys

path = sys.argv[1]

with open(path, "r", encoding="utf-8", errors="replace") as f:
    source = f.read()

tree = ast.parse(source, filename=path)

print(f"File: {path}")
print(f"Lines: {len(source.splitlines())}")
print()

print("IMPORTS:")
for node in tree.body:
    if isinstance(node, ast.Import):
        for alias in node.names:
            print(f"  import {alias.name}")
    elif isinstance(node, ast.ImportFrom):
        module = node.module or ""
        names = ", ".join(alias.name for alias in node.names)
        print(f"  from {module} import {names}")

print()
print("FUNCTIONS:")

for node in ast.walk(tree):
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        decorators = [
            ast.unparse(d) for d in node.decorator_list
        ]

        print(
            f"  {node.name} "
            f"(line {node.lineno})"
        )

        if decorators:
            print(
                f"    decorators: "
                f"{', '.join(decorators)}"
            )

print()
print("CLASSES:")

for node in ast.walk(tree):
    if isinstance(node, ast.ClassDef):
        print(
            f"  {node.name} "
            f"(line {node.lineno})"
        )

        methods = [
            child.name
            for child in node.body
            if isinstance(
                child,
                (ast.FunctionDef, ast.AsyncFunctionDef)
            )
        ]

        for method in methods:
            print(f"    method: {method}")
""",
        str(target),
    ]

    return run_subprocess(command)


# =============================================================================
# 7. GIT STATUS
# =============================================================================

@mcp.tool()
def git_status() -> str:
    """
    Return the current Git repository status.
    """
    return run_subprocess(
        ["git", "status", "--short", "--branch"]
    )


# =============================================================================
# 8. GIT DIFF
# =============================================================================

@mcp.tool()
def git_diff(
    staged: bool = False,
    relative_path: str = "",
) -> str:
    """
    Return the current Git diff.

    staged:
        Show staged changes instead of working-tree changes.

    relative_path:
        Optional project-relative path to restrict the diff.
    """
    command = ["git", "diff"]

    if staged:
        command.append("--cached")

    if relative_path:
        try:
            target = safe_path(relative_path)
        except ValueError as exc:
            return f"ERROR: {exc}"

        command.extend(
            ["--", relative_to_project(target)]
        )

    return run_subprocess(command)


# =============================================================================
# 9. GIT LOG
# =============================================================================

@mcp.tool()
def git_log(
    count: int = 10,
) -> str:
    """
    Return recent Git commits.
    """
    count = max(1, min(count, 100))

    return run_subprocess(
        [
            "git",
            "log",
            f"-{count}",
            "--oneline",
            "--decorate",
        ]
    )


# =============================================================================
# 10. RUN PYTHON SCRIPT
# =============================================================================

@mcp.tool()
def run_python_script(
    relative_path: str,
    arguments: list[str] | None = None,
    timeout: int = 300,
) -> str:
    """
    Run a Python script inside the Copperbelt project.

    The script must be located inside the project.

    Example:

        run_python_script(
            "scripts/phase7_6_sensitivity_analysis.py"
        )
    """
    try:
        script = safe_path(relative_path)
    except ValueError as exc:
        return f"ERROR: {exc}"

    if not script.exists():
        return f"ERROR: Script does not exist: {relative_path}"

    if not script.is_file():
        return "ERROR: Script path is not a file."

    if script.suffix.lower() != ".py":
        return "ERROR: run_python_script requires a .py file."

    command = [sys.executable, str(script)]

    if arguments:
        command.extend(arguments)

    return run_subprocess(command, timeout=timeout)


# =============================================================================
# 11. RUN PYTEST
# =============================================================================

@mcp.tool()
def run_pytest(
    path: str = "",
    extra_args: list[str] | None = None,
    timeout: int = 600,
) -> str:
    """
    Run pytest inside the project.

    path:
        Optional test file/directory.

    extra_args:
        Optional additional pytest arguments.
    """
    command = [sys.executable, "-m", "pytest"]

    if path:
        try:
            target = safe_path(path)
        except ValueError as exc:
            return f"ERROR: {exc}"

        command.append(str(target))

    if extra_args:
        command.extend(extra_args)

    return run_subprocess(command, timeout=timeout)


# =============================================================================
# 12. RUN PROJECT COMMAND
# =============================================================================

@mcp.tool()
def run_command(
    command: list[str],
    timeout: int = 120,
) -> str:
    """
    Run a command from the Copperbelt project root.

    This is intentionally executed with cwd fixed to the project root.
    """
    if not command:
        return "ERROR: No command supplied."

    return run_subprocess(command, timeout=timeout)


# =============================================================================
# 13. LIST AUDIT OUTPUTS
# =============================================================================

@mcp.tool()
def list_audit_outputs(
    pattern: str = "*",
) -> str:
    """
    List generated audit outputs under figures/audit.
    """
    audit_dir = safe_path("figures/audit")

    if not audit_dir.exists():
        return "Audit directory does not exist."

    results = []

    for item in sorted(audit_dir.glob(pattern)):
        relative = item.relative_to(PROJECT_ROOT)
        suffix = "/" if item.is_dir() else ""
        results.append(f"{relative}{suffix}")

    if not results:
        return "No audit outputs found."

    return "\n".join(results)


# =============================================================================
# 14. READ AUDIT OUTPUT
# =============================================================================

@mcp.tool()
def read_audit_output(
    relative_path: str,
    max_chars: int = 200000,
) -> str:
    """
    Read a generated audit output from figures/audit.

    The path must remain inside figures/audit.
    """
    audit_root = safe_path("figures/audit")

    try:
        target = safe_path(
            str(Path("figures/audit") / relative_path)
        )
    except ValueError as exc:
        return f"ERROR: {exc}"

    if target != audit_root and audit_root not in target.parents:
        return "ERROR: File is outside figures/audit."

    if not target.exists():
        return f"ERROR: Audit file does not exist: {relative_path}"

    if not target.is_file():
        return "ERROR: Not a file."

    try:
        text = target.read_text(
            encoding="utf-8",
            errors="replace",
        )
    except OSError as exc:
        return f"ERROR: {exc}"

    if len(text) > max_chars:
        text = (
            text[:max_chars]
            + "\n\n--- FILE TRUNCATED ---"
        )

    return text


# =============================================================================
# 15. FIND PROJECT REFERENCES
# =============================================================================

@mcp.tool()
def find_project_references(
    query: str,
    max_results: int = 200,
) -> str:
    """
    Search the major research/project directories for a term.

    Searches:
        src/
        scripts/
        tests/
        docs/
        figures/audit/
    """
    directories = [
        "src",
        "scripts",
        "tests",
        "docs",
        "figures/audit",
    ]

    results = []

    for directory in directories:
        target = safe_path(directory)

        if not target.exists():
            continue

        for file in target.rglob("*"):

            if not file.is_file():
                continue

            if file.suffix.lower() not in {
                ".py",
                ".md",
                ".txt",
                ".csv",
                ".json",
                ".yaml",
                ".yml",
                ".log",
            }:
                continue

            try:
                text = file.read_text(
                    encoding="utf-8",
                    errors="replace",
                )
            except OSError:
                continue

            for line_number, line in enumerate(
                text.splitlines(),
                start=1,
            ):
                if query.lower() in line.lower():

                    rel = file.relative_to(PROJECT_ROOT)

                    results.append(
                        f"{rel}:{line_number}: {line.strip()}"
                    )

                    if len(results) >= max_results:
                        return (
                            "\n".join(results)
                            + f"\n\n--- RESULT LIMIT ({max_results}) REACHED ---"
                        )

    if not results:
        return "No references found."

    return "\n".join(results)


# =============================================================================
# SERVER STARTUP
# =============================================================================

if __name__ == "__main__":
    print(
        "Copperbelt MCP server starting on "
        "http://127.0.0.1:8000/mcp",
        flush=True,
    )

    mcp.run(
        transport="streamable-http",
        host="127.0.0.1",
        port=8000,
    )