# =============================================================================
# Copperbelt Mineral Prospectivity — MCP server (v3, unified)
#
# THREAT MODEL AND HONEST LIMITS
# ------------------------------
# 1. MCP-level path restrictions CANNOT sandbox arbitrary Python execution.
#    Any script that runs can rewrite any file the user can write, including
#    frozen files and .git/. run_python_script is a TRUSTED escape hatch.
#    The allowlist narrows it but does not eliminate the risk.
#
# 2. Frozen-file enforcement is enforced ONLY at the MCP tool surface, not
#    at the execution layer. An approved script (or a pytest run) can bypass
#    it. Execution is gated by a pre-existing SHA-256 baseline stored outside
#    the project and the baseline is never recaptured by this server.
#
# 3. Mutations and approved diagnostic executions are preceded by a debounced
#    tracked-files-only stash snapshot. Untracked and ignored files are excluded.
#
# 4. Every tool call is logged to an out-of-band file the agent cannot edit.
#
# 5. Server `instructions` are NOT security controls. The controls are
#    guard_write(), guard_exec(), guard_test_path(), safe_path(),
#    ALLOWLISTED_SCRIPTS, FROZEN_FILES, and the frozen-file baseline.
#
# FROZEN HIERARCHY
# ----------------
#   canonical model definition  -> src/v11_spatial_stability_final.py
#   canonical fitted state      -> figures/v11_fold_{1..4}_trace.nc
#   shared validation design    -> src/validation_strategies.py
#   persistent context          -> AI_PROJECT_CONTEXT.md
#
#   Derived products (v11_oof_predictions.csv, v11_geological_stability_matrix.csv,
#   Phase 7 outputs, etc.) are NOT frozen. They are regenerated from the
#   canonical artifacts above. Freezing them would hide regeneration errors
#   rather than surface them.
# =============================================================================

from __future__ import annotations

import functools
import hashlib
import inspect
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from mcp.server import MCPServer


# =============================================================================
# PROJECT CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(
    r"C:\Users\vanmu\copperbelt-mineral-prospectivity"
).resolve()
PROJECT_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"


# -----------------------------------------------------------------------------
# Frozen files — canonical model definitions and fitted V11 artifacts.
#
# These files are authoritative research artifacts. They must not be
# overwritten or regenerated during downstream analysis. A genuine change
# requires an explicit model-versioning decision.
# -----------------------------------------------------------------------------

FROZEN_FILES: frozenset[str] = frozenset({
    # Canonical V11 model definition
    "src/v11_spatial_stability_final.py",

    # Shared spatial-CV / fold strategy — changing it alters the validation design
    "src/validation_strategies.py",

    # Persistent project instructions/context
    "AI_PROJECT_CONTEXT.md",

    # Canonical V11 fitted posterior traces (fold-wise)
    "figures/v11_fold_1_trace.nc",
    "figures/v11_fold_2_trace.nc",
    "figures/v11_fold_3_trace.nc",
    "figures/v11_fold_4_trace.nc",
})

# Server-internal subtrees the agent must never write into.
FROZEN_PREFIXES: tuple[str, ...] = (
    ".git/",
    ".trash/",
)

# The only places write tools may touch.
WRITABLE_PREFIXES: tuple[str, ...] = (
    "src/",
    "scripts/",
    "tests/",
    "docs/",
    "figures/audit/",
    "paper_1/stages/",
    "README.md"
)


# -----------------------------------------------------------------------------
# Curated execution allowlist.
#
# Only lightweight, reviewed diagnostics may be executed by name.
# v11_spatial_stability_final.py is intentionally EXCLUDED: it is frozen, and
# running it would overwrite the four V11 trace files.
# -----------------------------------------------------------------------------

ALLOWLISTED_SCRIPTS: frozenset[str] = frozenset({
    "src/compute_fold_counts.py",
    "src/validate_litho_contact_pipeline.py",
})

ALLOWLISTED_TESTS: frozenset[str] = frozenset({
    "tests/test_distance_sensitivity_math.py",
    "tests/test_litho_contact_pipeline.py",
    "tests/test_mcp_server_policy.py",
})

MAX_SUBPROCESS_OUTPUT_BYTES = 64 * 1024
MAX_SCRIPT_TIMEOUT_SECONDS = 120
MAX_PYTEST_TIMEOUT_SECONDS = 120


# -----------------------------------------------------------------------------
# Snapshot / recovery
# -----------------------------------------------------------------------------

_LAST_SNAPSHOT_TS: float = 0.0
_SNAPSHOT_DEBOUNCE_SECONDS: int = 300   # 5 min


# -----------------------------------------------------------------------------
# Out-of-band state: audit log and frozen-file baseline.
#
# Both live OUTSIDE the project so an approved script cannot silently
# rewrite them along with the artifacts they are meant to protect.
# -----------------------------------------------------------------------------

_STATE_DIR = Path.home() / ".copperbelt-mcp"

AUDIT_LOG = Path(
    os.environ.get(
        "MCP_AUDIT_LOG",
        str(_STATE_DIR / "calls.log"),
    )
)

FROZEN_BASELINE = Path(
    os.environ.get(
        "MCP_FROZEN_BASELINE",
        str(_STATE_DIR / "frozen_baseline.json"),
    )
)


# =============================================================================
# MCP SERVER
# =============================================================================

mcp = MCPServer(
    "Copperbelt Mineral Prospectivity",
    instructions=(
        "MCP server for the Copperbelt Mineral Prospectivity research project. "
        "Writes are restricted to src/, scripts/, tests/, docs/ and "
        "figures/audit/. The files src/v11_spatial_stability_final.py, "
        "src/validation_strategies.py, AI_PROJECT_CONTEXT.md and the four "
        "canonical V11 posterior traces (figures/v11_fold_*_trace.nc) are "
        "frozen. Script execution is restricted to a small allowlist of "
        "reviewed diagnostics; pytest is restricted to named, reviewed test files. "
        "IMPORTANT: subprocess execution is a trusted operation — a running "
        "Python process can bypass all MCP-level path restrictions. "
        "Call verify_frozen_intact() to compare against the existing baseline. "
        "The server never captures or replaces that baseline. "
        "Every mutation is preceded by a tracked-files-only recovery snapshot."
    ),
)


# =============================================================================
# NORMALISATION / PATH GUARDS
# =============================================================================

def _norm(relative_path: str) -> str:
    p = (relative_path or "").replace("\\", "/").strip()
    while p.startswith("./"):
        p = p[2:]
    return p.lstrip("/")


def safe_path(relative_path: str = "") -> Path:
    if not isinstance(relative_path, str):
        raise ValueError("Requested path must be a relative string.")
    normalized = relative_path.replace("\\", "/")
    if normalized.startswith("/") or re.match(r"^[A-Za-z]:", normalized):
        raise ValueError("Absolute paths are not permitted.")
    target = (PROJECT_ROOT / Path(normalized)).resolve()
    if target != PROJECT_ROOT and PROJECT_ROOT not in target.parents:
        raise ValueError("Requested path is outside the Copperbelt project.")
    return target


def relative_to_project(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT)).replace("\\", "/")


def _is_internal_path(path: Path) -> bool:
    try:
        relative = relative_to_project(path).casefold()
    except ValueError:
        return True
    return relative == ".git" or relative.startswith((".git/", ".trash/"))


def guard_write(relative_path: str) -> str | None:
    try:
        target = safe_path(relative_path)
        p = relative_to_project(target)
    except (TypeError, ValueError) as exc:
        return f"ERROR: {exc}"

    if not p:
        return "ERROR: Empty relative path."

    if p in FROZEN_FILES:
        return (
            f"ERROR: '{relative_path}' is frozen. "
            "It must be regenerated by its source pipeline, not edited. "
            "If a genuine bug is suspected, report it — do not modify."
        )

    for prefix in FROZEN_PREFIXES:
        if p == prefix.rstrip("/") or p.startswith(prefix):
            return f"ERROR: '{relative_path}' is inside a frozen subtree ({prefix})."

    if not any(
        p == prefix.rstrip("/") or p.startswith(prefix)
        for prefix in WRITABLE_PREFIXES
    ):
        return (
            f"ERROR: '{relative_path}' is outside the permitted write areas: "
            f"{list(WRITABLE_PREFIXES)}"
        )

    return None


def guard_exec(relative_path: str) -> str | None:
    """
    Enforce the curated execution allowlist.

    Policy gate, not a sandbox. Any script that passes can still do anything
    the user can do.
    """
    try:
        p = relative_to_project(safe_path(relative_path))
    except (TypeError, ValueError) as exc:
        return f"ERROR: {exc}"

    if p in FROZEN_FILES:
        return f"ERROR: '{relative_path}' is frozen and cannot be executed."

    # Canonical figure artifacts may never be executed, even if they somehow
    # appear in the allowlist.
    if p.startswith("figures/") and not p.startswith("figures/audit/"):
        return (
            f"ERROR: '{relative_path}' is a canonical figure artifact and "
            "may not be executed."
        )

    if p in ALLOWLISTED_SCRIPTS:
        return None

    return (
        f"ERROR: '{relative_path}' is not an approved executable. "
        "Only the reviewed scripts in ALLOWLISTED_SCRIPTS may run. "
        "Use list_executables() to see the current set."
    )


def guard_test_path(relative_path: str) -> str | None:
    """Allow only named, reviewed tests; reject traversal and symlink aliases."""
    try:
        p = relative_to_project(safe_path(relative_path))
    except (TypeError, ValueError) as exc:
        return f"ERROR: {exc}"
    if p in ALLOWLISTED_TESTS:
        return None
    return f"ERROR: Path '{relative_path}' is not an approved test file."


# =============================================================================
# SUBPROCESS
# =============================================================================

def _python_environment(*, for_pytest: bool = False) -> dict[str, str]:
    system_root = Path(os.environ.get("SystemRoot", r"C:\Windows"))
    python_dir = PROJECT_PYTHON.resolve().parent
    path_entries = [
        str(python_dir),
        str(system_root / "System32"),
        str(system_root),
    ]
    environment = {
        key: os.environ[key]
        for key in ("SystemRoot", "WINDIR", "TEMP", "TMP", "SystemDrive", "ProgramData", "LOCALAPPDATA", "APPDATA", "USERPROFILE")
        if key in os.environ
    }
    environment.update({
        "PATH": os.pathsep.join(path_entries),
        "PYTHONPATH": str(PROJECT_ROOT),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    if for_pytest:
        environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    return environment


def _git_environment() -> dict[str, str]:
    environment = {
        key: os.environ[key]
        for key in ("SystemRoot", "WINDIR", "TEMP", "TMP", "PATH")
        if key in os.environ
    }
    environment.update({
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_COUNT": "2",
        "GIT_CONFIG_KEY_0": "core.fsmonitor",
        "GIT_CONFIG_VALUE_0": "false",
        "GIT_CONFIG_KEY_1": "core.pager",
        "GIT_CONFIG_VALUE_1": "cat",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_PAGER": "cat",
        "GIT_TERMINAL_PROMPT": "0",
    })
    return environment


def _project_python() -> str:
    if not PROJECT_PYTHON.is_file():
        raise FileNotFoundError(f"Project Python interpreter not found: {PROJECT_PYTHON}")
    return str(PROJECT_PYTHON)


def run_subprocess(
    command: list[str],
    timeout: int = 120,
    *,
    env: dict[str, str] | None = None,
    output_limit: int = MAX_SUBPROCESS_OUTPUT_BYTES,
) -> str:
    if not command or any(not isinstance(part, str) for part in command):
        return "ERROR: Command must be a non-empty list of strings."
    if timeout < 1:
        return "ERROR: Timeout must be a positive number of seconds."
    if output_limit < 1:
        return "ERROR: Output limit must be positive."

    process = None
    captured = {"stdout": bytearray(), "stderr": bytearray()}
    state = {"bytes": 0, "truncated": False}
    lock = threading.Lock()

    def drain(name: str, stream) -> None:
        while True:
            chunk = stream.read(8192)
            if not chunk:
                return
            with lock:
                available = max(0, output_limit - state["bytes"])
                kept = chunk[:available]
                captured[name].extend(kept)
                state["bytes"] += len(kept)
                if len(kept) < len(chunk):
                    state["truncated"] = True

    try:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        assert process.stdout is not None and process.stderr is not None
        readers = [
            threading.Thread(target=drain, args=("stdout", process.stdout)),
            threading.Thread(target=drain, args=("stderr", process.stderr)),
        ]
        for reader in readers:
            reader.start()
        timed_out = False
        try:
            return_code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.kill()
            return_code = process.wait()
        for reader in readers:
            reader.join()
        output = captured["stdout"].decode("utf-8", errors="replace")
        if captured["stderr"]:
            output += "\n--- STDERR ---\n"
            output += captured["stderr"].decode("utf-8", errors="replace")
        if state["truncated"]:
            output += f"\n--- OUTPUT TRUNCATED AT {output_limit} BYTES ---"
        if timed_out:
            output += f"\nERROR: Command timed out after {timeout} seconds."
        else:
            output += f"\n--- EXIT CODE: {return_code} ---"
        return output.strip()
    except (OSError, ValueError) as exc:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        return f"ERROR: {type(exc).__name__}: {exc}"


# =============================================================================
# Snapshot (auto-invoked before mutations and allowlisted executions)
# =============================================================================

def _run_git(*args: str, timeout: int = 30) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=PROJECT_ROOT, capture_output=True,
        text=True, timeout=timeout, encoding="utf-8", errors="replace",
        env=_git_environment(),
    )


def maybe_snapshot(reason: str) -> str:
    """
    Take a recovery snapshot of the working tree if we haven't done so
    recently. Debounced to avoid a stash per keystroke.
    """
    global _LAST_SNAPSHOT_TS

    now = time.time()
    if now - _LAST_SNAPSHOT_TS < _SNAPSHOT_DEBOUNCE_SECONDS:
        return "(snapshot debounced; recent recovery point active)"

    status = _run_git("status", "--porcelain", "--untracked-files=no")
    if status.returncode != 0:
        return f"(WARN snapshot: git status failed: {status.stderr.strip()[:120]})"

    if not status.stdout.strip():
        _LAST_SNAPSHOT_TS = now
        return "(no tracked changes; untracked and ignored files were left untouched)"

    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    msg = f"mcp-snapshot-{ts}-{reason}"[:80]

    push = _run_git("stash", "push", "-m", msg)
    if push.returncode != 0:
        return f"(WARN snapshot: stash push failed: {push.stderr.strip()[:160]})"
    if "No local changes to save" in push.stdout:
        _LAST_SNAPSHOT_TS = now
        return "(no tracked changes to snapshot; untracked and ignored files were left untouched)"

    apply = _run_git("stash", "apply", "--index")
    if apply.returncode != 0:
        return (
            f"(WARN snapshot '{msg}' created but apply failed; "
            f"recover manually: git stash list)"
        )

    _LAST_SNAPSHOT_TS = now
    return f"(snapshot '{msg}' created; recover with: git stash list)"


# =============================================================================
# FROZEN-FILE INTEGRITY BASELINE
#
# run_python_script() and run_pytest() can modify files the user can write,
# including frozen artifacts. Execution is gated on the pre-existing baseline,
# which is checked before and after the subprocess and never recaptured.
# =============================================================================

def _hash_file(path: Path) -> str | None:
    """SHA-256 of a file's raw bytes, or None if it does not exist / unreadable."""
    try:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _load_frozen_baseline() -> dict | None:
    if not FROZEN_BASELINE.exists():
        return None
    try:
        baseline = json.loads(FROZEN_BASELINE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(baseline, dict) or not isinstance(baseline.get("files"), dict):
        return None
    return baseline


def _frozen_integrity_report() -> tuple[bool, str]:
    baseline = _load_frozen_baseline()
    if baseline is None:
        return False, f"ERROR: Existing frozen baseline is missing or invalid: {FROZEN_BASELINE}"
    if baseline.get("project_root") != str(PROJECT_ROOT):
        return False, "ERROR: Frozen baseline belongs to a different project root."

    lines = [
        f"Baseline captured: {baseline.get('captured_at', 'unknown')}",
        f"Project root:      {baseline.get('project_root', 'unknown')}",
        "Baseline coverage uses the configured .nc trace paths; .pkl traces are not covered.",
        "",
        "FROZEN FILE STATUS:",
    ]
    drifted: list[str] = []
    missing: list[str] = []
    ok: list[str] = []
    entries = baseline["files"]

    for rel in sorted(FROZEN_FILES):
        entry = entries.get(rel)
        if not isinstance(entry, dict) or "exists" not in entry or "sha256" not in entry:
            drifted.append(rel)
            lines.append(f"  {rel}: MISSING BASELINE ENTRY")
            continue
        try:
            target = safe_path(rel)
        except ValueError:
            missing.append(rel)
            lines.append(f"  {rel}: UNREACHABLE")
            continue

        exists_at_baseline = entry["exists"] is True
        expected_hash = entry["sha256"]
        if not target.exists():
            if exists_at_baseline:
                missing.append(rel)
                lines.append(f"  {rel}: MISSING (was present at baseline)")
            else:
                ok.append(rel)
                lines.append(f"  {rel}: absent at baseline and now")
            continue

        current = _hash_file(target)
        if exists_at_baseline and current is not None and current == expected_hash:
            ok.append(rel)
            lines.append(f"  {rel}: OK")
        else:
            drifted.append(rel)
            lines.append(f"  {rel}: DRIFTED OR CREATED SINCE BASELINE")

    lines.extend([
        "",
        f"Summary: {len(ok)} OK, {len(drifted)} drifted, {len(missing)} missing",
    ])
    if drifted or missing:
        lines.extend([
            "",
            "ACTION REQUIRED: A frozen artifact differs from the existing "
            "baseline. Investigate before continuing; this server never "
            "recaptures or replaces the baseline.",
        ])
    return not (drifted or missing), "\n".join(lines)


# =============================================================================
# AUDIT DECORATOR
# =============================================================================

def audited(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        started = time.monotonic()
        try:
            result = fn(*args, **kwargs)
        except Exception as exc:
            record = _audit_record(
                fn,
                args,
                kwargs,
                started,
                status="error",
                error_type=type(exc).__name__,
            )
            audit_error = _append_audit_record(record)
            if audit_error:
                raise RuntimeError(
                    f"{fn.__name__} failed and audit logging also failed: "
                    f"{audit_error}"
                ) from exc
            raise

        status = _audit_result_status(result)
        record = _audit_record(fn, args, kwargs, started, status=status, result=result)
        audit_error = _append_audit_record(record)
        if audit_error:
            message = f"ERROR: Tool completed but audit logging failed: {audit_error}"
            if isinstance(result, str):
                return f"{result}\n{message}"
            raise OSError(message)
        return result

    return wrapper


def _audit_result_status(result) -> str:
    if not isinstance(result, str):
        return "ok"
    if result.startswith("ERROR") or "ERROR: Command timed out" in result:
        return "error"
    match = re.search(r"--- EXIT CODE: (-?\d+) ---", result)
    if match and int(match.group(1)) != 0:
        return "error"
    return "ok"


def _audit_record(
    fn,
    args: tuple,
    kwargs: dict,
    started: float,
    *,
    status: str,
    result=None,
    error_type: str | None = None,
) -> dict:
    bound = inspect.signature(fn).bind_partial(*args, **kwargs)
    safe_arguments = {}
    for name, value in bound.arguments.items():
        lower_name = name.lower()
        if lower_name in {
            "content", "old_text", "new_text", "query", "terms",
            "arguments", "args", "password", "secret", "token",
        }:
            if isinstance(value, str):
                safe_arguments[name] = f"<redacted length={len(value)}>"
            elif isinstance(value, (list, tuple)):
                safe_arguments[name] = f"<redacted count={len(value)}>"
            else:
                safe_arguments[name] = "<redacted>"
        elif isinstance(value, str):
            safe_arguments[name] = value[:120]
        elif isinstance(value, (bool, int, float)) or value is None:
            safe_arguments[name] = value
        elif isinstance(value, (list, tuple)):
            safe_arguments[name] = f"<list count={len(value)}>"
        else:
            safe_arguments[name] = f"<{type(value).__name__}>"

    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "tool": fn.__name__,
        "arguments": safe_arguments,
        "duration_ms": round((time.monotonic() - started) * 1000, 2),
        "status": status,
    }
    if error_type:
        record["error_type"] = error_type
    if result is not None:
        record["output_chars"] = len(result) if isinstance(result, str) else None
        if isinstance(result, str):
            match = re.search(r"--- EXIT CODE: (-?\d+) ---", result)
            if match:
                record["exit_code"] = int(match.group(1))
    return record


def _append_audit_record(record: dict) -> str | None:
    try:
        AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
        with AUDIT_LOG.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    except (OSError, TypeError, ValueError) as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


# =============================================================================
# 1. PROJECT INFORMATION
# =============================================================================

@mcp.tool()
@audited
def project_info() -> str:
    """Return basic information about the connected Copperbelt project."""
    return (
        f"Project: Copperbelt Mineral Prospectivity\n"
        f"Root: {PROJECT_ROOT}\n"
        f"Exists: {PROJECT_ROOT.exists()}\n"
        f"MCP server Python: {sys.executable}\n"
        f"Diagnostic Python: {PROJECT_PYTHON}\n"
        f"Audit log: {AUDIT_LOG}\n"
        f"Frozen baseline: {FROZEN_BASELINE}\n"
        f"Frozen files: {sorted(FROZEN_FILES)}\n"
        f"Frozen prefixes: {list(FROZEN_PREFIXES)}\n"
        f"Writable prefixes: {list(WRITABLE_PREFIXES)}\n"
        f"Allowlisted executables: {len(ALLOWLISTED_SCRIPTS)}\n"
        f"Allowlisted tests: {sorted(ALLOWLISTED_TESTS)}\n"
    )


# =============================================================================
# 2. LIST EXECUTABLES
# =============================================================================

@mcp.tool()
@audited
def list_executables() -> str:
    """List the scripts the agent is currently permitted to execute."""
    lines = ["ALLOWLISTED DIAGNOSTIC SCRIPTS:"]
    for p in sorted(ALLOWLISTED_SCRIPTS):
        lines.append(f"  {p}")
    lines.extend(["", "ALLOWLISTED TEST FILES:"])
    lines.extend(f"  {p}" for p in sorted(ALLOWLISTED_TESTS))
    lines.append("")
    lines.append(
        "NOTE: Execution is a trusted boundary. A running script can bypass "
        "all MCP-level path restrictions. Call verify_frozen_intact() after "
        "execution to detect frozen-file drift."
    )
    return "\n".join(lines)


# =============================================================================
# 3. VERIFY FROZEN INTACT
# =============================================================================

@mcp.tool()
@audited
def verify_frozen_intact(recapture: bool = False) -> str:
    """
    Compare frozen-file contents against the existing baseline without changing it.
    """
    if recapture:
        return "ERROR: Baseline recapture is disabled; the existing baseline is immutable."
    return _frozen_integrity_report()[1]


def _execution_integrity_gate() -> tuple[bool, str]:
    intact, report = _frozen_integrity_report()
    if not intact:
        return False, f"ERROR: Execution refused because frozen integrity is not verified.\n{report}"
    return True, report


# =============================================================================
# 4. PROJECT CONTEXT
# =============================================================================

@mcp.tool()
@audited
def get_project_context() -> str:
    """Return the high-level project structure and important research files."""
    important_dirs = [
        "src", "scripts", "data", "figures", "figures/audit",
        "notebooks", "tests", "docs",
    ]
    lines = [
        "COPPERBELT MINERAL PROSPECTIVITY PROJECT",
        f"Root: {PROJECT_ROOT}", "", "IMPORTANT DIRECTORIES:",
    ]
    for directory in important_dirs:
        path = safe_path(directory)
        status = "exists" if path.exists() else "missing"
        lines.append(f"  {directory}/ [{status}]")

    lines.extend(["", "ROOT FILES:"])
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
# 5. LIST PROJECT FILES
# =============================================================================

@mcp.tool()
@audited
def list_project_files(
    relative_path: str = "",
    pattern: str = "*",
    recursive: bool = False,
) -> str:
    """List files and directories inside the project."""
    try:
        target = safe_path(relative_path)
    except ValueError as exc:
        return f"ERROR: {exc}"

    if not target.exists():
        return f"ERROR: Path does not exist: {relative_path}"
    if not target.is_dir():
        return f"ERROR: Not a directory: {relative_path}"

    try:
        iterator = target.rglob(pattern) if recursive else target.glob(pattern)
        results = []
        truncated = False
        for item in sorted(iterator):
            if _is_internal_path(item):
                continue
            try:
                rel = item.relative_to(PROJECT_ROOT)
                suffix = "/" if item.is_dir() else ""
                results.append(f"{rel}{suffix}")
                if len(results) >= 1000:
                    truncated = True
                    break
            except ValueError:
                continue
        output = "\n".join(results) if results else "No matching files found."
        if truncated:
            output += "\n--- RESULT LIMIT (1000) REACHED ---"
        return output
    except OSError as exc:
        return f"ERROR: {exc}"


# =============================================================================
# 6. READ PROJECT FILE
# =============================================================================

@mcp.tool()
@audited
def read_project_file(relative_path: str, max_chars: int = 200000) -> str:
    """
    Read a text file inside the Copperbelt project. Content is wrapped in
    untrusted-content markers so downstream reasoning treats it as data.
    """
    try:
        target = safe_path(relative_path)
    except ValueError as exc:
        return f"ERROR: {exc}"

    if not target.exists():
        return f"ERROR: File does not exist: {relative_path}"
    if not target.is_file():
        return f"ERROR: Not a file: {relative_path}"

    allowed = {
        ".py", ".md", ".txt", ".json", ".yaml", ".yml", ".toml",
        ".csv", ".tsv", ".log", ".ini", ".cfg", ".xml", ".tex", ".bib",
    }
    if target.suffix.lower() not in allowed:
        return f"ERROR: File type '{target.suffix}' is not enabled for text reading."
    if _is_internal_path(target):
        return "ERROR: Internal repository and trash paths are not readable."
    max_chars = max(1, min(max_chars, 200000))

    try:
        text = target.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"ERROR reading file: {exc}"

    truncated = ""
    if len(text) > max_chars:
        text = text[:max_chars]
        truncated = f"\n\n--- FILE TRUNCATED (limit {max_chars} chars) ---"

    return (
        f"--- BEGIN UNTRUSTED PROJECT CONTENT: {relative_to_project(target)} ---\n"
        f"{text}{truncated}\n"
        f"--- END UNTRUSTED PROJECT CONTENT ---"
    )


# =============================================================================
# 7. WRITE / CREATE PROJECT FILE
# =============================================================================

@mcp.tool()
@audited
def write_project_file(
    relative_path: str,
    content: str,
    overwrite: bool = True,
    create_parents: bool = True,
) -> str:
    """
    Create or overwrite a UTF-8 text file inside the project.

    A recovery snapshot is taken automatically (debounced) before the write.
    Writing a script does not grant permission to execute it.
    """
    err = guard_write(relative_path)
    if err:
        return err

    try:
        target = safe_path(relative_path)
    except ValueError as exc:
        return f"ERROR: {exc}"

    if target == PROJECT_ROOT:
        return "ERROR: Cannot write to the project root itself."
    if target.exists() and target.is_dir():
        return "ERROR: Target path is a directory."
    if target.exists() and not overwrite:
        return f"ERROR: File already exists: {relative_path}"
    if len(content.encode("utf-8")) > 1_000_000:
        return "ERROR: File content exceeds the 1 MB write limit."

    snapshot_note = maybe_snapshot("write")

    try:
        if create_parents:
            target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8", newline="")

        return (
            f"SUCCESS: Wrote {relative_to_project(target)}\n"
            f"Bytes: {target.stat().st_size}\n"
            f"{snapshot_note}"
        )
    except OSError as exc:
        return f"ERROR writing file: {exc}"


# =============================================================================
# 8. PATCH PROJECT FILE
# =============================================================================

@mcp.tool()
@audited
def replace_in_project_file(
    relative_path: str,
    old_text: str,
    new_text: str,
    replace_all: bool = False,
) -> str:
    """Replace text in an existing project file. Snapshot is taken first."""
    err = guard_write(relative_path)
    if err:
        return err

    try:
        target = safe_path(relative_path)
    except ValueError as exc:
        return f"ERROR: {exc}"

    if not target.exists():
        return f"ERROR: File does not exist: {relative_path}"
    if not target.is_file():
        return "ERROR: Target is not a file."

    try:
        text = target.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"ERROR reading file: {exc}"

    occurrences = text.count(old_text)
    if occurrences == 0:
        return "ERROR: old_text was not found."
    if not replace_all and occurrences != 1:
        return (
            f"ERROR: old_text occurs {occurrences} times. "
            "Refusing ambiguous replacement."
        )

    snapshot_note = maybe_snapshot("patch")

    updated = (
        text.replace(old_text, new_text)
        if replace_all
        else text.replace(old_text, new_text, 1)
    )
    replacements = occurrences if replace_all else 1

    try:
        target.write_text(updated, encoding="utf-8", newline="")
    except OSError as exc:
        return f"ERROR writing file: {exc}"

    return (
        f"SUCCESS: Updated {relative_to_project(target)}\n"
        f"Replacements: {replacements}\n"
        f"{snapshot_note}"
    )


# =============================================================================
# 9. ARCHIVE PROJECT FILE
# =============================================================================

@mcp.tool()
@audited
def delete_project_file(relative_path: str) -> str:
    """Archive a file into .trash/<timestamp>/ instead of unlinking it."""
    err = guard_write(relative_path)
    if err:
        return err

    try:
        target = safe_path(relative_path)
    except ValueError as exc:
        return f"ERROR: {exc}"

    if target == PROJECT_ROOT:
        return "ERROR: Cannot delete project root."
    if not target.exists():
        return f"ERROR: File does not exist: {relative_path}"
    if not target.is_file():
        return "ERROR: Target is not a file. Directory deletion is disabled."

    snapshot_note = maybe_snapshot("archive")

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = PROJECT_ROOT / ".trash" / ts / relative_to_project(target)
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(target), str(dest))
        return (
            f"SUCCESS: Archived {relative_to_project(target)} -> "
            f"{dest.relative_to(PROJECT_ROOT)}\n{snapshot_note}"
        )
    except OSError as exc:
        return f"ERROR archiving file: {exc}"


# =============================================================================
# 10. SEARCH PROJECT
# =============================================================================

@mcp.tool()
@audited
def search_project(
    query: str,
    relative_path: str = "",
    pattern: str = "*.py",
    max_results: int = 200,
) -> str:
    """Search project text files for a case-insensitive string."""
    try:
        target = safe_path(relative_path)
    except ValueError as exc:
        return f"ERROR: {exc}"

    if not target.exists() or not target.is_dir():
        return f"ERROR: Directory does not exist: {relative_path}"

    max_results = max(1, min(max_results, 500))
    matches = []
    ql = query.lower()
    try:
        for file in target.rglob(pattern):
            if not file.is_file() or _is_internal_path(file):
                continue
            try:
                text = file.resolve().read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for n, line in enumerate(text.splitlines(), start=1):
                if ql in line.lower():
                    matches.append(
                        f"{file.relative_to(PROJECT_ROOT)}:{n}: {line.strip()[:300]}"
                    )
                    if len(matches) >= max_results:
                        return (
                            "\n".join(matches)
                            + f"\n\n--- RESULT LIMIT ({max_results}) REACHED ---"
                        )
    except OSError as exc:
        return f"ERROR: {exc}"
    return "\n".join(matches) if matches else "No matches found."


# =============================================================================
# 11. SCAN TERMINOLOGY
# =============================================================================

@mcp.tool()
@audited
def scan_terminology(
    terms: list[str],
    paths: list[str] | None = None,
    extensions: list[str] | None = None,
    max_results: int = 1000,
) -> str:
    """
    Locate forbidden/legacy terminology across the project. Frozen files are
    marked [FROZEN] so the agent knows where stale terms cannot be edited.
    """
    if not terms:
        return "ERROR: Provide at least one term."

    max_results = max(1, min(max_results, 500))
    search_roots = paths or [
        "src", "scripts", "tests", "docs", "figures/audit",
        "paper_1/manuscript", "paper_1/stages",
    ]
    exts = set(
        (e.lower() if e.startswith(".") else "." + e.lower())
        for e in (extensions or [
            ".py", ".md", ".txt", ".json", ".yaml", ".yml",
            ".toml", ".csv", ".tsv", ".rst", ".tex", ".bib",
        ])
    )

    hits: dict[str, list[str]] = {t: [] for t in terms}
    total, truncated = 0, False

    for root in search_roots:
        try:
            base = safe_path(root)
        except ValueError:
            continue
        if not base.exists():
            continue

        for file in base.rglob("*"):
            if (
                not file.is_file()
                or file.suffix.lower() not in exts
                or _is_internal_path(file)
            ):
                continue
            rel = str(file.relative_to(PROJECT_ROOT)).replace("\\", "/")
            frozen_marker = " [FROZEN]" if _norm(rel) in FROZEN_FILES else ""
            try:
                text = file.resolve().read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for n, line in enumerate(text.splitlines(), start=1):
                low = line.lower()
                for term in terms:
                    if term.lower() in low:
                        hits[term].append(
                            f"{rel}{frozen_marker}:{n}: {line.strip()[:200]}"
                        )
                        total += 1
                        if total >= max_results:
                            truncated = True
                            break
                if truncated:
                    break
            if truncated:
                break
        if truncated:
            break

    out = []
    for term in terms:
        out.append(f"=== '{term}' — {len(hits[term])} hit(s) ===")
        out.extend(hits[term] or ["  (none)"])
        out.append("")
    if truncated:
        out.append(f"--- RESULT LIMIT ({max_results}) REACHED ---")
    return "\n".join(out)


# =============================================================================
# 12. INSPECT PYTHON MODULE
# =============================================================================

@mcp.tool()
@audited
def inspect_python_module(relative_path: str) -> str:
    """Inspect a Python source file and return structural information."""
    try:
        target = safe_path(relative_path)
    except ValueError as exc:
        return f"ERROR: {exc}"

    if not target.exists():
        return f"ERROR: File does not exist: {relative_path}"
    if target.suffix.lower() != ".py":
        return "ERROR: inspect_python_module requires a .py file."

    command = [
        _project_python(), "-c",
        """
import ast, sys
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
        decorators = [ast.unparse(d) for d in node.decorator_list]
        print(f"  {node.name} (line {node.lineno})")
        if decorators:
            print(f"    decorators: {', '.join(decorators)}")
print()
print("CLASSES:")
for node in ast.walk(tree):
    if isinstance(node, ast.ClassDef):
        print(f"  {node.name} (line {node.lineno})")
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                print(f"    method: {child.name}")
""",
        str(target),
    ]
    return run_subprocess(command, env=_python_environment())


# =============================================================================
# 13. VALIDATE PYTHON FILE
# =============================================================================

@mcp.tool()
@audited
def validate_python_file(relative_path: str) -> str:
    """Syntax-check a Python file without executing it."""
    try:
        target = safe_path(relative_path)
    except ValueError as exc:
        return f"ERROR: {exc}"
    if not target.exists():
        return f"ERROR: File does not exist: {relative_path}"
    if target.suffix.lower() != ".py":
        return "ERROR: validate_python_file requires a .py file."
    try:
        python = _project_python()
    except OSError as exc:
        return f"ERROR: {exc}"
    return run_subprocess(
        [
            python,
            "-c",
            "import pathlib, sys; compile(pathlib.Path(sys.argv[1]).read_text(encoding='utf-8'), sys.argv[1], 'exec')",
            str(target),
        ],
        env=_python_environment(),
    )


# =============================================================================
# 14. GIT STATUS / DIFF / LOG
# =============================================================================

@mcp.tool()
@audited
def git_status() -> str:
    """Return the current Git repository status."""
    return run_subprocess(
        ["git", "--no-pager", "status", "--short", "--branch"],
        env=_git_environment(),
    )


@mcp.tool()
@audited
def git_diff(staged: bool = False, relative_path: str = "") -> str:
    """Return the current Git diff."""
    command = ["git", "--no-pager", "diff", "--no-ext-diff", "--no-textconv"]
    if staged:
        command.append("--cached")
    if relative_path:
        try:
            target = safe_path(relative_path)
        except ValueError as exc:
            return f"ERROR: {exc}"
        command.extend(["--", relative_to_project(target)])
    return run_subprocess(command, env=_git_environment())


@mcp.tool()
@audited
def git_log(count: int = 10) -> str:
    """Return recent Git commits."""
    count = max(1, min(count, 100))
    return run_subprocess(
        ["git", "--no-pager", "log", f"-{count}", "--oneline", "--decorate"],
        env=_git_environment(),
    )


# =============================================================================
# 15. GIT COMMAND (whitelisted, read-only)
# =============================================================================

GIT_READONLY_SUBCOMMANDS: frozenset[str] = frozenset({
    "status", "diff", "log", "show", "rev-parse", "ls-files",
})


@mcp.tool()
@audited
def git_command(args: list[str], timeout: int = 30) -> str:
    """Run a whitelisted read-only git subcommand from the project root."""
    if not args:
        return "ERROR: No git arguments supplied."
    if args[0] not in GIT_READONLY_SUBCOMMANDS:
        return (
            f"ERROR: git subcommand '{args[0]}' is not whitelisted. "
            f"Permitted: {sorted(GIT_READONLY_SUBCOMMANDS)}"
        )
    for arg in args[1:]:
        if (
            arg in {"-C", "-c", "-o", "--output", "--no-index", "--ext-diff", "--textconv"}
            or arg.startswith((
                "--git-dir", "--work-tree", "--config-env", "--output=",
                "--ext-diff=", "--textconv=",
            ))
        ):
            return f"ERROR: '{arg}' is not permitted in git_command."
    command = ["git", "--no-pager", *args]
    if args[0] in {"diff", "show", "log"}:
        command[3:3] = ["--no-ext-diff", "--no-textconv"]
    timeout = max(1, min(timeout, 120))
    return run_subprocess(command, timeout=timeout, env=_git_environment())


# =============================================================================
# 16. RUN PYTHON SCRIPT (allowlist-gated, snapshot-guarded)
# =============================================================================

@mcp.tool()
@audited
def run_python_script(
    relative_path: str,
    arguments: list[str] | None = None,
    timeout: int = MAX_SCRIPT_TIMEOUT_SECONDS,
) -> str:
    """
    Execute a Python script from the curated allowlist.

    This remains a trusted operation: a Python process can bypass MCP-level
    path restrictions. Only reviewed diagnostics are permitted, with fixed
    arguments and a bounded runtime.
    """
    err = guard_exec(relative_path)
    if err:
        return err

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
    try:
        python = _project_python()
    except OSError as exc:
        return f"ERROR: {exc}"

    p = relative_to_project(script)
    arguments = arguments or []
    if any(not isinstance(argument, str) for argument in arguments):
        return "ERROR: Script arguments must be strings."
    if p == "src/validate_litho_contact_pipeline.py":
        if len(arguments) != 2 or arguments[0] != "--report":
            return "ERROR: Validator requires --report <new figures/audit/*.json path>."
        try:
            report_path = safe_path(arguments[1])
        except ValueError as exc:
            return f"ERROR: {exc}"
        if (
            report_path.parent != safe_path("figures/audit")
            or report_path.suffix.lower() != ".json"
            or report_path.exists()
        ):
            return "ERROR: Validator report must be a new .json directly under figures/audit/."
    elif arguments:
        return f"ERROR: Arguments are not permitted for {p}."

    if timeout < 1:
        return "ERROR: Timeout must be a positive number of seconds."
    timeout = min(timeout, MAX_SCRIPT_TIMEOUT_SECONDS)
    intact, report = _execution_integrity_gate()
    if not intact:
        return report

    snapshot_note = maybe_snapshot(f"exec-{Path(p).stem}")
    result = run_subprocess(
        [python, str(script), *arguments],
        timeout=timeout,
        env=_python_environment(),
    )
    still_intact, after_report = _frozen_integrity_report()
    suffix = f"\n{snapshot_note}"
    if not still_intact:
        suffix += f"\n\nFROZEN INTEGRITY CHECK FAILED AFTER EXECUTION:\n{after_report}"
    return result + suffix


# =============================================================================
# 17. RUN PYTEST (hardened)
# =============================================================================

PYTEST_ALLOWED_FLAGS: frozenset[str] = frozenset({
    "-x", "-v", "-q", "-s",
    "--no-header", "--no-summary",
    "--tb=short", "--tb=long", "--tb=line",
    "-k", "--collect-only",
})


@mcp.tool()
@audited
def run_pytest(
    path: str = "",
    flags: list[str] | None = None,
    timeout: int = MAX_PYTEST_TIMEOUT_SECONDS,
) -> str:
    """
    Run one named, reviewed pytest file.

    Only a curated set of flags is accepted. Plugin autoload is disabled and
    the subprocess output and duration are bounded.
    """
    if not path:
        return "ERROR: Specify one approved test file."
    err = guard_test_path(path)
    if err:
        return err

    try:
        python = _project_python()
    except OSError as exc:
        return f"ERROR: {exc}"
    command = [python, "-m", "pytest"]
    try:
        command.append(str(safe_path(path)))
    except ValueError as exc:
        return f"ERROR: {exc}"

    if flags:
        if len(flags) > 10 or any(not isinstance(flag, str) or len(flag) > 200 for flag in flags):
            return "ERROR: Pytest flags are invalid or exceed the argument limit."
        i = 0
        while i < len(flags):
            f = flags[i]
            head = f.split("=", 1)[0]
            if head not in PYTEST_ALLOWED_FLAGS:
                return f"ERROR: pytest flag '{f}' is not permitted."
            command.append(f)
            if f == "-k":
                if i + 1 >= len(flags):
                    return "ERROR: -k requires an expression."
                if flags[i + 1].startswith("-"):
                    return "ERROR: -k expression cannot be another option."
                command.append(flags[i + 1])
                i += 1
            i += 1

    if timeout < 1:
        return "ERROR: Timeout must be a positive number of seconds."
    timeout = min(timeout, MAX_PYTEST_TIMEOUT_SECONDS)
    intact, report = _execution_integrity_gate()
    if not intact:
        return report

    snapshot_note = maybe_snapshot("pytest")
    result = run_subprocess(
        command,
        timeout=timeout,
        env=_python_environment(for_pytest=True),
    )
    still_intact, after_report = _frozen_integrity_report()
    suffix = f"\n{snapshot_note}"
    if not still_intact:
        suffix += f"\n\nFROZEN INTEGRITY CHECK FAILED AFTER TESTS:\n{after_report}"
    return result + suffix


# =============================================================================
# 18. LIST AUDIT OUTPUTS
# =============================================================================

@mcp.tool()
@audited
def list_audit_outputs(pattern: str = "*") -> str:
    """List generated audit outputs under figures/audit."""
    audit_dir = safe_path("figures/audit")
    if not audit_dir.exists():
        return "Audit directory does not exist."
    results = []
    for item in sorted(audit_dir.glob(pattern)):
        rel = item.relative_to(PROJECT_ROOT)
        suffix = "/" if item.is_dir() else ""
        results.append(f"{rel}{suffix}")
    return "\n".join(results) if results else "No audit outputs found."


# =============================================================================
# 19. READ AUDIT OUTPUT
# =============================================================================

@mcp.tool()
@audited
def read_audit_output(relative_path: str, max_chars: int = 200000) -> str:
    """Read a generated audit output from figures/audit."""
    audit_root = safe_path("figures/audit")
    try:
        target = safe_path(str(Path("figures/audit") / relative_path))
    except ValueError as exc:
        return f"ERROR: {exc}"

    if target != audit_root and audit_root not in target.parents:
        return "ERROR: File is outside figures/audit."
    if not target.exists():
        return f"ERROR: Audit file does not exist: {relative_path}"
    if not target.is_file():
        return "ERROR: Not a file."
    max_chars = max(1, min(max_chars, 200000))

    try:
        text = target.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"ERROR: {exc}"
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n--- FILE TRUNCATED ---"
    return (
        f"--- BEGIN UNTRUSTED PROJECT CONTENT: "
        f"{relative_to_project(target)} ---\n{text}\n"
        f"--- END UNTRUSTED PROJECT CONTENT ---"
    )


# =============================================================================
# 20. WRITE AUDIT OUTPUT
# =============================================================================

@mcp.tool()
@audited
def write_audit_output(
    relative_path: str,
    content: str,
    overwrite: bool = True,
) -> str:
    """Write a text audit output strictly inside figures/audit."""
    if len(content.encode("utf-8")) > 1_000_000:
        return "ERROR: Audit content exceeds the 1 MB write limit."
    audit_root = safe_path("figures/audit")
    full_rel = str(Path("figures/audit") / relative_path)

    err = guard_write(full_rel)
    if err:
        return err

    try:
        target = safe_path(full_rel)
    except ValueError as exc:
        return f"ERROR: {exc}"

    if target == audit_root or audit_root not in target.parents:
        return "ERROR: Destination is outside figures/audit."
    if target.exists() and target.is_dir():
        return "ERROR: Target is a directory."
    if target.exists() and not overwrite:
        return f"ERROR: File already exists: {relative_path}"

    snapshot_note = maybe_snapshot("audit-output")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8", newline="")
        return (
            f"SUCCESS: Wrote audit output {relative_to_project(target)}\n"
            f"Bytes: {target.stat().st_size}\n{snapshot_note}"
        )
    except OSError as exc:
        return f"ERROR writing audit output: {exc}"


# =============================================================================
# 21. FIND PROJECT REFERENCES
# =============================================================================

@mcp.tool()
@audited
def find_project_references(query: str, max_results: int = 200) -> str:
    """Search major research/project directories for a term."""
    directories = [
        "src", "scripts", "tests", "docs", "figures/audit",
        "paper_1/manuscript", "paper_1/stages",
    ]
    max_results = max(1, min(max_results, 500))
    results = []
    for directory in directories:
        target = safe_path(directory)
        if not target.exists():
            continue
        for file in target.rglob("*"):
            if not file.is_file() or _is_internal_path(file):
                continue
            if file.suffix.lower() not in {
                ".py", ".md", ".txt", ".csv", ".json", ".yaml", ".yml", ".log",
                ".tex", ".bib",
            }:
                continue
            try:
                text = file.resolve().read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for n, line in enumerate(text.splitlines(), start=1):
                if query.lower() in line.lower():
                    results.append(
                        f"{file.relative_to(PROJECT_ROOT)}:{n}: {line.strip()[:300]}"
                    )
                    if len(results) >= max_results:
                        return (
                            "\n".join(results)
                            + f"\n\n--- RESULT LIMIT ({max_results}) REACHED ---"
                        )
    return "\n".join(results) if results else "No references found."


# =============================================================================
# 22. CREATE DIRECTORY
# =============================================================================

@mcp.tool()
@audited
def create_project_directory(relative_path: str) -> str:
    """Create a directory inside the project (guarded)."""
    err = guard_write(relative_path)
    if err:
        return err
    try:
        target = safe_path(relative_path)
    except ValueError as exc:
        return f"ERROR: {exc}"
    if target == PROJECT_ROOT:
        return "ERROR: Project root already exists."
    snapshot_note = maybe_snapshot("mkdir")
    try:
        target.mkdir(parents=True, exist_ok=True)
        return (
            f"SUCCESS: Directory ready: {relative_to_project(target)}\n"
            f"{snapshot_note}"
        )
    except OSError as exc:
        return f"ERROR creating directory: {exc}"


# =============================================================================
# 23. SNAPSHOT WORKSPACE (manual)
# =============================================================================

@mcp.tool()
@audited
def snapshot_workspace(label: str = "manual") -> str:
    """
    Take a recovery snapshot now, bypassing the debounce.

    Mutating tools already do this automatically; call this when you want an
    explicit recovery point before a risky operation.
    """
    global _LAST_SNAPSHOT_TS
    _LAST_SNAPSHOT_TS = 0.0   # force
    label = (label or "manual").strip().replace(" ", "_")[:60]
    return maybe_snapshot(label)


# =============================================================================
# 24. LIST RECENT OUTPUTS
# =============================================================================

@mcp.tool()
@audited
def list_recent_outputs(minutes: int = 30) -> str:
    """
    List files under figures/audit/, docs/, src/, scripts/ modified in the
    last N minutes. Use after a regeneration pass to confirm every dependent
    output was refreshed.
    """
    cutoff = time.time() - max(1, minutes) * 60
    lines = [f"Files modified in the last {minutes} minute(s):", ""]
    any_hits = False
    for root in ("figures/audit", "docs", "src", "scripts"):
        base = safe_path(root)
        if not base.exists():
            continue
        for file in sorted(base.rglob("*")):
            if file.is_file() and file.stat().st_mtime >= cutoff:
                age = int(time.time() - file.stat().st_mtime)
                lines.append(f"  {file.relative_to(PROJECT_ROOT)}  ({age}s ago)")
                any_hits = True
    return "\n".join(lines) if any_hits else "No recent modifications."


# =============================================================================
# STARTUP CHECKS
# =============================================================================

def _verify_trash_gitignored() -> str | None:
    """Warn if .trash/ is not gitignored. Returns warning string or None."""
    gi = PROJECT_ROOT / ".gitignore"
    if not gi.exists():
        return (
            "WARN: .gitignore does not exist. Add '.trash/' so archived files "
            "do not appear as untracked."
        )
    try:
        text = gi.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"WARN: could not read .gitignore: {exc}"
    for line in text.splitlines():
        s = line.strip()
        if s in {".trash", ".trash/", "/.trash", "/.trash/", "**/.trash/"}:
            return None
    return (
        "WARN: '.trash/' is not listed in .gitignore. Add it so archived "
        "files do not appear as untracked."
    )


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)

    baseline = _load_frozen_baseline()
    present_count = (
        sum(
            1
            for entry in baseline["files"].values()
            if isinstance(entry, dict) and entry.get("exists") is True
        )
        if baseline is not None
        else 0
    )

    warn = _verify_trash_gitignored()

    startup_lines = [
        "Copperbelt MCP server starting on http://127.0.0.1:8000/mcp",
        f"  Root:                {PROJECT_ROOT}",
        f"  Frozen files:        {sorted(FROZEN_FILES)}",
        f"  Frozen prefixes:     {list(FROZEN_PREFIXES)}",
        f"  Writable prefixes:   {list(WRITABLE_PREFIXES)}",
        f"  Allowlisted scripts: {len(ALLOWLISTED_SCRIPTS)}",
        f"  Audit log:           {AUDIT_LOG}",
        f"  Frozen baseline:     {FROZEN_BASELINE} "
        f"({present_count}/{len(FROZEN_FILES)} marked present; unchanged)",
        f"  Snapshot debounce:   {_SNAPSHOT_DEBOUNCE_SECONDS}s",
    ]
    if baseline is None:
        startup_lines.append(
            "  ERROR: Existing frozen baseline is missing or invalid; execution is disabled."
        )
    startup_lines.append(
        "  NOTE: Baseline paths reference .nc traces; any .pkl trace files are not covered."
    )
    if warn:
        startup_lines.append("  " + warn)
    print("\n".join(startup_lines), flush=True)

    mcp.run(
        transport="streamable-http",
        host="127.0.0.1",
        port=8000,
    )