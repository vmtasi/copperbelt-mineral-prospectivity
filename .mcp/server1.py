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
#    it. verify_frozen_intact() provides after-the-fact detection against a
#    session-start SHA-256 baseline stored outside the project.
#
# 3. Every mutation and every session-script execution is preceded by a
#    debounced git-stash snapshot, so any damage is recoverable.
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
import json
import os
import shutil
import subprocess
import sys
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
)


# -----------------------------------------------------------------------------
# Curated execution allowlist.
#
# Existing analysis scripts the agent may execute by name.
# v11_spatial_stability_final.py is intentionally EXCLUDED: it is frozen, and
# running it would overwrite the four V11 trace files.
# -----------------------------------------------------------------------------

ALLOWLISTED_SCRIPTS: frozenset[str] = frozenset({
    "src/phase5_response_curves.py",
    "src/extract_v11_oof.py",
    "src/v11b_spatial_support_audit.py",
    "src/feature_x_dom_x_target.py",
    "src/phase6_audit.py",
    "src/phase7_spatial_validation.py",
    "src/phase7_final_validation.py",
    "src/phase7_multiscale_bootstrap.py",
    "src/phase7_6_sensitivity_analysis.py",
    "src/v11b_model_validation_upgraded.py",
    "src/hierachical_spatial_cv.py",
    "src/phase_4b.py",
    "src/phase3_ablation_suite.py",
    "src/baseline.py",
    "src/compute_fold_counts.py",
    "src/plot_mining_locations_folds.py",
    "src/compare_validation_strategies.py",
    "src/phase4_tectonic_heterogeneity.py",
    "src/ablation_suite.py",
    "src/v9_posterior_extraction.py",
    "src/v10_spatail_heterogeneity_upgraded.py",
})

# Session-scoped set populated by write_project_file / replace_in_project_file.
# Cleared on server restart.
SESSION_SCRIPTS: set[str] = set()


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
        "frozen. Script execution is restricted to a curated allowlist of "
        "existing analysis scripts, plus scripts written in this session. "
        "IMPORTANT: script execution is a trusted operation — a running "
        "Python process can bypass all MCP-level path restrictions. "
        "Call verify_frozen_intact() to detect frozen-file drift after any "
        "script execution. Prefer the narrow tools over broad ones. "
        "Every mutation is preceded by an automatic git-stash snapshot."
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
    target = (PROJECT_ROOT / relative_path).resolve()
    if target != PROJECT_ROOT and PROJECT_ROOT not in target.parents:
        raise ValueError("Requested path is outside the Copperbelt project.")
    return target


def relative_to_project(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT)).replace("\\", "/")


def guard_write(relative_path: str) -> str | None:
    p = _norm(relative_path)

    if not p:
        return "ERROR: Empty relative path."

    if p in FROZEN_FILES:
        return (
            f"ERROR: '{relative_path}' is frozen. "
            "It must be regenerated by its source pipeline, not edited. "
            "If a genuine bug is suspected, report it — do not modify."
        )

    for prefix in FROZEN_PREFIXES:
        if p.startswith(prefix):
            return f"ERROR: '{relative_path}' is inside a frozen subtree ({prefix})."

    if not any(p.startswith(prefix) for prefix in WRITABLE_PREFIXES):
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
    p = _norm(relative_path)

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

    if p in SESSION_SCRIPTS:
        return None

    return (
        f"ERROR: '{relative_path}' is not an approved executable. "
        "Allowed: scripts in ALLOWLISTED_SCRIPTS, or scripts written via "
        "write_project_file / replace_in_project_file during this session. "
        "Use list_executables() to see the current set."
    )


def guard_test_path(relative_path: str) -> str | None:
    """Prefix-based check that a pytest target actually lives under tests/."""
    p = _norm(relative_path).rstrip("/")
    if p == "tests" or p.startswith("tests/"):
        return None
    return f"ERROR: Path '{relative_path}' is not under tests/."


# =============================================================================
# SUBPROCESS
# =============================================================================

def run_subprocess(command: list[str], timeout: int = 120) -> str:
    try:
        result = subprocess.run(
            command, cwd=PROJECT_ROOT, capture_output=True, text=True,
            timeout=timeout, encoding="utf-8", errors="replace",
        )
        output = result.stdout
        if result.stderr:
            output += "\n--- STDERR ---\n" + result.stderr
        output += f"\n--- EXIT CODE: {result.returncode} ---"
        return output.strip()
    except subprocess.TimeoutExpired:
        return f"ERROR: Command timed out after {timeout} seconds."
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"


# =============================================================================
# SNAPSHOT (auto-invoked before mutations and session-script executions)
# =============================================================================

def _run_git(*args: str, timeout: int = 30) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=PROJECT_ROOT, capture_output=True,
        text=True, timeout=timeout, encoding="utf-8", errors="replace",
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

    status = _run_git("status", "--porcelain")
    if status.returncode != 0:
        return f"(WARN snapshot: git status failed: {status.stderr.strip()[:120]})"

    if not status.stdout.strip():
        _LAST_SNAPSHOT_TS = now
        return "(tree clean; HEAD is the recovery point)"

    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    msg = f"mcp-snapshot-{ts}-{reason}"[:80]

    push = _run_git("stash", "push", "-u", "-m", msg)
    if push.returncode != 0:
        return f"(WARN snapshot: stash push failed: {push.stderr.strip()[:160]})"

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
# run_python_script() and run_pytest() can modify any file the user can write,
# including the frozen artifacts. The MCP layer cannot prevent this. It can
# hash the frozen files at the start of every session and let the agent check
# at any later point whether they have changed.
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


def _capture_frozen_baseline() -> dict:
    """Hash every frozen file that currently exists on disk."""
    entries: dict[str, dict] = {}
    for rel in sorted(FROZEN_FILES):
        try:
            target = safe_path(rel)
        except ValueError:
            entries[rel] = {"exists": False, "sha256": None}
            continue
        entries[rel] = {
            "exists": target.exists(),
            "sha256": _hash_file(target) if target.exists() else None,
        }
    return {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "files": entries,
    }


def _write_frozen_baseline(snapshot: dict) -> None:
    FROZEN_BASELINE.parent.mkdir(parents=True, exist_ok=True)
    FROZEN_BASELINE.write_text(
        json.dumps(snapshot, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _load_frozen_baseline() -> dict | None:
    if not FROZEN_BASELINE.exists():
        return None
    try:
        return json.loads(FROZEN_BASELINE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


# =============================================================================
# AUDIT DECORATOR
# =============================================================================

def audited(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        result = fn(*args, **kwargs)
        try:
            AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
            args_summary = [str(a)[:120] for a in args]
            args_summary += [f"{k}={str(v)[:120]}" for k, v in kwargs.items()]
            record = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "tool": fn.__name__,
                "args": args_summary,
                "args_hash": hashlib.sha256(
                    json.dumps(args_summary, default=str).encode()
                ).hexdigest()[:16],
                "status": (
                    "error"
                    if isinstance(result, str) and result.startswith("ERROR")
                    else "ok"
                ),
                "summary": (
                    result[:200] if isinstance(result, str) else str(result)[:200]
                ),
            }
            with AUDIT_LOG.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            pass
        return result
    return wrapper


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
        f"Python: {sys.executable}\n"
        f"Audit log: {AUDIT_LOG}\n"
        f"Frozen baseline: {FROZEN_BASELINE}\n"
        f"Frozen files: {sorted(FROZEN_FILES)}\n"
        f"Frozen prefixes: {list(FROZEN_PREFIXES)}\n"
        f"Writable prefixes: {list(WRITABLE_PREFIXES)}\n"
        f"Allowlisted executables: {len(ALLOWLISTED_SCRIPTS)}\n"
        f"Session-written executables: {len(SESSION_SCRIPTS)}\n"
    )


# =============================================================================
# 2. LIST EXECUTABLES
# =============================================================================

@mcp.tool()
@audited
def list_executables() -> str:
    """List the scripts the agent is currently permitted to execute."""
    lines = ["ALLOWLISTED (existing analysis scripts):"]
    for p in sorted(ALLOWLISTED_SCRIPTS):
        lines.append(f"  {p}")
    lines.append("")
    lines.append("SESSION-WRITTEN (created via write_project_file this session):")
    if SESSION_SCRIPTS:
        for p in sorted(SESSION_SCRIPTS):
            lines.append(f"  {p}")
    else:
        lines.append("  (none)")
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
    Compare current frozen-file contents against the session baseline.

    - recapture=False (default): report drift since the session baseline.
    - recapture=True: overwrite the baseline with the current state. Use ONLY
      at the start of a session, before any script execution.

    This does not prevent modification. It detects it after the fact.
    """
    if recapture:
        snapshot = _capture_frozen_baseline()
        _write_frozen_baseline(snapshot)
        return (
            f"Baseline recaptured at {snapshot['captured_at']}.\n"
            f"Files hashed: {len(snapshot['files'])}\n"
            f"Baseline: {FROZEN_BASELINE}"
        )

    baseline = _load_frozen_baseline()
    if baseline is None:
        snapshot = _capture_frozen_baseline()
        _write_frozen_baseline(snapshot)
        return (
            f"No baseline found — one has been created now "
            f"({snapshot['captured_at']}).\n"
            f"Baseline: {FROZEN_BASELINE}\n"
            "Re-run verify_frozen_intact() to check for drift."
        )

    lines = [
        f"Baseline captured: {baseline.get('captured_at', 'unknown')}",
        f"Project root:      {baseline.get('project_root', 'unknown')}",
        "",
        "FROZEN FILE STATUS:",
    ]

    drifted, missing, ok = [], [], []

    for rel in sorted(FROZEN_FILES):
        entry = baseline.get("files", {}).get(rel, {})
        try:
            target = safe_path(rel)
        except ValueError:
            missing.append(rel)
            lines.append(f"  {rel}: UNREACHABLE")
            continue

        if not target.exists():
            if entry.get("exists"):
                missing.append(rel)
                lines.append(f"  {rel}: MISSING (was present at baseline)")
            else:
                lines.append(f"  {rel}: (absent at baseline and now)")
            continue

        current = _hash_file(target)
        if current == entry.get("sha256"):
            ok.append(rel)
            lines.append(f"  {rel}: OK")
        elif entry.get("sha256") is None:
            ok.append(rel)
            lines.append(f"  {rel}: OK (was absent at baseline, now present)")
        else:
            drifted.append(rel)
            lines.append(
                f"  {rel}: DRIFTED\n"
                f"    baseline: {entry['sha256']}\n"
                f"    current:  {current}"
            )

    lines.append("")
    lines.append(
        f"Summary: {len(ok)} OK, {len(drifted)} drifted, {len(missing)} missing"
    )

    if drifted or missing:
        lines.append("")
        lines.append(
            "ACTION REQUIRED: A frozen artifact has changed. Investigate "
            "before continuing. If the change was intended, that is a "
            "model-versioning decision and must be documented in "
            "AI_PROJECT_CONTEXT.md by a human — this server will not "
            "recapture the baseline on your behalf in the same session."
        )

    return "\n".join(lines)


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
        for item in sorted(iterator):
            try:
                rel = item.relative_to(PROJECT_ROOT)
                suffix = "/" if item.is_dir() else ""
                results.append(f"{rel}{suffix}")
            except ValueError:
                continue
        return "\n".join(results) if results else "No matching files found."
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
        ".csv", ".tsv", ".log", ".ini", ".cfg", ".xml",
    }
    if target.suffix.lower() not in allowed:
        return f"ERROR: File type '{target.suffix}' is not enabled for text reading."

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
    If the file is a .py under an executable root, it is added to the
    session-scoped execution allowlist.
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

    snapshot_note = maybe_snapshot("write")

    try:
        if create_parents:
            target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8", newline="")

        p = _norm(relative_path)
        if (
            target.suffix.lower() == ".py"
            and p not in FROZEN_FILES
            and any(p.startswith(root) for root in ("src/", "scripts/", "tests/"))
        ):
            SESSION_SCRIPTS.add(p)

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

    p = _norm(relative_path)
    if target.suffix.lower() == ".py" and p in SESSION_SCRIPTS:
        SESSION_SCRIPTS.add(p)  # idempotent

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
        SESSION_SCRIPTS.discard(_norm(relative_path))
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

    matches = []
    ql = query.lower()
    try:
        for file in target.rglob(pattern):
            if not file.is_file():
                continue
            try:
                text = file.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for n, line in enumerate(text.splitlines(), start=1):
                if ql in line.lower():
                    matches.append(
                        f"{file.relative_to(PROJECT_ROOT)}:{n}: {line.strip()}"
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

    search_roots = paths or ["src", "scripts", "tests", "docs", "figures/audit"]
    exts = set(
        (e.lower() if e.startswith(".") else "." + e.lower())
        for e in (extensions or [
            ".py", ".md", ".txt", ".json", ".yaml", ".yml",
            ".toml", ".csv", ".tsv", ".rst",
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
            if not file.is_file() or file.suffix.lower() not in exts:
                continue
            rel = str(file.relative_to(PROJECT_ROOT)).replace("\\", "/")
            frozen_marker = " [FROZEN]" if _norm(rel) in FROZEN_FILES else ""
            try:
                text = file.read_text(encoding="utf-8", errors="replace")
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
        sys.executable, "-c",
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
    return run_subprocess(command)


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
    return run_subprocess([sys.executable, "-m", "py_compile", str(target)])


# =============================================================================
# 14. GIT STATUS / DIFF / LOG
# =============================================================================

@mcp.tool()
@audited
def git_status() -> str:
    """Return the current Git repository status."""
    return run_subprocess(["git", "status", "--short", "--branch"])


@mcp.tool()
@audited
def git_diff(staged: bool = False, relative_path: str = "") -> str:
    """Return the current Git diff."""
    command = ["git", "diff"]
    if staged:
        command.append("--cached")
    if relative_path:
        try:
            target = safe_path(relative_path)
        except ValueError as exc:
            return f"ERROR: {exc}"
        command.extend(["--", relative_to_project(target)])
    return run_subprocess(command)


@mcp.tool()
@audited
def git_log(count: int = 10) -> str:
    """Return recent Git commits."""
    count = max(1, min(count, 100))
    return run_subprocess(["git", "log", f"-{count}", "--oneline", "--decorate"])


# =============================================================================
# 15. GIT COMMAND (whitelisted, read-only)
# =============================================================================

GIT_READONLY_SUBCOMMANDS: frozenset[str] = frozenset({
    "status", "diff", "log", "show", "rev-parse", "ls-files", "branch", "grep",
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
        if arg == "-C" or arg.startswith("--git-dir") or arg.startswith("--work-tree"):
            return f"ERROR: '{arg}' is not permitted in git_command."
    return run_subprocess(["git", *args], timeout=timeout)


# =============================================================================
# 16. RUN PYTHON SCRIPT (allowlist-gated, snapshot-guarded)
# =============================================================================

@mcp.tool()
@audited
def run_python_script(
    relative_path: str,
    arguments: list[str] | None = None,
    timeout: int = 600,
) -> str:
    """
    Execute a Python script from the curated allowlist.

    WARNING: This is a trusted escape hatch. A running script can bypass all
    MCP-level path restrictions (frozen files, write prefixes, git state).
    The server takes a recovery snapshot before executing a session-written
    script so damage is reversible. Call verify_frozen_intact() afterwards
    to detect frozen-file drift.
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

    p = _norm(relative_path)
    snapshot_note = ""
    if p in SESSION_SCRIPTS:
        snapshot_note = "\n" + maybe_snapshot(f"exec-{Path(p).stem}")

    command = [sys.executable, str(script)]
    if arguments:
        command.extend(arguments)

    result = run_subprocess(command, timeout=timeout)
    return result + snapshot_note if snapshot_note else result


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
    timeout: int = 900,
) -> str:
    """
    Run pytest against the project's test suite.

    Only a curated set of flags is accepted (no plugin-loading flags).
    A recovery snapshot is taken before execution.
    """
    command = [sys.executable, "-m", "pytest"]

    if path:
        err = guard_test_path(path)
        if err:
            return err
        try:
            command.append(str(safe_path(path)))
        except ValueError as exc:
            return f"ERROR: {exc}"

    if flags:
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
                command.append(flags[i + 1])
                i += 1
            i += 1

    snapshot_note = maybe_snapshot("pytest")
    result = run_subprocess(command, timeout=timeout)
    return result + f"\n{snapshot_note}"


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
    directories = ["src", "scripts", "tests", "docs", "figures/audit"]
    results = []
    for directory in directories:
        target = safe_path(directory)
        if not target.exists():
            continue
        for file in target.rglob("*"):
            if not file.is_file():
                continue
            if file.suffix.lower() not in {
                ".py", ".md", ".txt", ".csv", ".json", ".yaml", ".yml", ".log",
            }:
                continue
            try:
                text = file.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for n, line in enumerate(text.splitlines(), start=1):
                if query.lower() in line.lower():
                    results.append(
                        f"{file.relative_to(PROJECT_ROOT)}:{n}: {line.strip()}"
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

    # Capture the frozen-file baseline for this session.
    baseline = _capture_frozen_baseline()
    _write_frozen_baseline(baseline)
    present_count = sum(1 for v in baseline["files"].values() if v["exists"])

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
        f"({present_count}/{len(FROZEN_FILES)} present)",
        f"  Snapshot debounce:   {_SNAPSHOT_DEBOUNCE_SECONDS}s",
    ]
    if warn:
        startup_lines.append("  " + warn)
    print("\n".join(startup_lines), flush=True)

    mcp.run(
        transport="streamable-http",
        host="127.0.0.1",
        port=8000,
    )