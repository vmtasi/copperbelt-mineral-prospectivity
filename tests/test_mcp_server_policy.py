from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVER_PATH = ROOT / ".mcp" / "server1.py"
SPEC = importlib.util.spec_from_file_location("copperbelt_mcp_server_policy", SERVER_PATH)
assert SPEC is not None and SPEC.loader is not None
server = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = server
SPEC.loader.exec_module(server)


def test_safe_path_rejects_absolute_paths_and_project_escape():
    for path in (r"C:\outside\file.txt", r"\outside\file.txt", r"..\outside\file.txt"):
        try:
            server.safe_path(path)
        except ValueError:
            continue
        raise AssertionError(f"unsafe path accepted: {path}")


def test_write_guard_checks_resolved_path_and_internal_subtrees():
    assert server.guard_write("src/new_diagnostic.py") is None
    assert server.guard_write("src/../paper_1/manuscript/main.tex")
    assert server.guard_write("src/../AI_PROJECT_CONTEXT.md")
    assert server.guard_write(".git/config")
    assert server.guard_write(".trash/old.txt")


def test_execution_and_test_allowlists_are_exact():
    assert server.guard_exec("src/compute_fold_counts.py") is None
    assert server.guard_exec("src/compare_validation_strategies.py")
    assert server.guard_exec("src/rebuild_litho_contact_features.py")
    assert server.guard_exec("src/build_corrected_training_table.py")
    assert server.guard_test_path("tests/test_mcp_server_policy.py") is None
    assert server.guard_test_path("tests/../src/validation_strategies.py")
    assert server.guard_test_path("tests")


def test_subprocess_output_is_bounded():
    command = [
        sys.executable,
        "-c",
        "import sys; sys.stdout.write('x' * 4096); sys.stderr.write('y' * 4096)",
    ]
    output = server.run_subprocess(command, timeout=10, output_limit=128)
    assert "--- OUTPUT TRUNCATED AT 128 BYTES ---" in output
    assert output.count("x") + output.count("y") <= 128


def test_subprocess_timeout_is_reported():
    output = server.run_subprocess(
        [sys.executable, "-c", "import time; time.sleep(5)"],
        timeout=1,
    )
    assert "ERROR: Command timed out after 1 seconds." in output


def test_project_python_runs_with_restricted_environment():
    output = server.run_subprocess(
        [server._project_python(), "-c", "print('project-python-ok')"],
        timeout=10,
        env=server._python_environment(),
    )
    assert "project-python-ok" in output
    assert "--- EXIT CODE: 0 ---" in output


def test_audit_records_redact_content_and_search_terms():
    def sample_tool(content: str, query: str, relative_path: str):
        return None

    secret = "never-write-this-secret"
    record = server._audit_record(
        sample_tool,
        (),
        {
            "content": secret,
            "query": secret,
            "relative_path": "figures/audit/report.json",
        },
        time.monotonic(),
        status="ok",
    )
    encoded = json.dumps(record)
    assert secret not in encoded
    assert record["arguments"]["relative_path"] == "figures/audit/report.json"
    assert record["duration_ms"] >= 0


def test_audit_marks_nonzero_exit_as_error():
    assert server._audit_result_status("failure\n--- EXIT CODE: 3 ---") == "error"
    assert server._audit_result_status("ERROR: Command timed out") == "error"


def test_missing_baseline_is_reported_without_creating_one(tmp_path, monkeypatch):
    baseline = tmp_path / "frozen_baseline.json"
    monkeypatch.setattr(server, "FROZEN_BASELINE", baseline)
    intact, report = server._frozen_integrity_report()
    assert not intact
    assert "missing or invalid" in report
    assert not baseline.exists()


def test_baseline_detects_creation_of_previously_absent_frozen_file(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(server, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(server, "FROZEN_FILES", frozenset({"frozen.txt"}))
    baseline = tmp_path / "frozen_baseline.json"
    monkeypatch.setattr(server, "FROZEN_BASELINE", baseline)
    baseline.write_text(
        json.dumps({
            "project_root": str(tmp_path),
            "files": {"frozen.txt": {"exists": False, "sha256": None}},
        }),
        encoding="utf-8",
    )

    intact, _ = server._frozen_integrity_report()
    assert intact
    (tmp_path / "frozen.txt").write_text("new content", encoding="utf-8")
    intact, report = server._frozen_integrity_report()
    assert not intact
    assert "CREATED SINCE BASELINE" in report


def test_snapshot_does_not_sweep_untracked_files(monkeypatch):
    calls = []

    def fake_git(*args, **kwargs):
        calls.append(args)
        if args == ("status", "--porcelain", "--untracked-files=no"):
            return subprocess.CompletedProcess(args, 0, stdout=" M README.md\n", stderr="")
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(server, "_run_git", fake_git)
    monkeypatch.setattr(server, "_LAST_SNAPSHOT_TS", 0.0)
    server.maybe_snapshot("policy-test")
    push_call = next(call for call in calls if call[:2] == ("stash", "push"))
    assert "-u" not in push_call


def test_baseline_recapture_is_refused():
    result = server.verify_frozen_intact.__wrapped__(recapture=True)
    assert result.startswith("ERROR:")
    assert "disabled" in result
