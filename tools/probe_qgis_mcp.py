"""Probe the registered QGIS MCP server and the running QGIS plugin."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COMMAND = ["cmd.exe", "/d", "/s", "/c", str(ROOT / "tools" / "launch_qgis_mcp.cmd")]


def send(process: subprocess.Popen[str], payload: dict) -> None:
    assert process.stdin is not None
    process.stdin.write(json.dumps(payload) + "\n")
    process.stdin.flush()


def response_for(process: subprocess.Popen[str], request_id: int) -> dict:
    assert process.stdout is not None
    while True:
        line = process.stdout.readline()
        if not line:
            assert process.stderr is not None
            server_error = process.stderr.read().strip()
            raise RuntimeError(
                "QGIS MCP server exited before returning a response"
                + (f": {server_error}" if server_error else "")
            )
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        if message.get("id") == request_id:
            return message


def main() -> None:
    process = subprocess.Popen(
        COMMAND,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        bufsize=1,
    )
    try:
        send(
            process,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "qgis-mcp-probe", "version": "1.0"},
                },
            },
        )
        initialized = response_for(process, 1)
        if "error" in initialized:
            raise RuntimeError(initialized)
        send(
            process,
            {
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
                "params": {},
            },
        )
        send(
            process,
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "ping", "arguments": {}},
            },
        )
        ping = response_for(process, 2)
        if "error" in ping:
            raise RuntimeError(ping)
        send(
            process,
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": "diagnose", "arguments": {}},
            },
        )
        diagnose = response_for(process, 3)
        if "error" in diagnose:
            raise RuntimeError(diagnose)
        print(
            json.dumps(
                {"ping": ping, "diagnose": diagnose},
                indent=2,
            )
        )
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"QGIS MCP probe failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
