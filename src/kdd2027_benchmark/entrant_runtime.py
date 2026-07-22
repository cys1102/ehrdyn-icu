from __future__ import annotations

import json
import os
import resource
import selectors
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import ReleaseContractError
from .schema import validate_json_document


PROTOCOL_VERSION = "kdd215.runtime.v1"


@dataclass(frozen=True)
class ResourceContract:
    timeout_seconds: float = 15.0
    memory_bytes: int = 2_147_483_648
    cpu_seconds: int = 120
    max_output_bytes: int = 32_000_000


def load_entrant(path: Path, schema_name: str = "entrant_protocol") -> dict[str, Any]:
    value = validate_json_document(path, schema_name)
    return dict(value)


def _limits(contract: ResourceContract) -> None:
    resource.setrlimit(resource.RLIMIT_AS, (contract.memory_bytes, contract.memory_bytes))
    resource.setrlimit(resource.RLIMIT_CPU, (contract.cpu_seconds, contract.cpu_seconds))
    resource.setrlimit(resource.RLIMIT_FSIZE, (contract.max_output_bytes, contract.max_output_bytes))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))


class IsolatedEntrant:
    """Persistent JSONL subprocess with a read-only, network-free bwrap sandbox."""

    def __init__(self, declaration_path: Path, contract: ResourceContract | None = None,
                 declaration_schema: str = "entrant_protocol",
                 protocol_version: str = PROTOCOL_VERSION):
        self.declaration_path = declaration_path.resolve()
        self.declaration = load_entrant(self.declaration_path, declaration_schema)
        self.contract = contract or ResourceContract()
        self.protocol_version = protocol_version
        self.process: subprocess.Popen[str] | None = None
        self.stderr = ""

    def __enter__(self) -> "IsolatedEntrant":
        entrant_dir = self.declaration_path.parent
        command = [str(value) for value in self.declaration["command"]]
        if command[0].startswith("./"):
            command[0] = "/entrant/" + command[0][2:]
        bwrap = shutil.which("bwrap")
        if bwrap:
            command = [
                bwrap, "--die-with-parent", "--unshare-net", "--new-session",
                "--ro-bind", "/usr", "/usr", "--ro-bind", "/lib", "/lib",
                "--ro-bind-try", "/lib64", "/lib64", "--proc", "/proc",
                "--dev", "/dev", "--tmpfs", "/tmp", "--dir", "/work",
                "--ro-bind", str(entrant_dir), "/entrant", "--chdir", "/work",
                *command,
            ]
        elif os.environ.get("EHRDYN_ALLOW_UNSANDBOXED_ENTRANT") != "1":
            raise ReleaseContractError("bwrap is required for entrant execution")
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
            env={"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": "0", "LANG": "C.UTF-8"},
            preexec_fn=lambda: _limits(self.contract),
        )
        return self

    def request(self, operation: str, payload: dict[str, Any], seed: int) -> dict[str, Any]:
        if self.process is None or self.process.stdin is None or self.process.stdout is None:
            raise ReleaseContractError("entrant process is not running")
        message = {"protocol_version": self.protocol_version, "operation": operation, "seed": int(seed), "payload": payload}
        encoded = json.dumps(message, sort_keys=True, separators=(",", ":"), allow_nan=False)
        started = time.monotonic()
        try:
            self.process.stdin.write(encoded + "\n")
            self.process.stdin.flush()
        except BrokenPipeError as error:
            raise ReleaseContractError("entrant_crashed_before_response") from error
        selector = selectors.DefaultSelector()
        selector.register(self.process.stdout, selectors.EVENT_READ)
        if not selector.select(self.contract.timeout_seconds):
            self.process.kill()
            raise ReleaseContractError("entrant_timeout")
        line = self.process.stdout.readline()
        elapsed = time.monotonic() - started
        if not line:
            raise ReleaseContractError("entrant_crashed_or_empty_response")
        if len(line.encode()) > self.contract.max_output_bytes:
            raise ReleaseContractError("entrant_output_limit")
        try:
            response = json.loads(line)
        except json.JSONDecodeError as error:
            raise ReleaseContractError("entrant_malformed_json") from error
        if not isinstance(response, dict) or response.get("protocol_version") != self.protocol_version:
            raise ReleaseContractError("entrant_protocol_version_mismatch")
        if response.get("status") != "ok" or not isinstance(response.get("result"), dict):
            raise ReleaseContractError(f"entrant_failure:{response.get('error', 'unspecified')}")
        response["latency_seconds"] = elapsed
        return response

    def __exit__(self, *_: object) -> None:
        if self.process is None:
            return
        if self.process.stdin is not None:
            try:
                self.process.stdin.close()
            except BrokenPipeError:
                pass
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.process.kill()
        if self.process.stderr is not None:
            self.stderr = self.process.stderr.read()[:16_384]
            self.process.stderr.close()
        if self.process.stdout is not None:
            self.process.stdout.close()


def validate_policy_result(result: dict[str, Any], rows: int, action_count: int,
                           supported: list[int], tolerance: float = 1e-8) -> list[list[float]]:
    probabilities = result.get("probabilities")
    if not isinstance(probabilities, list) or len(probabilities) != rows:
        raise ReleaseContractError("entrant_action_dimension_failure")
    support = set(supported)
    normalized: list[list[float]] = []
    for row in probabilities:
        if not isinstance(row, list) or len(row) != action_count:
            raise ReleaseContractError("entrant_action_dimension_failure")
        values = []
        for index, value in enumerate(row):
            if not isinstance(value, (int, float)) or not __import__("math").isfinite(value):
                raise ReleaseContractError("entrant_nonfinite_probability")
            if value < 0:
                raise ReleaseContractError("entrant_negative_probability")
            if index not in support and value > tolerance:
                raise ReleaseContractError("entrant_unsupported_probability")
            values.append(float(value))
        if abs(sum(values) - 1.0) > tolerance:
            raise ReleaseContractError("entrant_probability_normalization_failure")
        normalized.append(values)
    return normalized
