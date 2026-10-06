"""Local Python tools. Tool arguments never control the task workspace."""

import csv
import io
import json
from copy import deepcopy
from pathlib import Path

from .config import AgentConfig


def _task_path(root: Path, value: str, directory: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("paths must be nonempty strings")
    relative = Path(value)
    if (not relative.parts or relative.is_absolute() or ".." in relative.parts
            or relative.parts[0] != directory):
        raise ValueError(f"path must be relative and inside {directory}/")
    # Reject all symlinks, including a symlinked input/output directory.
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("symlinks are not permitted in tool paths")
    resolved = current.resolve()
    if not resolved.is_relative_to(root / directory):
        raise ValueError("path escapes permitted directory")
    return resolved


def convert_csv_to_json(input_path: str, output_path: str, task_root: Path,
                        max_input_bytes: int = 1_000_000) -> dict:
    """Convert UTF-8 CSV to JSON, preserving strings and refusing overwrites.

    input/ and output/ must already exist in a private, caller-owned workspace.
    Header-only CSV produces []; missing/duplicate headers and ragged rows fail.
    This is path validation, not a sandbox against concurrent filesystem changes.
    """
    root = Path(task_root).resolve(strict=True)
    if type(max_input_bytes) is not int or max_input_bytes < 1:
        raise ValueError("max_input_bytes must be positive")
    source = _task_path(root, input_path, "input")
    destination = _task_path(root, output_path, "output")
    if not source.is_file():
        raise ValueError("input must be an existing regular file")
    if not destination.parent.is_dir():
        raise ValueError("output parent directory must already exist")
    with source.open("rb") as handle:
        data = handle.read(max_input_bytes + 1)
    if len(data) > max_input_bytes:
        raise ValueError("input exceeds max_input_bytes")
    reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig"), newline=""), strict=True)
    headers = reader.fieldnames
    if not headers or any(not name.strip() for name in headers) or len(set(headers)) != len(headers):
        raise ValueError("CSV requires nonempty, unique headers")
    records = []
    for row in reader:
        if None in row or any(value is None for value in row.values()):
            raise ValueError("CSV row length does not match header")
        records.append(row)
    serialized = json.dumps(records, ensure_ascii=False, indent=2)
    with destination.open("x", encoding="utf-8") as handle:
        handle.write(serialized + "\n")
    return {"status": "success", "artifact": destination.relative_to(root).as_posix(),
            "record_count": len(records)}


_CONVERSION_SCHEMA = {
    "type": "function",
    "name": "convert_csv_to_json",
    "description": "Convert UTF-8 CSV in input/ to JSON in output/. Return artifact path and record count. Values remain strings; output must not exist.",
    "parameters": {
        "type": "object",
        "properties": {"input_path": {"type": "string"}, "output_path": {"type": "string"}},
        "required": ["input_path", "output_path"],
        "additionalProperties": False,
    },
}


class ToolRegistry:
    """An explicit allowlist; no shell, eval, dynamic imports, or MCP."""

    def __init__(self, config: AgentConfig):
        self.config = config
        unknown = set(config.allowed_tools) - {"convert_csv_to_json"}
        if unknown:
            raise ValueError(f"Unknown configured tools: {sorted(unknown)}")

    def schemas(self) -> list[dict]:
        return [deepcopy(_CONVERSION_SCHEMA)] if "convert_csv_to_json" in self.config.allowed_tools else []

    def execute(self, name: str, arguments: str) -> dict:
        """Return a JSON-serializable result, including recoverable tool errors."""
        try:
            if name not in self.config.allowed_tools:
                raise ValueError("tool is unknown or not permitted")
            if not isinstance(arguments, str):
                raise ValueError("arguments must be a JSON string")
            values = json.loads(arguments)
            if not isinstance(values, dict) or set(values) != {"input_path", "output_path"}:
                raise ValueError("expected exactly input_path and output_path")
            return convert_csv_to_json(**values, task_root=self.config.task_root,
                                       max_input_bytes=self.config.max_input_bytes)
        except (ValueError, TypeError, OSError, csv.Error) as exc:
            return {"status": "error", "error": {"type": type(exc).__name__, "message": str(exc)}}
