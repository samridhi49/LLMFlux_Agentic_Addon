"""Explicit configuration for a single backend agent task."""

import math
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit


@dataclass(frozen=True)
class AgentConfig:
    """Use a served model identifier, not a models.yaml catalog key.

    The deadline is cooperative: checked between operations. HTTP timeouts
    bound socket waits; they do not cancel inference already running in vLLM.
    """

    model: str
    task_root: Path
    base_url: str = "http://127.0.0.1:8000/v1"
    allowed_tools: tuple[str, ...] = ("convert_csv_to_json",)
    max_turns: int = 6
    max_tool_calls: int = 5
    max_output_tokens: int = 2048
    request_timeout: float = 120.0
    task_timeout: float = 300.0
    max_input_bytes: int = 1_000_000

    def __post_init__(self):
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model must be a nonempty served model identifier")
        root = Path(self.task_root).expanduser().resolve(strict=True)
        if not root.is_dir():
            raise ValueError("task_root must be an existing directory")
        object.__setattr__(self, "task_root", root)
        url = urlsplit(self.base_url)
        if (url.scheme not in ("http", "https") or not url.hostname
                or url.username or url.password or url.query or url.fragment
                or url.path.rstrip("/") not in ("", "/v1")):
            raise ValueError("base_url must be an HTTP(S) server URL, optionally ending in /v1")
        base = self.base_url.rstrip("/")
        object.__setattr__(self, "base_url", base if base.endswith("/v1") else base + "/v1")
        for name in ("max_turns", "max_tool_calls", "max_output_tokens", "max_input_bytes"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("request_timeout", "task_timeout"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be a finite positive number")
        if isinstance(self.allowed_tools, str):
            raise ValueError("allowed_tools must be a sequence of names")
        names = tuple(self.allowed_tools)
        if any(not isinstance(name, str) or not name.strip() for name in names):
            raise ValueError("tool names must be nonempty strings")
        if len(names) != len(set(names)):
            raise ValueError("allowed_tools must not contain duplicates")
        object.__setattr__(self, "allowed_tools", names)
