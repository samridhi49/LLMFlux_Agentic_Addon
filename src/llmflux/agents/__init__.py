"""Agent layer for LLMFlux.

This package holds the agentic extension to LLMFlux: the tool registry, and
(later) the tools themselves, the ReAct-style execution loop, and the
Responses-API client. Nothing here changes LLMFlux's existing batch/serve
behavior.

Public surface:
    Tool              - one tool: its model-facing schema plus the callable.
    ToolRegistry      - the catalog the agent loop talks to.
    ToolError         - base class for registry/dispatch failures.
    ToolNotFoundError - raised when dispatching an unregistered tool.
    ToolArgumentError - raised when a tool call's arguments are invalid.
"""

from .registry import (
    Tool,
    ToolRegistry,
    ToolError,
    ToolNotFoundError,
    ToolArgumentError,
)
from .tools import build_default_registry, convert_file, web_browse

__all__ = [
    "Tool",
    "ToolRegistry",
    "ToolError",
    "ToolNotFoundError",
    "ToolArgumentError",
    "build_default_registry",
    "convert_file",
    "web_browse",
]
