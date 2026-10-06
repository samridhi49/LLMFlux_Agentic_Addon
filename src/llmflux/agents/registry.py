"""Tool registry for the LLMFlux agent layer.

The registry is the catalog the agent loop talks to. It is deliberately
**framework-agnostic**: a tool is just a name, a model-facing JSON schema, and
any Python callable. That keeps the door open to wrapping tools from anywhere
later -- a hand-written function, an ``llmflux.converters`` helper, a LangChain
tool, or an MCP client call -- without changing this file. The agent loop only
ever needs three things from the registry:

1. ``get_schemas(enabled)`` -- the list of tool schemas to send to the model
   (OpenAI / vLLM Responses-API "function" tool shape), optionally filtered to
   a user-specified allowlist.
2. ``dispatch(name, arguments)`` -- turn a model-issued tool call into a real
   function result, validating the arguments first.
3. Clear, typed errors it can catch and feed back to the model.

Argument validation here is intentionally lightweight (required-key and
unknown-key checks). Full JSON-Schema type validation can be layered on later
(e.g. via ``jsonschema``) without changing the public interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


class ToolError(Exception):
    """Base class for all tool-registry failures."""


class ToolNotFoundError(ToolError):
    """Raised when dispatching a tool name that is not registered."""


class ToolArgumentError(ToolError):
    """Raised when a tool call's arguments do not match the tool's schema."""


@dataclass
class Tool:
    """A single tool: what the model sees, plus what the code runs.

    Attributes:
        name: Unique identifier the model uses to call the tool.
        description: Plain-language explanation the model reads to decide when
            to use the tool.
        parameters: JSON Schema (an object schema) describing the tool's
            arguments. This is the ``parameters`` block of the function tool
            sent to the model.
        func: The Python callable invoked on dispatch. It is called with the
            tool-call arguments as keyword arguments.
    """

    name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    func: Optional[Callable[..., Any]] = None

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.name, str):
            raise ValueError("Tool.name must be a non-empty string")
        if not callable(self.func):
            raise ValueError(f"Tool {self.name!r} must have a callable 'func'")
        if not isinstance(self.parameters, dict):
            raise ValueError(f"Tool {self.name!r} parameters must be a dict (JSON Schema)")

    def to_schema(self) -> Dict[str, Any]:
        """Return the model-facing schema for this tool.

        Uses the flattened function-tool shape accepted by the OpenAI/vLLM
        Responses API:

            {"type": "function", "name": ..., "description": ..., "parameters": {...}}
        """
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters or {"type": "object", "properties": {}},
        }


class ToolRegistry:
    """Catalog of tools, with filtering and safe dispatch.

    Example:
        >>> registry = ToolRegistry()
        >>> registry.register_function(
        ...     name="echo",
        ...     description="Echo back the given text.",
        ...     parameters={
        ...         "type": "object",
        ...         "properties": {"text": {"type": "string"}},
        ...         "required": ["text"],
        ...     },
        ...     func=lambda text: text,
        ... )
        >>> registry.dispatch("echo", {"text": "hi"})
        'hi'
    """

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    # ------------------------------------------------------------------ #
    # Registration
    # ------------------------------------------------------------------ #
    def register(self, tool: Tool, *, replace: bool = False) -> Tool:
        """Register a :class:`Tool`.

        Args:
            tool: The tool to add.
            replace: If False (default), registering a name that already exists
                raises ``ValueError``. Pass True to overwrite.

        Returns:
            The registered tool (so calls can be chained/inspected).
        """
        if not isinstance(tool, Tool):
            raise TypeError("register() expects a Tool instance")
        if tool.name in self._tools and not replace:
            raise ValueError(f"Tool {tool.name!r} is already registered")
        self._tools[tool.name] = tool
        return tool

    def register_function(
        self,
        name: str,
        description: str,
        func: Callable[..., Any],
        parameters: Optional[Dict[str, Any]] = None,
        *,
        replace: bool = False,
    ) -> Tool:
        """Convenience wrapper: build a :class:`Tool` and register it."""
        tool = Tool(
            name=name,
            description=description,
            parameters=parameters or {"type": "object", "properties": {}},
            func=func,
        )
        return self.register(tool, replace=replace)

    def unregister(self, name: str) -> None:
        """Remove a tool. Raises :class:`ToolNotFoundError` if absent."""
        if name not in self._tools:
            raise ToolNotFoundError(f"Unknown tool: {name!r}")
        del self._tools[name]

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def names(self) -> List[str]:
        """All registered tool names, in registration order."""
        return list(self._tools)

    def get(self, name: str) -> Tool:
        """Return a registered tool or raise :class:`ToolNotFoundError`."""
        try:
            return self._tools[name]
        except KeyError:
            raise ToolNotFoundError(f"Unknown tool: {name!r}") from None

    # ------------------------------------------------------------------ #
    # The two methods the agent loop uses
    # ------------------------------------------------------------------ #
    def get_schemas(self, enabled: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Return tool schemas to send to the model.

        Args:
            enabled: Optional allowlist of tool names (the "user can specify"
                control). When None, every registered tool is returned. Names
                in the allowlist that are not registered are skipped (so a stale
                config entry never crashes the agent). Order follows the
                allowlist when given, else registration order.

        Returns:
            A list of function-tool schema dicts.
        """
        if enabled is None:
            return [tool.to_schema() for tool in self._tools.values()]
        return [self._tools[name].to_schema() for name in enabled if name in self._tools]

    def dispatch(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> Any:
        """Execute a model-issued tool call.

        Args:
            name: The tool name the model asked to call.
            arguments: The arguments object from the model (already parsed from
                JSON into a dict). None is treated as no arguments.

        Returns:
            Whatever the tool's callable returns.

        Raises:
            ToolNotFoundError: The name is not registered.
            ToolArgumentError: ``arguments`` is not a dict, is missing a
                required key, or contains an unexpected key (when the schema
                forbids extras).

        Note:
            Exceptions raised *inside* the tool's own callable are allowed to
            propagate. The agent loop decides whether to feed them back to the
            model or abort -- the registry's job is dispatch, not policy.
        """
        tool = self.get(name)
        args = {} if arguments is None else arguments
        if not isinstance(args, dict):
            raise ToolArgumentError(
                f"Arguments for {name!r} must be an object/dict, got {type(args).__name__}"
            )
        self._validate_arguments(tool, args)
        return tool.func(**args)

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #
    @staticmethod
    def _validate_arguments(tool: Tool, args: Dict[str, Any]) -> None:
        """Lightweight argument validation against the tool's JSON Schema.

        Checks required keys are present and, unless the schema sets
        ``additionalProperties: true``, rejects unknown keys. Type checking is
        intentionally left to a later layer.
        """
        schema = tool.parameters or {}
        properties = schema.get("properties", {})
        required = schema.get("required", [])

        missing = [key for key in required if key not in args]
        if missing:
            raise ToolArgumentError(
                f"Tool {tool.name!r} missing required argument(s): {', '.join(missing)}"
            )

        allow_extra = schema.get("additionalProperties", False)
        if not allow_extra and properties:
            unexpected = [key for key in args if key not in properties]
            if unexpected:
                raise ToolArgumentError(
                    f"Tool {tool.name!r} got unexpected argument(s): {', '.join(unexpected)}"
                )
