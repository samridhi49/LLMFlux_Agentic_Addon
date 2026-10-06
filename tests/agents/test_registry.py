"""Tests for the agent tool registry."""

import unittest

from llmflux.agents import (
    Tool,
    ToolRegistry,
    ToolNotFoundError,
    ToolArgumentError,
)


def _echo_schema():
    return {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }


class TestToolConstruction(unittest.TestCase):
    """Tool is a valid (schema + callable) pair, or it refuses to exist."""

    def test_valid_tool_builds(self):
        tool = Tool(name="echo", description="Echo text.",
                    parameters=_echo_schema(), func=lambda text: text)
        self.assertEqual(tool.name, "echo")

    def test_empty_name_rejected(self):
        with self.assertRaises(ValueError):
            Tool(name="", description="x", func=lambda: None)

    def test_non_callable_func_rejected(self):
        with self.assertRaises(ValueError):
            Tool(name="bad", description="x", func="not_callable")

    def test_non_dict_parameters_rejected(self):
        with self.assertRaises(ValueError):
            Tool(name="bad", description="x", parameters=[], func=lambda: None)

    def test_to_schema_uses_flattened_function_shape(self):
        tool = Tool(name="echo", description="Echo text.",
                    parameters=_echo_schema(), func=lambda text: text)
        schema = tool.to_schema()
        self.assertEqual(schema["type"], "function")
        self.assertEqual(schema["name"], "echo")
        self.assertEqual(schema["description"], "Echo text.")
        self.assertIn("properties", schema["parameters"])

    def test_to_schema_defaults_empty_object_when_no_parameters(self):
        tool = Tool(name="noop", description="Does nothing.", func=lambda: "ok")
        schema = tool.to_schema()
        self.assertEqual(schema["parameters"]["type"], "object")
        self.assertEqual(schema["parameters"]["properties"], {})


class TestRegistration(unittest.TestCase):
    """Registering, overwriting, and removing tools."""

    def setUp(self):
        self.registry = ToolRegistry()

    def test_register_function_and_contains(self):
        self.registry.register_function(
            "echo", "Echo text.", lambda text: text, _echo_schema())
        self.assertIn("echo", self.registry)
        self.assertEqual(len(self.registry), 1)
        self.assertEqual(self.registry.names(), ["echo"])

    def test_duplicate_name_rejected(self):
        self.registry.register_function("echo", "d", lambda text: text, _echo_schema())
        with self.assertRaises(ValueError):
            self.registry.register_function("echo", "d2", lambda text: text, _echo_schema())

    def test_duplicate_name_allowed_with_replace(self):
        self.registry.register_function("echo", "v1", lambda text: "v1", _echo_schema())
        self.registry.register_function(
            "echo", "v2", lambda text: "v2", _echo_schema(), replace=True)
        self.assertEqual(self.registry.dispatch("echo", {"text": "x"}), "v2")

    def test_register_rejects_non_tool(self):
        with self.assertRaises(TypeError):
            self.registry.register({"name": "echo"})

    def test_unregister_removes_tool(self):
        self.registry.register_function("echo", "d", lambda text: text, _echo_schema())
        self.registry.unregister("echo")
        self.assertNotIn("echo", self.registry)

    def test_unregister_unknown_raises(self):
        with self.assertRaises(ToolNotFoundError):
            self.registry.unregister("ghost")


class TestGetSchemas(unittest.TestCase):
    """get_schemas() feeds the model, with an optional user allowlist."""

    def setUp(self):
        self.registry = ToolRegistry()
        self.registry.register_function("read_file", "Read a file.", lambda path: path,
                                        {"type": "object", "properties": {"path": {"type": "string"}},
                                         "required": ["path"]})
        self.registry.register_function("web_browse", "Fetch a URL.", lambda url: url,
                                        {"type": "object", "properties": {"url": {"type": "string"}},
                                         "required": ["url"]})

    def test_returns_all_by_default(self):
        schemas = self.registry.get_schemas()
        names = [s["name"] for s in schemas]
        self.assertEqual(names, ["read_file", "web_browse"])

    def test_allowlist_filters_and_orders(self):
        schemas = self.registry.get_schemas(enabled=["web_browse"])
        self.assertEqual([s["name"] for s in schemas], ["web_browse"])

    def test_allowlist_skips_unknown_names(self):
        # A stale config entry must not crash the agent.
        schemas = self.registry.get_schemas(enabled=["read_file", "does_not_exist"])
        self.assertEqual([s["name"] for s in schemas], ["read_file"])

    def test_empty_allowlist_exposes_nothing(self):
        self.assertEqual(self.registry.get_schemas(enabled=[]), [])


class TestDispatch(unittest.TestCase):
    """dispatch() turns a model tool-call into a real result, safely."""

    def setUp(self):
        self.registry = ToolRegistry()
        self.calls = []

        def add(a, b):
            self.calls.append((a, b))
            return a + b

        self.registry.register_function(
            "add", "Add two integers.", add,
            {"type": "object",
             "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
             "required": ["a", "b"]})

    def test_happy_path_executes_and_returns(self):
        result = self.registry.dispatch("add", {"a": 2, "b": 3})
        self.assertEqual(result, 5)
        self.assertEqual(self.calls, [(2, 3)])

    def test_unknown_tool_raises(self):
        with self.assertRaises(ToolNotFoundError):
            self.registry.dispatch("subtract", {"a": 1, "b": 2})

    def test_missing_required_argument_raises(self):
        with self.assertRaises(ToolArgumentError):
            self.registry.dispatch("add", {"a": 1})

    def test_unexpected_argument_raises(self):
        with self.assertRaises(ToolArgumentError):
            self.registry.dispatch("add", {"a": 1, "b": 2, "c": 3})

    def test_non_dict_arguments_raises(self):
        with self.assertRaises(ToolArgumentError):
            self.registry.dispatch("add", ["a", "b"])

    def test_none_arguments_allowed_for_no_arg_tool(self):
        self.registry.register_function("ping", "Return pong.", lambda: "pong",
                                        {"type": "object", "properties": {}})
        self.assertEqual(self.registry.dispatch("ping", None), "pong")

    def test_additional_properties_true_allows_extras(self):
        self.registry.register_function(
            "flexible", "Accepts anything.", lambda **kw: kw,
            {"type": "object", "properties": {"x": {"type": "string"}},
             "additionalProperties": True})
        self.assertEqual(self.registry.dispatch("flexible", {"x": "a", "y": "b"}),
                         {"x": "a", "y": "b"})

    def test_tool_internal_exception_propagates(self):
        def boom():
            raise RuntimeError("tool failed internally")

        self.registry.register_function("boom", "Always fails.", boom,
                                        {"type": "object", "properties": {}})
        with self.assertRaises(RuntimeError):
            self.registry.dispatch("boom", {})


if __name__ == "__main__":
    unittest.main()
