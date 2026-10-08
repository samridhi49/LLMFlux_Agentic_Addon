"""Tests for the built-in agent tools (convert_file and web_browse).

These exercise both tools directly and through the registry, and -- following
the repo convention (see tests/slurm/test_connection.py) -- pin the specific
malicious inputs the safety guards must reject, using mocks so no real network
or external process is needed.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from llmflux.agents import build_default_registry
from llmflux.agents.tools import convert_file, web_browse


class TestConvertFileTool(unittest.TestCase):
    """convert_file: a pure-reuse wrapper around llmflux.converters."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dir = Path(self.temp_dir.name)
        self.csv_path = self.dir / "data.csv"
        self.csv_path.write_text("text\nhello world\nsecond row\n")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_happy_path_converts_and_summarizes(self):
        out = self.dir / "out.jsonl"
        result = convert_file(
            input_path=str(self.csv_path),
            prompt_template="Summarize: {text}",
            output_path=str(out),
        )
        self.assertEqual(result["output_path"], str(out))
        self.assertEqual(result["total_rows"], 2)
        self.assertTrue(out.exists())
        # One JSONL line per CSV row.
        self.assertEqual(len(out.read_text().strip().splitlines()), 2)

    def test_empty_prompt_template_rejected(self):
        with self.assertRaises(ValueError):
            convert_file(input_path=str(self.csv_path), prompt_template="  ")

    def test_missing_input_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            convert_file(
                input_path=str(self.dir / "nope.csv"),
                prompt_template="Summarize: {text}",
            )

    def test_through_registry_dispatch(self):
        out = self.dir / "via_registry.jsonl"
        registry = build_default_registry()
        result = registry.dispatch(
            "convert_file",
            {
                "input_path": str(self.csv_path),
                "prompt_template": "Summarize: {text}",
                "output_path": str(out),
            },
        )
        self.assertTrue(out.exists())
        self.assertEqual(result["total_rows"], 2)


class TestWebBrowseTool(unittest.TestCase):
    """web_browse: wraps requests, adds a scheme + SSRF safety layer."""

    def _getaddrinfo_returning(self, ip):
        """Build a fake getaddrinfo result mapping any host to `ip`."""
        return [(None, None, None, "", (ip, 0))]

    @patch("llmflux.agents.tools.requests.get")
    @patch("llmflux.agents.tools.socket.getaddrinfo")
    def test_happy_path_extracts_text(self, mock_resolve, mock_get):
        mock_resolve.return_value = self._getaddrinfo_returning("93.184.216.34")
        mock_response = MagicMock()
        mock_response.text = "<html><body><h1>Title</h1><script>x=1</script><p>Hello there</p></body></html>"
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        text = web_browse("https://example.com")
        self.assertIn("Title", text)
        self.assertIn("Hello there", text)
        # Script contents must be stripped.
        self.assertNotIn("x=1", text)

    def test_rejects_non_http_scheme(self):
        with self.assertRaises(ValueError):
            web_browse("file:///etc/passwd")
        with self.assertRaises(ValueError):
            web_browse("ftp://example.com/data")

    def test_rejects_url_without_host(self):
        with self.assertRaises(ValueError):
            web_browse("http://")

    @patch("llmflux.agents.tools.socket.getaddrinfo")
    def test_rejects_loopback_target_ssrf(self, mock_resolve):
        # A hostname that resolves to loopback must be rejected (SSRF guard).
        mock_resolve.return_value = self._getaddrinfo_returning("127.0.0.1")
        with self.assertRaises(ValueError):
            web_browse("http://evil.example.com")

    @patch("llmflux.agents.tools.socket.getaddrinfo")
    def test_rejects_private_target_ssrf(self, mock_resolve):
        # Internal/private address (e.g. cloud metadata range neighbor) rejected.
        mock_resolve.return_value = self._getaddrinfo_returning("10.0.0.5")
        with self.assertRaises(ValueError):
            web_browse("http://internal.example.com")

    @patch("llmflux.agents.tools.socket.getaddrinfo")
    def test_rejects_link_local_metadata_ssrf(self, mock_resolve):
        # 169.254.169.254 is the classic cloud metadata endpoint.
        mock_resolve.return_value = self._getaddrinfo_returning("169.254.169.254")
        with self.assertRaises(ValueError):
            web_browse("http://metadata.example.com")

    @patch("llmflux.agents.tools.socket.getaddrinfo")
    def test_rejects_unresolvable_host(self, mock_resolve):
        import socket as _socket
        mock_resolve.side_effect = _socket.gaierror("name resolution failed")
        with self.assertRaises(ValueError):
            web_browse("http://does-not-resolve.example.com")

    @patch("llmflux.agents.tools.requests.get")
    @patch("llmflux.agents.tools.socket.getaddrinfo")
    def test_truncates_long_content(self, mock_resolve, mock_get):
        mock_resolve.return_value = self._getaddrinfo_returning("93.184.216.34")
        mock_response = MagicMock()
        mock_response.text = "<p>" + ("a" * 50_000) + "</p>"
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        text = web_browse("https://example.com")
        self.assertLessEqual(len(text), 10_000)

    @patch("llmflux.agents.tools.requests.get")
    @patch("llmflux.agents.tools.socket.getaddrinfo")
    def test_through_registry_dispatch(self, mock_resolve, mock_get):
        mock_resolve.return_value = self._getaddrinfo_returning("93.184.216.34")
        mock_response = MagicMock()
        mock_response.text = "<p>Registry works</p>"
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        registry = build_default_registry()
        result = registry.dispatch("web_browse", {"url": "https://example.com"})
        self.assertIn("Registry works", result)


class TestDefaultRegistry(unittest.TestCase):
    """build_default_registry wires both tools in, ready for the agent loop."""

    def test_registers_both_tools(self):
        registry = build_default_registry()
        self.assertIn("convert_file", registry)
        self.assertIn("web_browse", registry)

    def test_schemas_match_responses_api_shape(self):
        registry = build_default_registry()
        schemas = registry.get_schemas()
        names = {s["name"] for s in schemas}
        self.assertEqual(names, {"convert_file", "web_browse"})
        for schema in schemas:
            self.assertEqual(schema["type"], "function")
            self.assertIn("parameters", schema)
            self.assertEqual(schema["parameters"]["type"], "object")

    def test_allowlist_can_disable_web_browse(self):
        registry = build_default_registry()
        schemas = registry.get_schemas(enabled=["convert_file"])
        self.assertEqual([s["name"] for s in schemas], ["convert_file"])


if __name__ == "__main__":
    unittest.main()
