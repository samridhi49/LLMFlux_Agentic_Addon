"""Built-in tools for the LLMFlux agent layer.

Two tools are defined here as proof-of-concept, each demonstrating a different
"lightweight wrapper" pattern:

* ``convert_file`` -- a *pure reuse* wrapper. It adds argument validation and a
  tidy, model-friendly return value around LLMFlux's existing, already-tested
  ``llmflux.converters`` code. No new dependency, no new logic.

* ``web_browse`` -- a *wrap-a-library* wrapper. It leans on the ``requests``
  library (already an LLMFlux dependency) for the hard part (HTTP), and adds our
  own safety layer: scheme allow-listing and an SSRF guard that rejects URLs
  resolving to loopback/link-local/private/reserved addresses. The guard mirrors
  the approach already used in ``llmflux.slurm.connection``. HTML-to-text uses
  the stdlib here to stay dependency-free; a production version could swap in a
  specialized extractor (e.g. trafilatura) behind the same wrapper.

``build_default_registry()`` wires both into a :class:`ToolRegistry`.
"""

from __future__ import annotations

import ipaddress
import socket
from html.parser import HTMLParser
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import requests

from ..converters import csv_to_jsonl
from .registry import ToolRegistry

# --------------------------------------------------------------------------- #
# convert_file -- pure reuse of llmflux.converters
# --------------------------------------------------------------------------- #

def convert_file(
    input_path: str,
    prompt_template: str,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Convert a CSV file to JSONL using an existing LLMFlux converter.

    Args:
        input_path: Path to the source CSV file.
        prompt_template: Template with ``{column}`` placeholders, e.g.
            ``"Summarize: {text}"``.
        output_path: Where to write the JSONL. Defaults to a temp file.

    Returns:
        A compact summary the model can read: output path and row counts.

    Raises:
        FileNotFoundError: If ``input_path`` does not exist.
        ValueError: If ``prompt_template`` is empty.
    """
    if not prompt_template or not prompt_template.strip():
        raise ValueError("prompt_template must be a non-empty string")

    result = csv_to_jsonl(
        input_path=input_path,
        output_path=output_path,
        prompt_template=prompt_template,
    )
    # Return a trimmed, model-friendly view rather than the full internal dict.
    return {
        "output_path": result.get("output_path"),
        "total_rows": result.get("total_rows"),
        "successful_conversions": result.get("successful_conversions"),
        "failed_conversions": result.get("failed_conversions"),
    }


_CONVERT_FILE_SCHEMA = {
    "type": "object",
    "properties": {
        "input_path": {"type": "string", "description": "Path to the source CSV file."},
        "prompt_template": {
            "type": "string",
            "description": "Template with {column} placeholders, e.g. 'Summarize: {text}'.",
        },
        "output_path": {
            "type": "string",
            "description": "Optional path for the JSONL output; a temp file is used if omitted.",
        },
    },
    "required": ["input_path", "prompt_template"],
    "additionalProperties": False,
}


# --------------------------------------------------------------------------- #
# web_browse -- wrap the requests library, add our own SSRF safety layer
# --------------------------------------------------------------------------- #

_ALLOWED_SCHEMES = {"http", "https"}
_MAX_CONTENT_CHARS = 10_000


class _TextExtractor(HTMLParser):
    """Minimal HTML-to-text: collect text, drop <script>/<style> contents."""

    def __init__(self) -> None:
        super().__init__()
        self._chunks: list[str] = []
        self._skip = False

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag in ("script", "style"):
            self._skip = True

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style"):
            self._skip = False

    def handle_data(self, data: str) -> None:
        if not self._skip and data.strip():
            self._chunks.append(data.strip())

    def get_text(self) -> str:
        return " ".join(self._chunks)


def _is_forbidden_address(addr: "ipaddress._BaseAddress") -> bool:
    """True for loopback/link-local/private/multicast/unspecified/reserved IPs.

    Mirrors the SSRF guard in llmflux.slurm.connection so a model-supplied URL
    cannot be used to reach internal services.
    """
    return bool(
        addr.is_loopback
        or addr.is_link_local
        or addr.is_private
        or addr.is_multicast
        or addr.is_unspecified
        or addr.is_reserved
    )


def _check_url_safe(url: str) -> None:
    """Reject non-http(s) schemes and URLs resolving to forbidden addresses."""
    parsed = urlparse(url)
    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise ValueError(f"url {url!r} must use http or https, got {parsed.scheme!r}")

    host = parsed.hostname
    if not host:
        raise ValueError(f"url {url!r} has no host")

    # Resolve the host the way the HTTP client will, and re-check every address
    # it maps to -- this catches names (and legacy IP encodings) that point at
    # loopback/metadata targets.
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise ValueError(f"could not resolve host {host!r}") from exc

    for info in infos:
        ip = info[4][0]
        if _is_forbidden_address(ipaddress.ip_address(ip)):
            raise ValueError(f"url {url!r} resolves to a forbidden address: {ip}")


def web_browse(url: str, timeout: int = 10) -> str:
    """Fetch a web page and return its visible text (truncated).

    Args:
        url: An http(s) URL.
        timeout: Per-request timeout in seconds.

    Returns:
        The page's text content, truncated to a model-friendly length.

    Raises:
        ValueError: If the URL is unsafe (bad scheme, no host, forbidden target).
        requests.HTTPError: If the server returns an error status.
    """
    _check_url_safe(url)
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()

    extractor = _TextExtractor()
    extractor.feed(response.text)
    text = extractor.get_text()
    return text[:_MAX_CONTENT_CHARS]


_WEB_BROWSE_SCHEMA = {
    "type": "object",
    "properties": {
        "url": {"type": "string", "description": "The http(s) URL to fetch."},
    },
    "required": ["url"],
    "additionalProperties": False,
}


# --------------------------------------------------------------------------- #
# Registry assembly
# --------------------------------------------------------------------------- #

def build_default_registry() -> ToolRegistry:
    """Return a registry pre-loaded with the built-in tools."""
    registry = ToolRegistry()
    registry.register_function(
        name="convert_file",
        description="Convert a CSV file to JSONL using a prompt template.",
        func=convert_file,
        parameters=_CONVERT_FILE_SCHEMA,
    )
    registry.register_function(
        name="web_browse",
        description="Fetch a web page over http(s) and return its visible text.",
        func=web_browse,
        parameters=_WEB_BROWSE_SCHEMA,
    )
    return registry
