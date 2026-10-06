"""Non-streaming vLLM Responses API transport; tools execute in runtime.py."""

import math
import os

import requests

from .config import AgentConfig


class ResponsesClient:
    """No automatic retries: a task may have already produced side effects."""

    def __init__(self, config: AgentConfig, api_key: str | None = None):
        self.config = config
        self.session = requests.Session()
        candidates = (api_key, os.getenv("LLMFLUX_API_KEY"))
        key = next((value.strip() for value in candidates if value and value.strip()), None)
        if key:
            self.session.headers["Authorization"] = f"Bearer {key}"

    def create(self, input: list[dict], tools: list[dict], *, timeout: float | None = None) -> dict:
        """Return the full response, preserving reasoning and function-call items."""
        budget = self.config.request_timeout if timeout is None else timeout
        if not math.isfinite(budget) or budget <= 0:
            raise ValueError("timeout must be finite and positive")
        payload = {"model": self.config.model, "input": input, "tools": tools,
                   "max_output_tokens": self.config.max_output_tokens,
                   "stream": False, "store": False}
        if tools:
            payload["tool_choice"] = "auto"
        with self.session.post(self.config.base_url + "/responses", json=payload,
                               timeout=min(budget, self.config.request_timeout)) as response:
            response.raise_for_status()
            data = response.json()
        if not isinstance(data, dict) or not isinstance(data.get("status"), str):
            raise ValueError("Responses API returned an invalid response object")
        if not isinstance(data.get("output"), list) or any(not isinstance(item, dict) for item in data["output"]):
            raise ValueError("Responses API output must be a list of objects")
        return data

    def close(self):
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
