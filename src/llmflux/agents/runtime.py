"""Sequential backend agent loop using Responses function tools."""

import json
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, field

import requests

from .config import AgentConfig
from .responses_client import ResponsesClient
from .tools import ToolRegistry


@dataclass
class AgentResult:
    """Task outcome. Completed means model completion, not verified correctness."""

    status: str = "failed"
    answer: str = ""
    error: str | None = None
    artifacts: list[str] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    usage: list[dict] = field(default_factory=list)
    turns: int = 0
    tool_calls: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


class AgentRuntime:
    """Execute one task at a time; each run starts fresh conversation state.

    An injected client is owned by the caller. Otherwise use this runtime as a
    context manager to close its HTTP session. Tools run synchronously; the
    cooperative deadline cannot interrupt Python tools or cancel remote work.
    """

    def __init__(self, config: AgentConfig, client: ResponsesClient | None = None,
                 api_key: str | None = None):
        self.config = config
        self.tools = ToolRegistry(config)
        self._owns_client = client is None
        self.client = client if client is not None else ResponsesClient(config, api_key=api_key)

    def run(self, task: str) -> AgentResult:
        if not isinstance(task, str) or not task.strip():
            raise ValueError("task must be a nonempty string")
        result = AgentResult()
        started = time.monotonic()
        deadline = started + self.config.task_timeout
        history = [{"role": "user", "content": task}]
        seen_calls = set()

        def finish(status, error=None):
            result.status, result.error = status, error
            result.events.append({"type": "finished", "status": status,
                                  "elapsed_seconds": time.monotonic() - started})
            return result

        for _ in range(self.config.max_turns):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return finish("deadline_exceeded", "Task deadline reached")
            result.turns += 1
            try:
                response = self.client.create(deepcopy(history), self.tools.schemas(), timeout=remaining)
            except (requests.RequestException, ValueError) as exc:
                # Do not include HTTP request details or credentials in task traces.
                return finish("failed", f"Model request failed ({type(exc).__name__})")
            status = response.get("status")
            result.events.append({"type": "model_response", "turn": result.turns,
                                  "response_id": response.get("id"), "status": status})
            if isinstance(response.get("usage"), dict):
                result.usage.append(deepcopy(response["usage"]))
            if time.monotonic() >= deadline:
                return finish("deadline_exceeded", "Task deadline reached")
            if status != "completed":
                return finish("incomplete" if status == "incomplete" else "failed",
                              f"Model response status: {status}")
            output = response.get("output")
            if not isinstance(output, list) or any(not isinstance(item, dict) for item in output):
                return finish("failed", "Invalid model output")
            calls = [item for item in output if item.get("type") == "function_call"]
            # Preserve reasoning, messages, and tool-call items for the next turn.
            history.extend(deepcopy(output))
            if not calls:
                text = []
                for item in output:
                    if item.get("type") != "message" or item.get("role") != "assistant":
                        continue
                    content = item.get("content", [])
                    if not isinstance(content, list):
                        return finish("failed", "Invalid assistant message")
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "output_text" and isinstance(part.get("text"), str):
                            text.append(part["text"])
                result.answer = "\n".join(text)
                if not result.answer.strip():
                    return finish("failed", "No final text or function call returned")
                return finish("completed")
            if result.turns == self.config.max_turns:
                return finish("limit_exceeded", "No model turns remain to consume tool results")
            if result.tool_calls + len(calls) > self.config.max_tool_calls:
                return finish("limit_exceeded", "Tool call limit reached")
            # Validate all IDs before performing any side effects in this turn.
            ids = [call.get("call_id") for call in calls]
            if (any(not isinstance(value, str) or not value for value in ids)
                    or len(set(ids)) != len(ids) or seen_calls.intersection(ids)):
                return finish("failed", "Missing or repeated tool call ID")
            seen_calls.update(ids)
            for call in calls:
                if time.monotonic() >= deadline:
                    return finish("deadline_exceeded", "Task deadline reached")
                name = call.get("name")
                if not isinstance(name, str):
                    return finish("failed", "Missing tool name")
                result.tool_calls += 1
                tool_result = self.tools.execute(name, call.get("arguments"))
                result.events.append({"type": "tool_result", "call_id": call["call_id"],
                                      "name": name, "arguments": call.get("arguments"),
                                      "result": deepcopy(tool_result)})
                if tool_result.get("status") == "success" and tool_result.get("artifact"):
                    result.artifacts.append(tool_result["artifact"])
                history.append({"type": "function_call_output", "call_id": call["call_id"],
                                "output": json.dumps(tool_result, ensure_ascii=False)})
        return finish("limit_exceeded", "Model turn limit reached")

    def close(self):
        if self._owns_client:
            self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
