"""A scripted LLM transport for agent tests (no network)."""

from __future__ import annotations

import itertools
import json
from typing import Any, Callable

import helpers  # noqa: F401  (puts src/ on sys.path)

from stataudit.agent.llm import ToolLLM
from stataudit.httpclient import HTTPResponse

_ids = itertools.count(1)
FAKE_KEY = "sk-ant-test-0123456789abcdefghij"


def anthropic_reply(calls: list[tuple[str, dict]], text: str = "") -> dict[str, Any]:
    content: list[dict[str, Any]] = [{"type": "text", "text": text}] if text else []
    for name, args in calls:
        content.append({"type": "tool_use", "id": f"toolu_{next(_ids)}", "name": name, "input": args})
    return {"id": "msg", "type": "message", "role": "assistant", "content": content,
            "stop_reason": "tool_use" if calls else "end_turn",
            "usage": {"input_tokens": 100, "output_tokens": 20, "cache_read_input_tokens": 5}}


def openai_reply(calls: list[tuple[str, dict]], text: str = "") -> dict[str, Any]:
    msg: dict[str, Any] = {"role": "assistant", "content": text or None}
    if calls:
        msg["tool_calls"] = [{"id": f"call_{next(_ids)}", "type": "function",
                              "function": {"name": n, "arguments": json.dumps(a)}} for n, a in calls]
    return {"choices": [{"index": 0, "message": msg, "finish_reason": "tool_calls" if calls else "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20}}


class ScriptedTransport:
    """Answers each request with the next scripted step.

    A step is a list of ``(tool_name, arguments)`` pairs, a string (a text-only
    turn), an ``HTTPResponse`` (returned as is), or a callable taking the
    request payload and returning one of those.
    """

    def __init__(self, provider: str, steps: list[Any]):
        self.provider = provider
        self.steps = list(steps)
        self.requests: list[dict[str, Any]] = []

    def __call__(self, method: str, url: str, headers: Any, body: bytes | None, timeout: float) -> HTTPResponse:
        payload = json.loads(body or b"{}")
        self.requests.append({"method": method, "url": url, "headers": dict(headers), "payload": payload})
        if not self.steps:
            raise AssertionError("the scripted LLM ran out of steps")
        step = self.steps.pop(0)
        if callable(step) and not isinstance(step, HTTPResponse):
            step = step(payload)
        if isinstance(step, HTTPResponse):
            return step
        calls, text = ([], step) if isinstance(step, str) else (step, "")
        reply = anthropic_reply(calls, text) if self.provider == "anthropic" else openai_reply(calls, text)
        return HTTPResponse(200, json.dumps(reply).encode())

    def all_text(self) -> str:
        return "\n".join(json.dumps(r["payload"]) for r in self.requests)


def factory(transport: ScriptedTransport, provider: str = "anthropic") -> Callable[[], ToolLLM]:
    return lambda: ToolLLM(provider, "test-model", FAKE_KEY, transport=transport, sleep=lambda s: None)
