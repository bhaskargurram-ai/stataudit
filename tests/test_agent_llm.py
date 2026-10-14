"""The tool-use LLM client: both wire formats, retries, configuration from STATAUDIT_LLM_*."""

from __future__ import annotations

import json
import unittest

from agent_fakes import ScriptedTransport

from stataudit.agent.llm import (
    LLMConfigError,
    LLMToolError,
    ToolLLM,
    ToolSpec,
    Turn,
    client_from_env,
    parse_anthropic,
    parse_openai,
)
from stataudit.httpclient import HTTPResponse

TOOL = ToolSpec("submit", "submit it", {"type": "object", "properties": {"x": {"type": "string"}}})


class TestRequests(unittest.TestCase):
    def test_anthropic_request_and_reply(self):
        tr = ScriptedTransport("anthropic", [[("submit", {"x": "1"})]])
        llm = ToolLLM("anthropic", "m1", "key123456", transport=tr)
        reply = llm.chat("sys", [Turn("user", text="hi")], [TOOL], force_tool="submit")
        req = tr.requests[0]
        self.assertTrue(req["url"].endswith("/v1/messages"))
        self.assertEqual(req["headers"]["x-api-key"], "key123456")
        p = req["payload"]
        self.assertEqual(p["model"], "m1")
        self.assertEqual(p["tool_choice"], {"type": "tool", "name": "submit"})
        self.assertEqual(p["tools"][0]["input_schema"]["type"], "object")
        self.assertNotIn("temperature", p)
        self.assertEqual(reply.tool_calls[0].name, "submit")
        self.assertEqual(reply.tool_calls[0].arguments, {"x": "1"})
        self.assertEqual(reply.usage.input_tokens, 100)

    def test_openai_request_reasoning_effort_and_tool_results(self):
        tr = ScriptedTransport("openai", [[("submit", {"x": "2"})]])
        llm = ToolLLM("openai", "gpt-x", "key123456", transport=tr, reasoning_effort="none")
        from stataudit.agent.llm import ToolCall, ToolResult

        turns = [Turn("user", text="hi"), Turn("assistant", tool_calls=[ToolCall("c1", "submit", {"x": "0"})]),
                 Turn("tool", results=[ToolResult("c1", "submit", "rejected", is_error=True)])]
        reply = llm.chat("sys", turns, [TOOL])
        p = tr.requests[0]["payload"]
        self.assertTrue(tr.requests[0]["url"].endswith("/chat/completions"))
        self.assertEqual(tr.requests[0]["headers"]["Authorization"], "Bearer key123456")
        self.assertEqual(p["reasoning_effort"], "none")
        self.assertIn("max_completion_tokens", p)
        self.assertEqual(p["messages"][0], {"role": "system", "content": "sys"})
        self.assertEqual(p["messages"][-1]["role"], "tool")
        self.assertTrue(p["messages"][-1]["content"].startswith("ERROR: "))
        self.assertEqual(reply.tool_calls[0].arguments, {"x": "2"})

    def test_compatible_server_uses_max_tokens(self):
        tr = ScriptedTransport("openai", ["ok"])
        ToolLLM("openai", "m", "", base_url="http://localhost:8000/v1", transport=tr).chat("s", [Turn("user", text="x")], [])
        self.assertIn("max_tokens", tr.requests[0]["payload"])
        self.assertTrue(tr.requests[0]["url"].startswith("http://localhost:8000/v1/"))

    def test_retries_then_succeeds(self):
        sleeps: list[float] = []
        tr = ScriptedTransport("anthropic", [HTTPResponse(429, b"{}", {"retry-after": "2"}), HTTPResponse(503, b""), "done"])
        llm = ToolLLM("anthropic", "m", "k" * 10, transport=tr, sleep=sleeps.append)
        reply = llm.chat("s", [Turn("user", text="x")], [])
        self.assertEqual(reply.text, "done")
        self.assertEqual(sleeps[0], 2.0)
        self.assertEqual(len(tr.requests), 3)

    def test_client_error_is_not_retried_and_hint_has_no_key(self):
        body = json.dumps({"error": {"message": "invalid x-api-key"}}).encode()
        tr = ScriptedTransport("anthropic", [HTTPResponse(401, body)])
        with self.assertRaises(LLMToolError) as ctx:
            ToolLLM("anthropic", "m", "secret-key-value", transport=tr).chat("s", [Turn("user", text="x")], [])
        self.assertIn("401", str(ctx.exception))
        self.assertNotIn("secret-key-value", str(ctx.exception))

    def test_parse_bad_shapes(self):
        with self.assertRaises(LLMToolError):
            parse_anthropic({"nope": 1})
        with self.assertRaises(LLMToolError):
            parse_openai({"choices": []})
        r = parse_openai({"choices": [{"message": {"tool_calls": [{"id": "a", "function": {"name": "f", "arguments": "{bad"}}]}}]})
        self.assertTrue(r.tool_calls[0].parse_error)


class TestConfig(unittest.TestCase):
    def test_env(self):
        env = {"STATAUDIT_LLM_PROVIDER": "openai", "STATAUDIT_LLM_MODEL": "gpt-6-luna",
               "STATAUDIT_LLM_API_KEY": "k" * 20, "STATAUDIT_LLM_REASONING_EFFORT": "none"}
        llm = client_from_env(env)
        self.assertEqual((llm.provider, llm.model, llm.reasoning_effort), ("openai", "gpt-6-luna", "none"))
        # reasoning effort is OpenAI-compatible only
        env2 = dict(env, STATAUDIT_LLM_PROVIDER="anthropic")
        self.assertEqual(client_from_env(env2).reasoning_effort, "")

    def test_errors(self):
        base = {"STATAUDIT_LLM_PROVIDER": "anthropic", "STATAUDIT_LLM_MODEL": "m", "STATAUDIT_LLM_API_KEY": "k" * 10}
        for drop, msg in (("STATAUDIT_LLM_MODEL", "no default model"), ("STATAUDIT_LLM_PROVIDER", "PROVIDER"),
                          ("STATAUDIT_LLM_API_KEY", "API_KEY")):
            env = {k: v for k, v in base.items() if k != drop}
            with self.subTest(drop=drop), self.assertRaises(LLMConfigError) as ctx:
                client_from_env(env)
            self.assertIn(msg, str(ctx.exception))
        with self.assertRaises(LLMConfigError):
            client_from_env(dict(base, STATAUDIT_LLM_PROVIDER="mystery"))
        with self.assertRaises(LLMConfigError):
            client_from_env(dict(base, STATAUDIT_LLM_REASONING_EFFORT="ultra"))
        # generic LLM_* names are deliberately not read
        with self.assertRaises(LLMConfigError):
            client_from_env({"LLM_PROVIDER": "anthropic", "LLM_MODEL": "m", "LLM_API_KEY": "k" * 10})
        # a local server needs no key
        local = client_from_env({"STATAUDIT_LLM_PROVIDER": "openai", "STATAUDIT_LLM_MODEL": "m",
                                 "STATAUDIT_LLM_BASE_URL": "http://localhost:1234/v1"})
        self.assertEqual(local.api_key, "")

    def test_config_fallbacks(self):
        llm = client_from_env({"STATAUDIT_LLM_API_KEY": "k" * 10}, provider="anthropic", model="from-config")
        self.assertEqual(llm.model, "from-config")


if __name__ == "__main__":
    unittest.main()
