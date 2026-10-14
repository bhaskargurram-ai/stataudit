"""Optional LLM agent (experimental, bring your own key).

* :mod:`stataudit.agent.fix` - ``stataudit fix``: the model proposes a corrected
  analysis; stataudit re-runs it under the tracer and accepts it only if the
  audit says it is better.
* :mod:`stataudit.agent.explain` - ``stataudit explain``: a plain-language
  summary whose every number must appear in the findings.

The model never decides a verdict. Its tools are read-only, its only output is
a schema-validated submit tool, and anything it writes is executed only in a
subprocess with a timeout and a scrubbed environment, after a static
denylist check. See docs/AGENT.md and docs/SECURITY_MODEL.md.
"""
