"""Text, Markdown, HTML and JSON rendering."""

from __future__ import annotations

import json
import unittest
from html.parser import HTMLParser

import numpy as np
from helpers import make_record

import stataudit
from stataudit.audits import audit_records


def _report():
    r = np.random.default_rng(7)
    control = r.normal(12.0, 1.5, 60).round(1)
    treated = np.concatenate([r.normal(12.2, 1.5, 11), [21.5, 24.0, 26.5]]).round(1)
    recs = [
        make_record("ttest_ind", treated, control, code="stats.ttest_ind(treated, control)"),
        make_record("chi2_contingency", np.array([[14, 6, 3, 0], [14, 9, 4, 0], [1, 1, 1, 1]])),
        make_record("pearsonr", np.arange(30.0), np.arange(30.0) + r.normal(size=30)),
    ]
    return audit_records(recs, source="analysis.py")


class _Collector(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags: list[str] = []
        self.external: list[str] = []

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        for k, v in attrs:
            if k in ("src", "href") and v and v.startswith(("http:", "https:", "//")):
                self.external.append(v)


class TestRenderers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rep = _report()

    def test_text(self):
        txt = self.rep.to_text()
        self.assertIn("stataudit report", txt)
        self.assertIn("does NOT hold", txt)
        self.assertIn("Welch's t-test", txt)
        self.assertIn("Holm-Bonferroni", txt)

    def test_text_ascii_and_color(self):
        txt = self.rep.to_text(ascii_only=True)
        txt.encode("ascii")  # must not raise
        self.assertIn("[XX]", txt)
        self.assertIn("\033[", self.rep.to_text(color=True))

    def test_markdown(self):
        md = self.rep.to_markdown()
        self.assertTrue(md.startswith("# stataudit report"))
        self.assertIn("| Analysis | Why | p-value |", md)
        self.assertIn("Multiple comparisons (Holm-Bonferroni)", md)
        self.assertIn("```python", md)
        self.assertIn("❌", md)

    def test_html_is_self_contained_and_escaped(self):
        html = self.rep.to_html()
        self.assertTrue(html.lstrip().lower().startswith("<!doctype html>"))
        self.assertIn("prefers-color-scheme: dark", html)
        p = _Collector()
        p.feed(html)
        self.assertEqual(p.external, [])
        self.assertNotIn("script", p.tags)
        self.assertNotIn("link", p.tags)
        # escaping
        rec = make_record("ttest_ind", [1.0, 2, 3, 4], [2.0, 3, 4, 9], code="ttest_ind(a<b>, c)")
        h = audit_records([rec]).to_html()
        self.assertIn("a&lt;b&gt;", h)
        self.assertNotIn("<b>, c)", h)

    def test_html_embedded_fragment(self):
        from stataudit.reports.html import render_html

        frag = render_html(self.rep, embedded=True)
        self.assertNotIn("<html", frag)
        self.assertIn("banner", frag)

    def test_json(self):
        data = json.loads(self.rep.to_json())
        self.assertEqual(data["tool"], "stataudit")
        self.assertEqual(data["status"], "does_not_hold")
        self.assertEqual(len(data["results"]), 3)
        first = data["results"][0]
        self.assertEqual(first["func"], "ttest_ind")
        self.assertIn("verdict", first)
        self.assertTrue(any(a["name"] == "Welch's t-test" for a in first["alternatives"]))
        self.assertEqual(data["counts"]["does_not_hold"], 2)

    def test_json_handles_nan(self):
        rep = stataudit.audit_call("ttest_ind", [[1.0, 2, np.nan, 4], [2.0, 3, 4, 5]])
        json.loads(rep.to_json())  # NaN must have been converted to null


if __name__ == "__main__":
    unittest.main()
