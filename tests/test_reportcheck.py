"""check-report: extraction of APA-style results, recomputation against scipy, and statuses."""

from __future__ import annotations

import json
import shutil
import subprocess
import unittest
from pathlib import Path

from click.testing import CliRunner
from helpers import EXAMPLES, TempDirMixin
from scipy import stats

from stataudit.cli import main as cli
from stataudit.reportcheck import (
    CONSISTENT,
    GROSS,
    IMPOSSIBLE,
    INCONSISTENT,
    check_file,
    check_text,
    html_to_text,
    parse_num,
    recompute_p,
)


def one(text: str, alpha: float = 0.05):
    rc = check_text(text, alpha=alpha)
    assert len(rc.results) == 1, [r.label for r in rc.results]
    return rc.results[0]


class TestRecompute(unittest.TestCase):
    """The recomputed p-values are exactly scipy's."""

    def test_against_scipy(self):
        cases = [
            ("t", 2.20, 28, None, 2 * stats.t.sf(2.20, 28)),
            ("t", -3.1, 12.7, None, 2 * stats.t.sf(3.1, 12.7)),
            ("z", 1.96, None, None, 2 * stats.norm.sf(1.96)),
            ("F", 3.81, 2, 45, stats.f.sf(3.81, 2, 45)),
            ("chi2", 7.31, 2, None, stats.chi2.sf(7.31, 2)),
            ("Q", 25.1, 12, None, stats.chi2.sf(25.1, 12)),
        ]
        for kind, stat, d1, d2, expected in cases:
            with self.subTest(kind=kind):
                self.assertAlmostEqual(recompute_p(kind, stat, d1, d2), expected, places=14)

    def test_r_matches_pearsonr_on_real_data(self):
        import numpy as np

        rng = np.random.default_rng(1)
        x = rng.normal(size=50)
        y = 0.3 * x + rng.normal(size=50)
        r, p = stats.pearsonr(x, y)
        self.assertAlmostEqual(recompute_p("r", r, 48), p, places=10)

    def test_t_matches_ttest_ind_on_real_data(self):
        import numpy as np

        rng = np.random.default_rng(2)
        a, b = rng.normal(0, 1, 15), rng.normal(0.8, 1, 15)
        res = stats.ttest_ind(a, b)
        self.assertAlmostEqual(recompute_p("t", res.statistic, 28), res.pvalue, places=12)

    def test_one_tailed_halves(self):
        self.assertAlmostEqual(recompute_p("t", 1.8, 15, tails=1), stats.t.sf(1.8, 15), places=14)

    def test_parse_num_precision(self):
        n = parse_num(".036")
        self.assertEqual((n.value, round(n.half, 6)), (0.036, 0.0005))
        n = parse_num("2.3 × 10-5")
        self.assertAlmostEqual(n.value, 2.3e-5)
        self.assertAlmostEqual(n.half, 0.05e-5)
        self.assertEqual(parse_num("12").half, 0.5)


class TestPhrasings(unittest.TestCase):
    """Real-world ways of writing results are found and judged correctly."""

    CONSISTENT_CASES = [
        "t(28) = 2.20, p = .036",
        "t(28)=2.20, p=.036",
        "t (28) = 2.20, p = 0.036",
        "*t*(28) = 2.20, *p* = .036",
        "_t_(28) = 2.20, _p_ = .036",
        "**t**(28) = 2.20, **p** = .036",
        "t[28] = 2.20, p = .036",
        "t(28) = −2.20, p = .036",            # Unicode minus
        "t(28) = –2.20, p = .036",            # en dash as minus
        "t(27.4) = 2.10, p = .045",           # Welch df
        "t(40) = 3.9, p < .001",
        "t(40) = 0.4, p > .05",
        "t(40) = 0.4, ns",
        "t(40) = 0.4, n.s.",
        "t(28) = 2.20, p ≤ .05",
        "t(28) = 2.20, p-value = .036",
        "t(28) = 2.20, ps < .05",
        "F(2, 45) = 3.81, p = .030",
        "F(2,45) = 3.81, p = .03",
        "F(1; 38) = 4.50, MSE = 2.10, p = .040",
        "F(1, 30) < 1, ns",
        "F[2, 45] = 3.81, p = .03",
        "r(48) = .32, p = .023",
        "r(48) = −.32, p = .023",
        "r(48) = 0.32, p = .023",
        "χ2(2, N = 120) = 7.31, p = .026",
        "χ²(2) = 7.31, p = .026",
        "χ^2(2) = 7.31, p = .026",
        "$\\chi^2(2) = 7.31$, $p = .026$",
        "chi2(2) = 7.31, p = .026",
        "Chi-square(2) = 7.31, p = .026",
        "chi-squared(2) = 7.31, p = .026",
        "X2(2) = 7.31, p = .026",
        "z = 2.31, p = .021",
        "Z = 2.31, p = .021",
        "Q(12) = 25.1, p = .014",
        "t = 2.20, df = 28, p-value = 0.03622",   # R t.test print style
        "X-squared = 7.31, df = 2, p-value = 0.02586",
        "F = 3.2, num df = 2, denom df = 26.7, p-value = 0.05678",
        "z = 4.50, p = 6.8 × 10⁻⁶",
        "z = 4.50, p = 6.8e-6",
    ]

    def test_consistent_phrasings(self):
        for text in self.CONSISTENT_CASES:
            with self.subTest(text=text):
                r = one(text)
                self.assertEqual(r.status, CONSISTENT, f"{r.label}: {r.message}")

    def test_kinds_and_values(self):
        r = one("χ²(2, N = 120) = 7.31, p = .026")
        self.assertEqual((r.kind, r.df1.value, r.n, r.stat.value), ("chi2", 2, 120, 7.31))
        r = one("F(1; 38) = 4.50, MSE = 2.10, p = .040")
        self.assertEqual((r.kind, r.df1.value, r.df2.value), ("F", 1, 38))
        r = one("r(48) = −.32, p = .023")
        self.assertEqual((r.kind, r.stat.value), ("r", -0.32))

    def test_statistic_without_p_is_skipped(self):
        self.assertEqual(check_text("We found t(28) = 2.20 overall.").results, [])
        self.assertEqual(check_text("z-scores were computed; p values follow.").results, [])

    def test_words_are_not_statistics(self):
        rc = check_text("The format(2) = 3 call and set(1, 2) = 4, p = .03 are code, not results. Max(3) = 4, p = .5")
        self.assertEqual(rc.results, [])

    def test_several_results_in_one_sentence(self):
        rc = check_text("Both effects were reliable, F(1, 40) = 6.10, p = .018, and F(2, 80) = 4.20, p = .018.")
        self.assertEqual([r.kind for r in rc.results], ["F", "F"])
        self.assertEqual([r.status for r in rc.results], [CONSISTENT, CONSISTENT])

    def test_result_wrapped_over_lines_and_line_numbers(self):
        text = "Intro.\n\nTime on task was related to experience, *r*(70) = −.28,\n*p* = .017, as expected.\n"
        r = one(text)
        self.assertEqual(r.status, CONSISTENT)
        self.assertEqual(r.line, 3)

    def test_table_rows_stand_alone(self):
        text = "| test | result |\n|---|---|\n| A | t(20) = 2.5, p = .021 |\n| B | t(20) = 0.5, p = .62 |\n"
        rc = check_text(text)
        self.assertEqual([(r.line, r.status) for r in rc.results], [(3, CONSISTENT), (4, CONSISTENT)])


class TestStatuses(unittest.TestCase):
    def test_rounding_to_reported_decimals(self):
        # t(28) = 2.2 could be 2.15..2.25 -> p in [.0324, .0398]; p = .04 rounds into that range
        self.assertEqual(one("t(28) = 2.2, p = .04").status, CONSISTENT)
        # true p = .0362 rounds to .04, so p = .04 is consistent; written to 3 decimals it is not
        self.assertEqual(one("t(28) = 2.20, p = .04").status, CONSISTENT)
        r = one("t(28) = 2.20, p = .040")
        self.assertEqual(r.status, INCONSISTENT)
        self.assertIn("0.0362", r.message)
        # p reported to 2 decimals: .03 covers [.025, .035)
        self.assertEqual(one("t(28) = 2.20, p = .03").status, INCONSISTENT)
        self.assertEqual(one("t(28) = 2.25, p = .03").status, CONSISTENT)

    def test_inconsistent_but_not_gross(self):
        r = one("t(35) = 2.04, p = .03")
        self.assertEqual(r.status, INCONSISTENT)
        self.assertTrue(r.reported_significant)
        self.assertTrue(r.computed_significant)

    def test_gross_both_directions(self):
        r = one("t(28) = 1.20, p < .05")
        self.assertEqual(r.status, GROSS)
        self.assertIn("not significant", r.message)
        r = one("t(28) = 2.20, p = .36")
        self.assertEqual(r.status, GROSS)
        self.assertIn("significant", r.message)
        r = one("t(28) = 2.20, ns")
        self.assertEqual(r.status, GROSS)
        self.assertEqual(one("F(2, 69) = 2.95, p = .03").status, GROSS)
        self.assertEqual(one("z = 1.70, p < .05").status, GROSS)

    def test_alpha_changes_gross(self):
        # p = .03 reported, true p = .049: inconsistent at .05, but a flip at alpha = .04
        self.assertEqual(one("t(35) = 2.04, p = .03", alpha=0.05).status, INCONSISTENT)
        self.assertEqual(one("t(35) = 2.04, p = .03", alpha=0.04).status, GROSS)

    def test_p_equal_alpha_counts_as_significant(self):
        r = one("t(30) = 2.04, p = .05")  # true p = .0502
        self.assertEqual(r.status, CONSISTENT)
        self.assertTrue(r.reported_significant)

    def test_p_less_than_001(self):
        self.assertEqual(one("t(40) = 3.9, p < .001").status, CONSISTENT)
        r = one("t(40) = 2.9, p < .001")  # true p = .006
        self.assertEqual(r.status, INCONSISTENT)
        r = one("t(40) = 3.9, p = .000")
        self.assertEqual(r.status, CONSISTENT)
        self.assertTrue(any("p < .001" in n for n in r.notes))

    def test_one_tailed(self):
        # t(15) = 1.80: two-tailed p = .092, one-tailed p = .046
        r = one("A directional prediction: t(15) = 1.80, p = .046, one-tailed.")
        self.assertEqual((r.status, r.tails), (CONSISTENT, 1))
        # one-tailed mentioned elsewhere in the document: consistent with a note
        rc = check_text("All tests were one-tailed.\n\nThe effect: t(15) = 1.80, p = .046.")
        self.assertEqual(rc.results[0].status, CONSISTENT)
        self.assertTrue(rc.results[0].notes)
        # never mentioned: a gross inconsistency
        self.assertEqual(one("The effect: t(15) = 1.80, p = .046.").status, GROSS)

    def test_impossible_values(self):
        for text, why in [
            ("t(10) = 1.0, p = 1.20", "greater than 1"),
            ("r(48) = 1.32, p = .02", "between -1 and 1"),
            ("t(0) = 2.1, p = .04", "degrees of freedom"),
            ("F(2, 30) = -1.5, p = .3", "cannot be negative"),
            ("χ2(1) = -3.2, p = .07", "cannot be negative"),
            ("t(20) = 2.0, p = 0", "exactly 0"),
        ]:
            with self.subTest(text=text):
                r = one(text)
                self.assertEqual(r.status, IMPOSSIBLE, r.message)
                self.assertIn(why, r.message)

    def test_adjusted_p_values(self):
        r = one("After Bonferroni correction, t(28) = 2.50, p = .055.")
        self.assertEqual(r.status, CONSISTENT)
        self.assertTrue(r.adjusted)
        r = one("Holm-adjusted: t(28) = 2.50, p = .005.")  # raw p = .019 > .005: impossible for an adjusted p
        self.assertEqual(r.status, INCONSISTENT)
        # the adjustment word of an earlier sentence does not leak into the next result
        rc = check_text("We used Bonferroni. Next, z = 1.70, p < .05.")
        self.assertFalse(rc.results[0].adjusted)
        self.assertEqual(rc.results[0].status, GROSS)

    def test_statistic_bounds(self):
        self.assertEqual(one("F(1, 30) < 1, ns").status, CONSISTENT)
        self.assertEqual(one("F(1, 30) < 1, p < .05").status, GROSS)
        self.assertEqual(one("t(30) > 5, p < .001").status, CONSISTENT)


class TestDocuments(TempDirMixin, unittest.TestCase):
    def test_example_report(self):
        rc = check_file(EXAMPLES / "reports" / "results.md")
        self.assertEqual(len(rc.results), 8)
        self.assertEqual(rc.counts(), {CONSISTENT: 5, INCONSISTENT: 1, GROSS: 2, IMPOSSIBLE: 0})
        self.assertEqual(rc.status, GROSS)

    def test_html(self):
        page = ("<html><head><style>p{}</style><script>var t='t(5) = 9, p = .9'</script></head><body>"
                "<p>The effect, <i>t</i>(28) = 2.20, <i>p</i> &lt; .05, and <i>&chi;</i><sup>2</sup>(1) = 0.5, "
                "<i>p</i> = .48.</p><table><tr><td>F(2, 45) = 3.81</td><td>p = .40</td></tr></table></body></html>")
        text = html_to_text(page)
        self.assertNotIn("var t", text)
        path = Path(self.tmp) / "r.html"
        path.write_text(page, encoding="utf-8")
        rc = check_file(path)
        self.assertEqual([(r.kind, r.status) for r in rc.results],
                         [("t", CONSISTENT), ("chi2", CONSISTENT), ("F", GROSS)])
        self.assertIn("HTML", rc.note)

    @unittest.skipUnless(shutil.which("pdftotext"), "pdftotext not installed")
    def test_pdf_when_pdftotext_exists(self):
        pdf = Path(self.tmp) / "r.pdf"
        if not _make_pdf(pdf, ["The effect was t(28) = 2.20, p = .36 in study 1.", "Study 2: z = 2.31, p = .021."]):
            self.skipTest("could not build a PDF here")
        rc = check_file(pdf)
        self.assertEqual([r.status for r in rc.results], [GROSS, CONSISTENT])
        self.assertEqual(rc.results[1].page, 2)

    def test_renderers(self):
        rc = check_text("Bad: t(28) = 2.20, p = .36 <script>. Good: z = 2.31, p = .021.", source="x.md")
        d = json.loads(rc.to_json())
        self.assertEqual(d["kind"], "check-report")
        self.assertEqual(d["counts"][GROSS], 1)
        self.assertAlmostEqual(d["results"][0]["computed_p"], 2 * stats.t.sf(2.2, 28))
        md = rc.to_markdown()
        self.assertIn("| ❌ | line 1 |", md)
        html = rc.to_html()
        self.assertTrue(html.startswith("<!doctype html>"))
        self.assertNotIn("<script>", html)
        txt = rc.to_text()
        self.assertIn("gross inconsistency", txt)
        self.assertNotIn("z = 2.31", txt)          # consistent ones hidden unless verbose
        self.assertIn("z = 2.31", rc.to_text(verbose=True))


def _make_pdf(path: Path, pages: list[str]) -> bool:
    """A minimal hand-written PDF with one line of text per page."""
    objs = []
    kids = []
    n_pages = len(pages)
    font_id = 3 + 2 * n_pages
    for i, text in enumerate(pages):
        page_id, content_id = 3 + 2 * i, 4 + 2 * i
        kids.append(f"{page_id} 0 R")
        stream = f"BT /F1 11 Tf 50 700 Td ({text}) Tj ET".encode("latin-1")
        objs.append((page_id, f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {content_id} 0 R "
                              f"/Resources << /Font << /F1 {font_id} 0 R >> >> >>".encode()))
        objs.append((content_id, b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream"))
    objs.append((font_id, b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"))
    objs.insert(0, (2, f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {n_pages} >>".encode()))
    objs.insert(0, (1, b"<< /Type /Catalog /Pages 2 0 R >>"))
    objs.sort()
    out = bytearray(b"%PDF-1.4\n")
    offsets = {}
    for oid, body in objs:
        offsets[oid] = len(out)
        out += f"{oid} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for oid in range(1, len(objs) + 1):
        out += f"{offsets[oid]:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    path.write_bytes(bytes(out))
    try:
        proc = subprocess.run(["pdftotext", str(path), "-"], capture_output=True, timeout=30)
    except OSError:
        return False
    return proc.returncode == 0 and b"t(28)" in proc.stdout


class TestCheckReportCLI(TempDirMixin, unittest.TestCase):
    def test_exit_codes_and_outputs(self):
        runner = CliRunner()
        md = Path(self.tmp) / "r.md"
        md.write_text("Fine: t(28) = 2.20, p = .036. Off: t(35) = 2.04, p = .03.\n", encoding="utf-8")
        res = runner.invoke(cli, ["check-report", str(md)])
        self.assertEqual(res.exit_code, 1, res.output)
        self.assertIn("inconsistent", res.output)
        res = runner.invoke(cli, ["check-report", str(md), "--fail-on", "gross"])
        self.assertEqual(res.exit_code, 0, res.output)
        out_json, out_md, out_html = (str(Path(self.tmp) / n) for n in ("o.json", "o.md", "o.html"))
        res = runner.invoke(cli, ["check-report", str(md), "--fail-on", "never", "-q", "--json", out_json, "--md", out_md,
                                  "--html", out_html])
        self.assertEqual(res.exit_code, 0, res.output)
        self.assertEqual(json.loads(Path(out_json).read_text())["counts"][INCONSISTENT], 1)
        self.assertIn("# stataudit check-report", Path(out_md).read_text())
        self.assertIn("<table>", Path(out_html).read_text())

    def test_gross_fails_with_gross_threshold(self):
        md = Path(self.tmp) / "r.md"
        md.write_text("t(28) = 1.20, p < .05\n", encoding="utf-8")
        res = CliRunner().invoke(cli, ["check-report", str(md), "--fail-on", "gross"])
        self.assertEqual(res.exit_code, 1)

    def test_pdf_without_pdftotext_is_a_usage_error(self):
        pdf = Path(self.tmp) / "x.pdf"
        pdf.write_bytes(b"%PDF-1.4\n")
        from unittest import mock

        with mock.patch("stataudit.reportcheck.shutil.which", return_value=None):
            res = CliRunner().invoke(cli, ["check-report", str(pdf)])
        self.assertEqual(res.exit_code, 2)
        self.assertIn("pdftotext", res.output)


if __name__ == "__main__":
    unittest.main()
