"""Leakage rules SA201-SA204: each flags its pattern and stays quiet on the correct version."""

from __future__ import annotations

import textwrap
import unittest

import helpers  # noqa: F401  (sets up sys.path)

from stataudit.lint import lint_source


def leak_rules(src: str) -> list[tuple[str, int]]:
    rep = lint_source(textwrap.dedent(src), "a.py")
    return [(f.rule, f.line) for f in rep.findings if f.rule.startswith("SA2")]


class PreprocessingTests(unittest.TestCase):
    def test_fit_transform_then_split(self):
        self.assertEqual(leak_rules("""
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y)
        """), [("SA201", 3)])

    def test_fit_on_full_then_transform_after_split(self):
        found = leak_rules("""
            scaler = StandardScaler().fit(X)
            X_tr, X_te, y_tr, y_te = train_test_split(X, y)
            model.fit(scaler.transform(X_tr), y_tr)
        """)
        self.assertEqual(found, [("SA201", 2)])

    def test_vectorizer_on_all_documents(self):
        self.assertEqual(leak_rules("""
            vec = TfidfVectorizer()
            X = vec.fit_transform(docs)
            X_train, X_test, y_train, y_test = train_test_split(X, labels)
        """), [("SA201", 3)])

    def test_split_first_is_fine(self):
        self.assertEqual(leak_rules("""
            X_train, X_test, y_train, y_test = train_test_split(X, y)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        """), [])

    def test_pipeline_in_cross_validation_is_fine(self):
        self.assertEqual(leak_rules("""
            pipe = make_pipeline(StandardScaler(), LogisticRegression())
            pipe.fit(X, y)
            scores = cross_val_score(pipe, X, y, cv=5)
        """), [])

    def test_fit_without_later_use_is_fine(self):
        # fitted on everything, but nothing it learned reaches the evaluation
        self.assertEqual(leak_rules("""
            StandardScaler().fit(X)
            X_train, X_test, y_train, y_test = train_test_split(X, y)
        """), [])


class SelectionAndResamplingTests(unittest.TestCase):
    def test_selectkbest_before_cross_validation(self):
        self.assertEqual(leak_rules("""
            sel = SelectKBest(f_classif, k=20)
            X_sel = sel.fit_transform(X, y)
            scores = cross_val_score(LogisticRegression(), X_sel, y, cv=5)
        """), [("SA202", 3)])

    def test_selection_on_training_split_is_fine(self):
        self.assertEqual(leak_rules("""
            X_train, X_test, y_train, y_test = train_test_split(X, y)
            sel = SelectKBest(f_classif, k=20)
            X_train_sel = sel.fit_transform(X_train, y_train)
            scores = cross_val_score(LogisticRegression(), X_train_sel, y_train, cv=5)
        """), [])

    def test_smote_before_split(self):
        self.assertEqual(leak_rules("""
            X_res, y_res = SMOTE().fit_resample(X, y)
            X_train, X_test, y_train, y_test = train_test_split(X_res, y_res)
        """), [("SA204", 2)])

    def test_smote_on_training_split_is_fine(self):
        self.assertEqual(leak_rules("""
            X_train, X_test, y_train, y_test = train_test_split(X, y)
            X_res, y_res = SMOTE().fit_resample(X_train, y_train)
        """), [])


class TestSetTests(unittest.TestCase):
    def test_transformer_fitted_on_test_split(self):
        self.assertEqual(leak_rules("""
            X_train, X_test, y_train, y_test = train_test_split(X, y)
            X_train = StandardScaler().fit_transform(X_train)
            X_test = StandardScaler().fit_transform(X_test)
        """), [("SA203", 4)])

    def test_model_selection_on_test_split(self):
        self.assertEqual(leak_rules("""
            best, best_c = 0, None
            for c in [0.1, 1, 10]:
                m = SVC(C=c).fit(X_train, y_train)
                s = m.score(X_test, y_test)
                if s > best:
                    best, best_c = s, c
        """), [("SA203", 5)])

    def test_single_final_evaluation_is_fine(self):
        self.assertEqual(leak_rules("""
            for c in [0.1, 1, 10]:
                scores = cross_val_score(SVC(C=c), X_train, y_train)
            final = SVC(C=1).fit(X_train, y_train).score(X_test, y_test)
        """), [])

    def test_suppression_comment(self):
        self.assertEqual(leak_rules("""
            X_scaled = StandardScaler().fit_transform(X)  # stataudit: ignore[SA201]
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y)
        """), [])

    def test_string_split_is_not_a_data_split(self):
        self.assertEqual(leak_rules("""
            scaler = StandardScaler().fit(X)
            parts = line.split(",")
            fields = text.split(sep)
            scaler.transform(X)
        """), [])



class ExampleTests(unittest.TestCase):
    def test_leakage_example(self):
        from helpers import EXAMPLES  # noqa: PLC0415

        from stataudit.lint import lint_file  # noqa: PLC0415

        rules = sorted({f.rule for f in lint_file(EXAMPLES / "07_ml_leakage.py").findings})
        self.assertEqual(rules, ["SA201", "SA202", "SA203"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
