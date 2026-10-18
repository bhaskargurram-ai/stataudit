"""Static checks for train/test leakage in machine-learning evaluation code.

Used by :mod:`stataudit.lint` (rules SA201-SA204). A model evaluated on data
that influenced its training looks better than it is: the classic cases are
a scaler, imputer or feature selector fitted on the full data set before the
train/test split or cross-validation, oversampling before the split, and
using the test split to choose between models or hyperparameters (see
Kapoor & Narayanan, "Leakage and the reproducibility crisis in ML-based
science", Patterns 2023, for a taxonomy and its prevalence).

The analysis follows names through the module in source order and does not
model control flow; a finding is a prompt to check, like every lint rule.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

__all__ = ["LeakFinding", "find_leakage", "PREPROCESSORS", "SELECTORS", "RESAMPLERS", "SPLITTERS"]

#: Transformers that learn statistics from the data they are fitted on.
PREPROCESSORS = {
    "StandardScaler", "MinMaxScaler", "MaxAbsScaler", "RobustScaler", "Normalizer", "QuantileTransformer",
    "PowerTransformer", "SimpleImputer", "KNNImputer", "IterativeImputer", "PCA", "KernelPCA", "TruncatedSVD",
    "FastICA", "NMF", "FactorAnalysis", "TfidfVectorizer", "CountVectorizer", "HashingVectorizer",
    "TfidfTransformer", "OneHotEncoder", "OrdinalEncoder", "TargetEncoder", "KBinsDiscretizer",
    "SplineTransformer", "ColumnTransformer", "Pipeline", "make_pipeline", "make_column_transformer",
    "TSNE", "UMAP", "Isomap", "LocallyLinearEmbedding",
}
#: Feature selectors (most use the target).
SELECTORS = {
    "SelectKBest", "SelectPercentile", "SelectFpr", "SelectFdr", "SelectFwe", "GenericUnivariateSelect",
    "RFE", "RFECV", "SelectFromModel", "SequentialFeatureSelector", "VarianceThreshold", "mutual_info_classif",
    "mutual_info_regression", "f_classif", "f_regression", "chi2",
}
#: Resamplers whose synthetic or duplicated rows must not reach the test split.
RESAMPLERS = {
    "SMOTE", "SMOTENC", "SMOTEN", "ADASYN", "BorderlineSMOTE", "SVMSMOTE", "KMeansSMOTE", "RandomOverSampler",
    "RandomUnderSampler", "SMOTEENN", "SMOTETomek", "NearMiss", "TomekLinks", "ClusterCentroids",
}
#: Calls that split data or evaluate by cross-validation, with the position of the X argument.
SPLITTERS = {"train_test_split": 0, "cross_val_score": 1, "cross_validate": 1, "cross_val_predict": 1,
             "learning_curve": 1, "validation_curve": 1, "permutation_test_score": 1, "split": 0}

_FIT = {"fit", "fit_transform", "fit_resample", "partial_fit"}


@dataclass
class LeakFinding:
    rule: str
    node: ast.AST
    message: str


def _name(node: ast.AST | None) -> str | None:
    """A plain variable name, or ``df`` for ``df[...]`` / ``df.values`` style expressions."""
    while isinstance(node, (ast.Subscript, ast.Attribute)) and not (
            isinstance(node, ast.Attribute) and isinstance(node.value, ast.Call)):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def _ctor(node: ast.AST) -> str | None:
    """``StandardScaler`` for ``StandardScaler(...)`` / ``preprocessing.StandardScaler(...)``."""
    if isinstance(node, ast.Call):
        f = node.func
        if isinstance(f, ast.Name):
            return f.id
        if isinstance(f, ast.Attribute):
            return f.attr
    return None


def _kind(cls: str | None) -> str | None:
    if cls in PREPROCESSORS:
        return "preprocessor"
    if cls in SELECTORS:
        return "selector"
    if cls in RESAMPLERS:
        return "resampler"
    return None


def _is_test_name(name: str | None) -> bool:
    if not name:
        return False
    n = name.lower()
    return n in ("x_te", "xte", "x_test", "test", "test_x", "xtest", "df_test", "test_df", "x_val_test") or \
        n.startswith(("x_test", "test_")) or n.endswith(("_test", "_te"))


def _sorted_nodes(tree: ast.AST) -> list[ast.AST]:
    nodes = [n for n in ast.walk(tree) if hasattr(n, "lineno")]
    return sorted(nodes, key=lambda n: (n.lineno, n.col_offset))


def find_leakage(tree: ast.Module) -> list[LeakFinding]:
    """Rules SA201 (preprocessing before split), SA202 (feature selection before split),
    SA203 (fitting on, or selecting with, the test split) and SA204 (resampling before split).

    A fit on data that is later split leaks only if its result reaches the
    evaluation: the transformed data is what gets split, or the fitted object
    is used (``transform``, ``predict``, ``score``) after the split. Fitting a
    pipeline and then passing it to ``cross_val_score`` does not leak, because
    cross-validation refits a clone on each training fold.
    """
    objects: dict[str, str] = {}  # variable -> class name of the estimator it holds
    derived: dict[str, tuple[str, str, str, ast.AST]] = {}  # output var -> (fit data, desc, kind, fit node)
    fitted: dict[str, tuple[str, str, str, ast.AST]] = {}   # object var -> (fit data, desc, kind, fit node)
    split_at: dict[str, ast.Call] = {}  # data var -> first split call
    findings: list[LeakFinding] = []
    reported: set[tuple[str, int]] = set()

    def add(rule: str, node: ast.AST, msg: str) -> None:
        key = (rule, getattr(node, "lineno", 0))
        if key not in reported:
            reported.add(key)
            findings.append(LeakFinding(rule, node, msg))

    for node in _sorted_nodes(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            cls = _ctor(node.value)
            if _kind(cls):
                objects[node.targets[0].id] = cls  # type: ignore[assignment]
        if not isinstance(node, ast.Call):
            continue
        name = node.func.attr if isinstance(node.func, ast.Attribute) else (
            node.func.id if isinstance(node.func, ast.Name) else "")
        if name in SPLITTERS and (name != "split" or isinstance(node.func, ast.Attribute)):
            pos = SPLITTERS[name]
            data = _name(node.args[pos]) if len(node.args) > pos else None
            if data is None:
                continue
            split_at.setdefault(data, node)
            seen: set[str] = set()
            cur = data
            while cur in derived and cur not in seen:  # scaled -> selected -> split: report every step
                seen.add(cur)
                _report(add, derived[cur], cur, node, name)
                cur = derived[cur][0]
            continue
        if not isinstance(node.func, ast.Attribute) or not node.args:
            continue
        recv = node.func.value
        recv_name = recv.id if isinstance(recv, ast.Name) else None
        # using an object that was fitted on data split afterwards
        if name in ("transform", "predict", "predict_proba", "score", "decision_function") and recv_name in fitted:
            src, desc, kind, fit_node = fitted[recv_name]
            split = split_at.get(src)
            if split is not None and fit_node.lineno < split.lineno < node.lineno:
                _report(add, fitted[recv_name], src, split, _ctor(split) or "split")
            target = _assign_target(tree, node)
            if target and name == "transform":
                derived[target] = fitted[recv_name]
            continue
        if name not in _FIT:
            continue
        cls = objects.get(recv_name) if recv_name else _ctor(recv)
        kind = _kind(cls)
        data = _name(node.args[0])
        if data is None:
            continue
        desc = f"`{recv_name}` ({cls})" if recv_name and cls else f"`{cls}`" if cls else "a model"
        if _is_test_name(data) and name in ("fit", "fit_transform", "partial_fit"):
            what = "a scaler/encoder/imputer" if kind == "preprocessor" else (
                "a feature selector" if kind == "selector" else "a model")
            add("SA203", node, f"{desc} is fitted on `{data}`, which looks like the test split: {what} "
                               "fitted on test data makes the test score optimistic. Fit on the training split and "
                               "only call `.transform`/`.predict` on the test split.")
            continue
        if kind is None or _is_train_name(data):
            continue
        entry = (data, desc, kind, node)
        if recv_name:
            fitted[recv_name] = entry
        target = _assign_target(tree, node)
        if target and name in ("fit_transform", "fit_resample"):
            derived[target] = entry
        elif target and name == "fit":  # scaler = StandardScaler().fit(X): fit returns the estimator
            fitted[target] = entry
            objects[target] = cls  # type: ignore[assignment]

    # SA203 (model selection on the test split): a loop that both fits and scores on test-named data
    for loop in (n for n in ast.walk(tree) if isinstance(n, (ast.For, ast.While))):
        calls = [c for c in ast.walk(loop) if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)]
        fits = [c for c in calls if c.func.attr == "fit"]
        scored = [c for c in calls if c.func.attr in ("score", "predict", "predict_proba", "decision_function")
                  and c.args and _is_test_name(_name(c.args[0]))]
        compares = any(isinstance(n, ast.Compare) or (isinstance(n, ast.Call) and _ctor(n) in ("max", "argmax", "min"))
                       for n in ast.walk(loop))
        if fits and scored and compares:
            add("SA203", scored[0], f"Inside a loop, models are fitted and then scored on "
                                    f"`{_name(scored[0].args[0])}` and the scores are compared: choosing a model or "
                                    "hyperparameter with the test split makes its test score optimistic. Choose with "
                                    "a validation split or cross-validation on the training data, and use the test "
                                    "split once at the end.")
    return findings


def _is_train_name(name: str) -> bool:
    n = name.lower()
    return n in ("x_tr", "xtr", "train", "xtrain") or n.startswith(("x_train", "train_")) or n.endswith(("_train", "_tr"))


def _report(add, entry: tuple[str, str, str, ast.AST], data: str, split: ast.Call, method: str) -> None:  # noqa: ANN001
    src, desc, kind, fit_node = entry
    line = getattr(fit_node, "lineno", 0)
    split_desc = "a cross-validation `.split`" if method == "split" else f"`{method}`"
    what = f"{desc} is fitted on `{src}` (line {line})" if data == src else \
        f"`{data}` comes from {desc} fitted on `{src}` (line {line})"
    tail = {"preprocessor": "the fitted statistics include the test rows. Fit it on the training split only, or "
                            "put it in a Pipeline that is cross-validated.",
            "selector": "features were chosen using the test rows (and usually their labels). Select features "
                        "inside the cross-validation, e.g. in a Pipeline.",
            "resampler": "resampled or synthetic rows derived from test rows end up in training, and test rows "
                         "may be duplicated. Resample the training split only (imblearn's Pipeline does this "
                         "inside cross-validation)."}[kind]
    rule = {"preprocessor": "SA201", "selector": "SA202", "resampler": "SA204"}[kind]
    add(rule, fit_node, f"{what}, before the data is split by {split_desc} (line {split.lineno}): {tail}")


def _assign_target(tree: ast.Module, call: ast.Call) -> str | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and node.value is call:
            t = node.targets[0]
            if isinstance(t, ast.Name):
                return t.id
            if isinstance(t, ast.Tuple) and t.elts and isinstance(t.elts[0], ast.Name):
                return t.elts[0].id  # X_res, y_res = sampler.fit_resample(X, y)
    return None
