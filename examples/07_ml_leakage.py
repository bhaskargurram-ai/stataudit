"""A classifier evaluation with three kinds of train/test leakage (for `stataudit lint`).

Each step looks routine, and each one lets the test rows influence training:
the scaler's statistics, the selected features and the chosen C all used
the test split. Run `stataudit lint examples/07_ml_leakage.py` to see SA201,
SA202 and SA203. This file is for linting; it is not meant to be run.
"""

from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = make_classification(n_samples=200, n_features=500, n_informative=5, random_state=0)

# SA201: scaled with statistics of all 200 rows, test rows included
X_scaled = StandardScaler().fit_transform(X)

# SA202: the 20 features most associated with y, chosen using every label
selector = SelectKBest(f_classif, k=20)
X_sel = selector.fit_transform(X_scaled, y)

X_train, X_test, y_train, y_test = train_test_split(X_sel, y, test_size=0.3, random_state=0)

# SA203: C chosen by its score on the test split
best_score, best_c = 0.0, None
for c in (0.01, 0.1, 1.0, 10.0):
    model = SVC(C=c).fit(X_train, y_train)
    score = model.score(X_test, y_test)
    if score > best_score:
        best_score, best_c = score, c

print(f"test accuracy {best_score:.2f} with C={best_c}")
