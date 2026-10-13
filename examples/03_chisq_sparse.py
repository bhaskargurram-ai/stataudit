"""Example 3 - chi-square on a sparse contingency table.

Scenario: churned customers by plan (free / pro / enterprise) and the reason
they gave (price / missing features / support / security). The assistant ran
chi2_contingency on the 3x4 table and reported that churn reasons "depend on
the plan". Only four enterprise customers churned, and exactly one customer
in the whole table cited security - most expected counts are far below 5,
so the chi-square approximation is not trustworthy.

Run:  stataudit run examples/03_chisq_sparse.py
"""

import numpy as np
from scipy import stats

#                  price  features  support  security
table = np.array([
    [14,    6,        3,       0],   # free
    [14,    9,        4,       0],   # pro
    [1,     1,        1,       1],   # enterprise
])

chi2, p, dof, expected = stats.chi2_contingency(table)
print(f"chi2 = {chi2:.2f}, dof = {dof}, p = {p:.3f}")
print("Conclusion: churn reasons depend on the plan." if p < 0.05 else "No evidence of association.")
