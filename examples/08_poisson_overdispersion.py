"""Example 8 - a Poisson regression on overdispersed counts.

Scenario: support tickets per store in the 30 days before and after a new
in-store layout (60 stores, 30 with the new layout). The assistant fitted a
Poisson regression, found the new layout "significantly" reduces tickets
(rate ratio 0.75, p = 0.01) and wrote it up. The counts vary about five
times more than a Poisson model allows, so its standard errors are far too
small.

Needs statsmodels. Run:  stataudit run examples/08_poisson_overdispersion.py
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

rng = np.random.default_rng(4)

stores = pd.DataFrame({"new_layout": np.repeat([0, 1], 30)})
mean_tickets = np.exp(1.5)  # the layout has no effect
stores["tickets"] = rng.negative_binomial(1.2, 1.2 / (1.2 + mean_tickets), size=60)

fit = smf.poisson("tickets ~ new_layout", data=stores).fit(disp=0)
ratio, p = np.exp(fit.params["new_layout"]), fit.pvalues["new_layout"]
print(f"Rate ratio = {ratio:.2f}, p = {p:.2g}")
print("Conclusion: the new layout reduces tickets." if p < 0.05 else "No effect detected.")
