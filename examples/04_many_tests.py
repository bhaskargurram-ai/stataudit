"""Example 4 - fifteen uncorrected tests.

Scenario: an A/B test with 15 secondary metrics. The assistant looped over
the metrics, ran a Welch t-test on each, and listed every metric with
p < 0.05 as "significantly improved". Only a couple of the effects are real.

Run:  stataudit run examples/04_many_tests.py
"""

import numpy as np
from scipy import stats

rng = np.random.default_rng(2024)

metrics = [
    "session_length", "pages_per_visit", "bounce_rate", "add_to_cart", "checkout_start",
    "checkout_done", "search_use", "filter_use", "wishlist_adds", "review_reads",
    "support_chats", "return_visits", "newsletter_signups", "share_clicks", "scroll_depth",
]
true_effect = {"add_to_cart": 0.9, "checkout_done": 0.8}

n = 40
significant = []
for name in metrics:
    control = rng.normal(0.0, 1.0, n)
    variant = rng.normal(true_effect.get(name, 0.0), 1.0, n)
    res = stats.ttest_ind(variant, control, equal_var=False)
    flag = "*" if res.pvalue < 0.05 else ""
    print(f"{name:<20} p = {res.pvalue:.4f} {flag}")
    if res.pvalue < 0.05:
        significant.append(name)

print(f"\n{len(significant)} metrics significantly improved: {', '.join(significant)}")
