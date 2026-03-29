"""
Figure 1: The Garden of Forking Paths in Random Effects Specification

Shows how many distinct lme4-style model formulas are possible for a
typical 2x2 repeated-measures design (factors A and B, crossed with
participants and items).  The combinatorial explosion makes exhaustive
pre-registration of the random-effects structure impractical.
"""

import itertools
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ------------------------------------------------------------------
# 1.  Enumerate random-effects structures
# ------------------------------------------------------------------
# Fixed effects of interest: intercept, A, B, A:B  (4 terms)
# Random effects can be specified for TWO grouping factors:
#   - by_subject
#   - by_item
# For each grouping factor, the analyst can include any non-empty
# subset of {intercept, A, B, A:B}, plus choose whether to allow or
# suppress correlations among random slopes (when >1 term included).

fixed_terms = ["(Intercept)", "A", "B", "A:B"]

def count_re_structures_for_grouping(terms):
    """
    For a single grouping factor, count distinct RE specifications.
    - Any non-empty subset of `terms` can be included.
    - If the subset has >1 element, correlations can be ON or OFF.
    Returns list of (subset, corr) labels.
    """
    structures = []
    for r in range(1, len(terms) + 1):
        for subset in itertools.combinations(terms, r):
            if len(subset) == 1:
                structures.append((subset, "—"))  # no correlation choice
            else:
                structures.append((subset, "corr"))
                structures.append((subset, "no corr"))
    return structures

subj_structures = count_re_structures_for_grouping(fixed_terms)
item_structures = count_re_structures_for_grouping(fixed_terms)

total = len(subj_structures) * len(item_structures)

# ------------------------------------------------------------------
# 2.  Categorise by "complexity" (total number of RE parameters)
# ------------------------------------------------------------------
def n_params(subset, corr):
    k = len(subset)
    if k == 1 or corr == "no corr":
        return k           # one variance per term, no covariance
    else:
        return k + k*(k-1)//2  # variances + upper-triangle covariances

counts_by_complexity = {}
for ss, sc in subj_structures:
    for is_, ic in item_structures:
        n = n_params(ss, sc) + n_params(is_, ic)
        counts_by_complexity[n] = counts_by_complexity.get(n, 0) + 1

complexities = sorted(counts_by_complexity)
counts = [counts_by_complexity[c] for c in complexities]

# ------------------------------------------------------------------
# 3.  Plot
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

# Left panel: cumulative number of models
cum = np.cumsum(counts)
axes[0].barh(complexities, counts, color="#4C72B0", edgecolor="white", height=0.7)
axes[0].set_xlabel("Number of distinct models", fontsize=11)
axes[0].set_ylabel("Total random-effects parameters", fontsize=11)
axes[0].set_title("Model count by RE complexity", fontsize=12)
for i, (c, v) in enumerate(zip(complexities, counts)):
    if v > 2:
        axes[0].text(v + 0.5, c, str(v), va='center', fontsize=8)
axes[0].set_xlim(0, max(counts) * 1.18)

# Right panel: pie of broad categories
labels_pie = ["Minimal\n(1–3 params)", "Intermediate\n(4–7 params)",
              "Maximal\n(≥8 params)"]
groups = [
    sum(v for k, v in counts_by_complexity.items() if k <= 3),
    sum(v for k, v in counts_by_complexity.items() if 4 <= k <= 7),
    sum(v for k, v in counts_by_complexity.items() if k >= 8),
]
colors = ["#55A868", "#C44E52", "#4C72B0"]
wedges, texts, autotexts = axes[1].pie(
    groups, labels=labels_pie, colors=colors,
    autopct=lambda p: f"{p:.0f}%\n({int(round(p/100*total))})",
    startangle=90, textprops={"fontsize": 9})
axes[1].set_title(f"Total: {total} distinct models\nfor a 2×2 design", fontsize=12)

fig.suptitle(
    "How many random-effects models can one fit to a 2×2 design?",
    fontsize=13, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig("fig1_forking_paths.pdf", bbox_inches="tight")
fig.savefig("fig1_forking_paths.png", dpi=150, bbox_inches="tight")
print(f"Total distinct RE structures: {total}")
plt.show()
