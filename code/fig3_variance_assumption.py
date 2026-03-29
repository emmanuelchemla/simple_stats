"""
Figure 3: The hidden variance homogeneity assumption in maximal models

Barr et al.'s (2013) recommendation to "keep it maximal" implicitly
assumes that random-slope variance is the same across conditions
(i.e., that the covariance structure is constant).  This figure shows
two scenarios:

  A) Homogeneous variance: the assumption holds; pooling is efficient.
  B) Heterogeneous variance: the assumption is violated; the maximal
     model distorts inference.

We illustrate the bias in estimated fixed-effect SE when the wrong
variance structure is imposed.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

rng = np.random.default_rng(0)

N_SIMS  = 2000
N_SUBJ  = 30
N_ITEMS = 16

TRUE_EFFECT = 0.0   # null effect – we track Type I error

# ------------------------------------------------------------------
# Scenario A: homogeneous random slopes (sigma_slope same in both conds)
# ------------------------------------------------------------------
def gen_homogeneous(n_subj, n_items, sigma_int=0.6, sigma_slope=0.4):
    b_int   = rng.normal(0, sigma_int,   n_subj)
    b_slope = rng.normal(0, sigma_slope, n_subj)
    data = []
    for i in range(n_subj):
        for j in range(n_items):
            for c in [0, 1]:
                y = b_int[i] + c * b_slope[i] + rng.normal(0, 0.5)
                data.append((i, j, c, y))
    return np.array(data)


# ------------------------------------------------------------------
# Scenario B: heterogeneous – condition 1 has much larger subject variance
# ------------------------------------------------------------------
def gen_heterogeneous(n_subj, n_items, sigma_int=0.6,
                      sigma_slope_cond0=0.1, sigma_slope_cond1=0.8):
    b_int    = rng.normal(0, sigma_int,         n_subj)
    b_s0     = rng.normal(0, sigma_slope_cond0, n_subj)
    b_s1     = rng.normal(0, sigma_slope_cond1, n_subj)
    data = []
    for i in range(n_subj):
        for j in range(n_items):
            y0 = b_int[i] + b_s0[i] + rng.normal(0, 0.5)
            y1 = b_int[i] + b_s1[i] + rng.normal(0, 0.5)
            data.append((i, j, 0, y0))
            data.append((i, j, 1, y1))
    return np.array(data)


def ttest_on_diffs(data, n_subj):
    """Simple paired t-test on per-subject means."""
    means = []
    for i in range(n_subj):
        subj = data[data[:, 0] == i]
        m0 = subj[subj[:, 2] == 0, 3].mean()
        m1 = subj[subj[:, 2] == 1, 3].mean()
        means.append(m1 - m0)
    means = np.array(means)
    t, p = stats.ttest_1samp(means, 0)
    return p


def pooled_ttest(data, n_subj):
    """
    Mimics the maximal LMM's pooling by estimating a single slope
    variance from the grand average – correct under homogeneity,
    biased under heterogeneity.
    We simulate this by pooling variance from both conditions together
    before computing the SE.
    """
    all_means = []
    for i in range(n_subj):
        subj = data[data[:, 0] == i]
        m0 = subj[subj[:, 2] == 0, 3].mean()
        m1 = subj[subj[:, 2] == 1, 3].mean()
        all_means.append((m0, m1))
    all_means = np.array(all_means)
    grand_var = np.var(all_means, axis=0).mean()   # pooled
    diffs = all_means[:, 1] - all_means[:, 0]
    se = np.sqrt(2 * grand_var / n_subj)
    t  = diffs.mean() / se
    p  = 2 * stats.t.sf(abs(t), df=n_subj - 1)
    return p


# ------------------------------------------------------------------
# Run
# ------------------------------------------------------------------
results = {
    "homogeneous": {"simple": [], "pooled": []},
    "heterogeneous": {"simple": [], "pooled": []},
}

for _ in range(N_SIMS):
    d_h = gen_homogeneous(N_SUBJ, N_ITEMS)
    results["homogeneous"]["simple"].append(ttest_on_diffs(d_h, N_SUBJ))
    results["homogeneous"]["pooled"].append(pooled_ttest(d_h, N_SUBJ))

    d_v = gen_heterogeneous(N_SUBJ, N_ITEMS)
    results["heterogeneous"]["simple"].append(ttest_on_diffs(d_v, N_SUBJ))
    results["heterogeneous"]["pooled"].append(pooled_ttest(d_v, N_SUBJ))


def type1(ps, alpha=0.05):
    return np.mean(np.array(ps) < alpha)


# ------------------------------------------------------------------
# Plot
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=False)

scenarios = ["homogeneous", "heterogeneous"]
titles    = ["A  –  Homogeneous variance\n(model assumption holds)",
             "B  –  Heterogeneous variance\n(model assumption violated)"]

for ax, scenario, title in zip(axes, scenarios, titles):
    simple_p = np.array(results[scenario]["simple"])
    pooled_p = np.array(results[scenario]["pooled"])

    # P-value histograms
    bins = np.linspace(0, 1, 21)
    ax.hist(simple_p, bins=bins, alpha=0.55, color="#4C72B0",
            label=f"Paired t-test  (α-err={type1(simple_p):.3f})", density=True)
    ax.hist(pooled_p, bins=bins, alpha=0.55, color="#C44E52",
            label=f"Pooled/maximal (α-err={type1(pooled_p):.3f})", density=True)
    ax.axhline(1, color="gray", linestyle="--", linewidth=1, label="Uniform (ideal)")
    ax.set_xlabel("p-value", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8.5)

fig.suptitle(
    "Type I error under the null: effect of variance homogeneity assumption",
    fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig("fig3_variance_assumption.pdf", bbox_inches="tight")
fig.savefig("fig3_variance_assumption.png", dpi=150, bbox_inches="tight")
plt.show()
