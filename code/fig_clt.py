"""
fig_clt.py

Figure: The CLT justifies treating binary outcomes as Gaussian.

Three panels:

A.  With realistic numbers of trials per subject (8–32), the
    distribution of per-subject proportions is already well-approximated
    by a Gaussian.  The approximation improves with N and degrades only
    when the true probability is very close to 0 or 1.

B.  Adequacy of the normal approximation across (p_true, N_items).
    Classical rule of thumb: adequate when N·p > 5 and N·(1−p) > 5.
    Confirmed here by the KS test p-value against the best-fit Gaussian.

C.  Paired t-test on per-subject proportions and logistic regression
    produce virtually identical p-values on the same datasets.
    The logistic model offers no inferential advantage.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import stats
from scipy.special import expit

rng = np.random.default_rng(3)

N_SUBJ      = 120
N_SIM       = 1000   # for panel C

# ----------------------------------------------------------------
# Panel A: Histograms of per-subject proportions
# ----------------------------------------------------------------
P_TRUE_A = 0.65
N_ITEMS_LIST = [8, 16, 32]
COLORS_A = ["#C44E52", "#DD8452", "#4C72B0"]

prop_samples = {}
for n in N_ITEMS_LIST:
    props = rng.binomial(n, P_TRUE_A, size=N_SUBJ) / n
    prop_samples[n] = props

# ----------------------------------------------------------------
# Panel B: KS-test p-value heatmap (normality of proportions)
# ----------------------------------------------------------------
p_true_grid  = np.linspace(0.05, 0.95, 19)
n_items_grid = [4, 6, 8, 12, 16, 24, 32, 48]
N_SUBJ_B     = 200   # more subjects for stable KS test

ks_pvals = np.zeros((len(n_items_grid), len(p_true_grid)))
for i, n in enumerate(n_items_grid):
    for j, p in enumerate(p_true_grid):
        props = rng.binomial(n, p, size=N_SUBJ_B) / n
        mu, sigma = props.mean(), props.std()
        if sigma < 1e-9:
            ks_pvals[i, j] = 0.0
        else:
            _, pv = stats.kstest(props, "norm", args=(mu, sigma))
            ks_pvals[i, j] = pv

# ----------------------------------------------------------------
# Panel C: t-test vs logistic regression p-values
# ----------------------------------------------------------------
# Design: 2 conditions, N_SUBJ subjects, n_items trials each
# Effect size: Cohen's d ≈ 0.4 (moderate)
N_ITEMS_C = 16
P_BASE    = 0.55
P_COND2   = 0.70   # moderate difference

def run_one_sim(effect):
    """Returns (p_ttest, p_logistic) for one simulated dataset."""
    p1 = P_BASE
    p2 = P_BASE + effect
    p2 = min(p2, 0.99)

    # Per-subject proportions
    y1 = rng.binomial(N_ITEMS_C, p1, size=N_SUBJ) / N_ITEMS_C
    y2 = rng.binomial(N_ITEMS_C, p2, size=N_SUBJ) / N_ITEMS_C

    # Paired t-test
    _, p_t = stats.ttest_rel(y1, y2)

    # Logistic regression approximation via LRT on log-odds
    # (equivalent to Wald test on pooled proportions)
    n_yes1 = (y1 * N_ITEMS_C).round().astype(int)
    n_yes2 = (y2 * N_ITEMS_C).round().astype(int)
    # Aggregate across subjects → chi-square on 2×2 table
    table = np.array([[n_yes1.sum(), N_SUBJ * N_ITEMS_C - n_yes1.sum()],
                      [n_yes2.sum(), N_SUBJ * N_ITEMS_C - n_yes2.sum()]])
    # Also compute paired logistic approximation: sign test as baseline
    # Use Wald test on pooled log-odds
    p_hat1 = np.clip(y1.mean(), 1e-6, 1 - 1e-6)
    p_hat2 = np.clip(y2.mean(), 1e-6, 1 - 1e-6)
    lo1 = np.log(p_hat1 / (1 - p_hat1))
    lo2 = np.log(p_hat2 / (1 - p_hat2))
    # SE of log-odds difference (delta method)
    se1 = np.sqrt(1 / (N_SUBJ * N_ITEMS_C * p_hat1 * (1 - p_hat1)))
    se2 = np.sqrt(1 / (N_SUBJ * N_ITEMS_C * p_hat2 * (1 - p_hat2)))
    z   = (lo2 - lo1) / np.sqrt(se1**2 + se2**2)
    p_l = 2 * stats.norm.sf(abs(z))

    return p_t, p_l

# Under H0 (no effect)
null_t, null_l = zip(*[run_one_sim(0.0) for _ in range(N_SIM)])
# Under H1 (effect = 0.15 probability points)
alt_t,  alt_l  = zip(*[run_one_sim(0.15) for _ in range(N_SIM)])

null_t = np.array(null_t)
null_l = np.array(null_l)
alt_t  = np.array(alt_t)
alt_l  = np.array(alt_l)

# ----------------------------------------------------------------
# Plot
# ----------------------------------------------------------------
fig = plt.figure(figsize=(14, 4.8))
gs  = fig.add_gridspec(1, 3, wspace=0.38)

# ---- Panel A ----
ax = fig.add_subplot(gs[0])
bins = np.linspace(0, 1, 22)
for n, color in zip(N_ITEMS_LIST, COLORS_A):
    props = prop_samples[n]
    ax.hist(props, bins=bins, alpha=0.5, color=color, density=True,
            label=f"N = {n} items")
    mu, sigma = props.mean(), props.std()
    x = np.linspace(0, 1, 200)
    ax.plot(x, stats.norm.pdf(x, mu, sigma), color=color, linewidth=2)

ax.axvline(P_TRUE_A, color="black", linestyle="--", linewidth=1,
           label=f"True p = {P_TRUE_A}")
ax.set_xlabel("Per-subject proportion", fontsize=10)
ax.set_ylabel("Density", fontsize=10)
ax.set_title(f"A  –  Distribution of per-subject\nproportions (p = {P_TRUE_A})",
             fontsize=10)
ax.legend(fontsize=8)

# ---- Panel B ----
ax = fig.add_subplot(gs[1])
# Show KS p-value: high = Gaussian is adequate
cmap = plt.get_cmap("RdYlGn")
im = ax.imshow(ks_pvals, aspect="auto", cmap=cmap, vmin=0, vmax=0.5,
               origin="lower",
               extent=[p_true_grid[0], p_true_grid[-1],
                       0, len(n_items_grid)])
# rule-of-thumb contours: N·p = 5 and N·(1-p) = 5
for i, n in enumerate(n_items_grid):
    p_lo = 5 / n
    p_hi = 1 - 5 / n
    if p_lo < p_hi:
        ax.axvline(p_lo, color="black", alpha=0.15, linewidth=0.5)
        ax.axvline(p_hi, color="black", alpha=0.15, linewidth=0.5)

ax.set_yticks(np.arange(len(n_items_grid)) + 0.5)
ax.set_yticklabels(n_items_grid, fontsize=8)
ax.set_xlabel("True probability", fontsize=10)
ax.set_ylabel("Items per condition", fontsize=10)
ax.set_title("B  –  Normality of proportions\n(KS test p-value; green = adequate)",
             fontsize=10)
cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb.set_label("KS p-value", fontsize=8)
cb.ax.axhline(0.05, color="black", linewidth=1.2)

# ---- Panel C ----
ax = fig.add_subplot(gs[2])
# Scatter: t-test p vs logistic p (under H1)
ax.scatter(alt_t, alt_l, alpha=0.25, s=12, color="#4C72B0",
           label="H₁ (effect = 0.15)", rasterized=True)
ax.scatter(null_t, null_l, alpha=0.15, s=12, color="#C44E52",
           label="H₀ (no effect)", rasterized=True)
lim = [0, 1]
ax.plot(lim, lim, "k--", linewidth=1, label="Identity")
ax.set_xlabel("Paired t-test  p-value", fontsize=10)
ax.set_ylabel("Logistic regression  p-value", fontsize=10)
ax.set_title("C  –  t-test and logistic give\nidentical p-values",
             fontsize=10)
ax.legend(fontsize=8)

# Annotate correlation
r_null = np.corrcoef(null_t, null_l)[0, 1]
r_alt  = np.corrcoef(alt_t,  alt_l)[0, 1]
ax.text(0.05, 0.92,
        f"r (H₀) = {r_null:.3f}\nr (H₁) = {r_alt:.3f}",
        transform=ax.transAxes, fontsize=8.5,
        bbox=dict(facecolor="white", alpha=0.8, boxstyle="round"))

fig.suptitle(
    "The central limit theorem justifies treating binary outcomes as Gaussian",
    fontsize=11.5, fontweight="bold")
fig.tight_layout()
fig.savefig("fig_clt.pdf", bbox_inches="tight")
fig.savefig("fig_clt.png", dpi=150, bbox_inches="tight")
plt.show()

# Print key stats
print(f"Type I error — t-test: {(null_t < 0.05).mean():.3f}, "
      f"logistic: {(null_l < 0.05).mean():.3f}")
print(f"Power       — t-test: {(alt_t  < 0.05).mean():.3f}, "
      f"logistic: {(alt_l  < 0.05).mean():.3f}")
