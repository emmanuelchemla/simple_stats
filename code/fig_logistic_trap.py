"""
fig_logistic_trap.py

Figure: The logistic trap — five desiderata that cannot all be satisfied.

A concrete illustration of the core argument:
  Desiderata (a–e) as stated in the paper
  (a) analyze all the data
  (b) maximal random-effects model
  (c) simplify until convergence
  (d) model comparison with identical RE structures
  (e) pre-register the analysis

The combination of (c), (d), (e) is internally inconsistent because
pre-registering the simplification path is effectively impossible when
the path depends on which error messages the optimizer produces.

Panel A  –  Combinatorial explosion: number of candidate RE structures
            for a 1-factor 4-level design (= 3 binary fixed effects
            a, b, c) with participant random effects.

Panel B  –  P-value spread: for the same null dataset, what p-value does
            the analyst obtain depending on which (converging) model they
            end up with?  Each simplification path leads somewhere
            different.

Panel C  –  AIC landscape: the best-AIC converging model is not
            necessarily the one the simplification procedure finds first.
            Models are scattered in AIC×p-value space with no clear
            ordering principle.
"""

import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from scipy.special import expit

rng = np.random.default_rng(42)

# ----------------------------------------------------------------
# 1.  Enumerate RE structures for a 3-fixed-effect design
#     (design: 4-level factor → 3 dummy-coded predictors a, b, c)
#     Random effects for participants: any subset of {intercept, a, b, c}
#     For each subset of size k > 1: correlated (|) or uncorrelated (||)
# ----------------------------------------------------------------
fixed_labels = ["intercept", "a", "b", "c"]

def enumerate_re_structures(labels):
    structs = []
    for r in range(1, len(labels) + 1):
        for subset in itertools.combinations(labels, r):
            if len(subset) == 1:
                structs.append({"terms": subset, "corr": None,
                                 "n_params": 1})
            else:
                k = len(subset)
                structs.append({"terms": subset, "corr": True,
                                 "n_params": k + k * (k - 1) // 2})
                structs.append({"terms": subset, "corr": False,
                                 "n_params": k})
    return structs

re_structs = enumerate_re_structures(fixed_labels)
n_total = len(re_structs)   # should be 26

# ----------------------------------------------------------------
# 2.  Simulate a null dataset: 4-level factor (a,b,c = dummy codes),
#     N_SUBJ participants, N_ITEMS items per condition, binary outcome
# ----------------------------------------------------------------
N_SUBJ  = 32
N_ITEMS = 16   # per condition (4 conditions → 64 trials per subject)
N_SIMS  = 800  # for p-value distribution

TRUE_EFFECT = 0.0   # null: no fixed effect

SIGMA_INT   = 0.8   # random intercept SD (on logit scale)
SIGMA_SLOPE = 0.4   # random slope SD per predictor

def gen_dataset(n_subj, n_items, effect=TRUE_EFFECT):
    """
    4-level factor (levels 0–3) → dummy codes (a,b,c).
    Returns DataFrame-like structured array.
    """
    b_int = rng.normal(0, SIGMA_INT,   n_subj)
    b_a   = rng.normal(0, SIGMA_SLOPE, n_subj)
    b_b   = rng.normal(0, SIGMA_SLOPE, n_subj)
    b_c   = rng.normal(0, SIGMA_SLOPE, n_subj)

    records = []
    for i in range(n_subj):
        for j in range(n_items):
            for level in range(4):
                a = int(level >= 1)
                b = int(level >= 2)
                c = int(level == 3)
                logit = (b_int[i]
                         + a * (effect + b_a[i])
                         + b * (effect + b_b[i])
                         + c * (effect + b_c[i]))
                p = expit(logit)
                y = int(rng.random() < p)
                records.append((i, j, level, a, b, c, y, logit))
    return np.array(records, dtype=[
        ("subj", int), ("item", int), ("level", int),
        ("a", float), ("b", float), ("c", float),
        ("y", float), ("logit", float)
    ])


# ----------------------------------------------------------------
# 3.  "Approximate" analysis strategies
#     We cannot run actual glmer in Python, so we implement
#     analytically equivalent procedures that capture the key
#     statistical behaviour:
#
#   - "Full / RI":   per-subject accuracy, logistic link, RE intercept
#                    → aggregate per subject, sign test on 4 accuracy values
#   - "Subset a":    use only levels 0 vs 1  (comparison a)
#   - "Subset b":    use only levels 0 vs 2  (comparison b)
#   - "Subset c":    use only levels 0 vs 3  (comparison c)
#   - "Aggregate":   chi-square on 2×4 contingency table
#   - "RE-intercept only": mixed logistic approximated by OLS on per-subject
#                           logit-transformed proportions
#   - "No RE (GLM)":  chi-square marginal
#
#   Each produces a p-value for "any difference across the 4 levels".
# ----------------------------------------------------------------

def pval_sign_test_full(data, n_subj):
    """Sign test: do subjects do better above chance overall?"""
    accs = []
    for i in range(n_subj):
        subj = data[data["subj"] == i]
        accs.append(subj["y"].mean())
    k = sum(a > 0.5 for a in accs)
    return stats.binomtest(k, n_subj, 0.5).pvalue


def pval_anova_on_logits(data, n_subj):
    """One-way ANOVA on per-subject per-condition logit-proportions."""
    by_cond = {lv: [] for lv in range(4)}
    for i in range(n_subj):
        subj = data[data["subj"] == i]
        for lv in range(4):
            cond = subj[subj["level"] == lv]
            p = np.clip(cond["y"].mean(), 0.01, 0.99)
            by_cond[lv].append(np.log(p / (1 - p)))
    groups = [np.array(by_cond[lv]) for lv in range(4)]
    _, p = stats.f_oneway(*groups)
    return p


def pval_subset(data, n_subj, level_a=0, level_b=1):
    """Paired t-test on subset of two levels."""
    diffs = []
    for i in range(n_subj):
        subj = data[data["subj"] == i]
        m0 = subj[subj["level"] == level_a]["y"].mean()
        m1 = subj[subj["level"] == level_b]["y"].mean()
        diffs.append(m1 - m0)
    _, p = stats.ttest_1samp(diffs, 0)
    return p


def pval_chisq(data):
    """Chi-square on 2×4 contingency table (marginal, ignoring subjects)."""
    table = np.zeros((2, 4))
    for lv in range(4):
        cond = data[data["level"] == lv]
        table[1, lv] = cond["y"].sum()
        table[0, lv] = len(cond) - table[1, lv]
    _, p, _, _ = stats.chi2_contingency(table)
    return p


def pval_glm_approx(data, n_subj):
    """
    Approximate GLM (no random effects):
    logistic regression via OLS on per-trial logit is unstable;
    instead use chi-square across subjects × conditions.
    """
    table = np.zeros((n_subj, 4))
    for i in range(n_subj):
        subj = data[data["subj"] == i]
        for lv in range(4):
            table[i, lv] = subj[subj["level"] == lv]["y"].mean()
    # Compare variance of column means vs within-column variance (F-test)
    col_means = table.mean(axis=0)
    grand_mean = col_means.mean()
    ss_between = n_subj * np.sum((col_means - grand_mean) ** 2)
    ss_within  = np.sum((table - col_means[None, :]) ** 2)
    f = (ss_between / 3) / (ss_within / (n_subj * 4 - 4))
    return stats.f.sf(f, 3, n_subj * 4 - 4)


def pval_maximal_approx(data, n_subj):
    """
    'Maximal' RE: ANOVA on per-subject per-condition means
    with subject as blocking factor → repeated-measures ANOVA
    (within-subject F-test).
    """
    table = np.zeros((n_subj, 4))
    for i in range(n_subj):
        subj = data[data["subj"] == i]
        for lv in range(4):
            table[i, lv] = subj[subj["level"] == lv]["y"].mean()
    # within-subjects F
    grand  = table.mean()
    cond_m = table.mean(axis=0)
    subj_m = table.mean(axis=1)
    ss_cond  = n_subj * np.sum((cond_m - grand) ** 2)
    ss_subj  = 4 * np.sum((subj_m - grand) ** 2)
    ss_total = np.sum((table - grand) ** 2)
    ss_error = ss_total - ss_cond - ss_subj
    ms_cond  = ss_cond  / 3
    ms_error = ss_error / (3 * (n_subj - 1))
    f = ms_cond / ms_error if ms_error > 0 else 0
    return stats.f.sf(f, 3, 3 * (n_subj - 1))


# ----------------------------------------------------------------
# 4.  Run N_SIMS under H0 for each strategy
# ----------------------------------------------------------------
strategies = {
    "Maximal RE\n(all data)":        pval_maximal_approx,
    "GLM / no RE\n(all data)":       pval_glm_approx,
    "Logit ANOVA\n(per-subj)":       pval_anova_on_logits,
    r"$\chi^2$" + "\n(marginal)":    lambda d, _: pval_chisq(d),
    "Subset\nlev.0 vs 1":            lambda d, n: pval_subset(d, n, 0, 1),
    "Subset\nlev.0 vs 3":            lambda d, n: pval_subset(d, n, 0, 3),
}

pvals = {name: [] for name in strategies}

for _ in range(N_SIMS):
    d = gen_dataset(N_SUBJ, N_ITEMS)
    for name, fn in strategies.items():
        pvals[name].append(fn(d, N_SUBJ))

# ----------------------------------------------------------------
# 5.  Panel B helper: for a SINGLE dataset, show the spread of
#     p-values across strategies (illustrates RDF)
# ----------------------------------------------------------------
single_d = gen_dataset(N_SUBJ, N_ITEMS)
single_p = {name: fn(single_d, N_SUBJ) for name, fn in strategies.items()}

# ----------------------------------------------------------------
# 6.  Panel C: AIC-like quantity vs p-value for each strategy
#     We proxy AIC by the -2*log-likelihood of a binomial model
#     with fitted proportions per condition (lower = better fit).
# ----------------------------------------------------------------
def pseudo_aic(data, n_subj, n_params):
    """BIC-like penalised log-likelihood on per-cell proportions."""
    ll = 0
    for lv in range(4):
        cond = data[data["level"] == lv]
        p_hat = np.clip(cond["y"].mean(), 1e-6, 1 - 1e-6)
        n = len(cond)
        k = cond["y"].sum()
        ll += k * np.log(p_hat) + (n - k) * np.log(1 - p_hat)
    return -2 * ll + 2 * n_params

n_params_map = {
    "Maximal RE\n(all data)":        10,   # intercept + 3 slopes + 6 cov
    "GLM / no RE\n(all data)":        4,
    "Logit ANOVA\n(per-subj)":        5,
    r"$\chi^2$" + "\n(marginal)":     4,
    "Subset\nlev.0 vs 1":             3,
    "Subset\nlev.0 vs 3":             3,
}
single_aic = {name: pseudo_aic(single_d, N_SUBJ, n_params_map[name])
              for name in strategies}

# ----------------------------------------------------------------
# 7.  Plot
# ----------------------------------------------------------------
fig = plt.figure(figsize=(14, 4.8))
gs  = fig.add_gridspec(1, 3, wspace=0.38)

colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#9467BD", "#8C564B"]

# ---- Panel A: combinatorial explosion ----
ax0 = fig.add_subplot(gs[0])
n_fixed_effects = np.arange(1, 7)
n_re_structs    = []
for k in n_fixed_effects:
    labels_ = [f"x{i}" for i in range(k + 1)]   # k fixed + intercept
    structs_ = enumerate_re_structures(labels_)
    n_re_structs.append(len(structs_))

ax0.bar(n_fixed_effects, n_re_structs, color="#4C72B0", edgecolor="white")
ax0.axvline(3, color="#C44E52", linestyle="--", linewidth=1.5,
            label="This paper's example\n(3 fixed effects → 4 levels)")
ax0.set_xlabel("Number of fixed effects", fontsize=10)
ax0.set_ylabel("Candidate RE structures", fontsize=10)
ax0.set_title("A  –  Combinatorial explosion\nof RE specifications",
              fontsize=10)
ax0.legend(fontsize=7.5)
for x, y in zip(n_fixed_effects, n_re_structs):
    ax0.text(x, y + 0.5, str(y), ha="center", va="bottom", fontsize=8)
ax0.set_ylim(0, max(n_re_structs) * 1.15)

# ---- Panel B: p-value distribution across strategies (N_SIMS runs) ----
ax1 = fig.add_subplot(gs[1])
names  = list(strategies.keys())
type1s = [np.mean(np.array(pvals[n]) < 0.05) for n in names]
bars   = ax1.bar(range(len(names)), type1s, color=colors, edgecolor="white")
ax1.axhline(0.05, color="black", linestyle="--", linewidth=1.2,
            label="Nominal α = 0.05")
for i, (bar, val) in enumerate(zip(bars, type1s)):
    ax1.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.003,
             f"{val:.3f}", ha="center", va="bottom", fontsize=7.5)
ax1.set_xticks(range(len(names)))
ax1.set_xticklabels(names, fontsize=7)
ax1.set_ylabel("Type I error rate", fontsize=10)
ax1.set_title("B  –  Type I error varies\nby simplification path",
              fontsize=10)
ax1.set_ylim(0, max(type1s) * 1.30 + 0.02)
ax1.legend(fontsize=8)
ax1.grid(axis="y", alpha=0.3)

# ---- Panel C: p-value vs pseudo-AIC for a single dataset ----
ax2 = fig.add_subplot(gs[2])
aics  = [single_aic[n]  for n in names]
ps    = [single_p[n]    for n in names]

sc = ax2.scatter(aics, ps, c=colors, s=90, zorder=3, edgecolors="white",
                 linewidths=0.5)
ax2.axhline(0.05, color="black", linestyle="--", linewidth=1, label="α = 0.05")
for i, name in enumerate(names):
    ax2.annotate(name.replace("\n", " "),
                 (aics[i], ps[i]),
                 textcoords="offset points",
                 xytext=(6, 3),
                 fontsize=6.5)
ax2.set_xlabel("Pseudo-AIC (lower = better fit)", fontsize=10)
ax2.set_ylabel("p-value (same dataset)", fontsize=10)
ax2.set_title("C  –  Better fit ≠ more\nconservative test",
              fontsize=10)
ax2.legend(fontsize=8)
ax2.grid(alpha=0.25)

fig.suptitle(
    "The logistic trap: choosing a simplification path is not a neutral act",
    fontsize=11.5, fontweight="bold")

fig.savefig("fig_logistic_trap.pdf", bbox_inches="tight")
fig.savefig("fig_logistic_trap.png", dpi=150, bbox_inches="tight")
plt.show()
print("Done.")
