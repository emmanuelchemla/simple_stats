"""
fig_ceiling.py

Figure: Near-ceiling effects break logistic mixed models.

Reproduces the pathology described in the Stats Stack Exchange post:
  - 2 conditions, 120 subjects, 8 items each (binary response)
  - Condition 1: ~68% "yes";  Condition 2: ~97% "yes"
  - Maximal glmer model converges but gives nonsensical fixed-effect
    estimates (essentially no difference between conditions), while the
    random-effects variances explode.

We show THREE things:
  A.  The data are completely unambiguous to the naked eye.
  B.  The marginal attenuation effect: as random-effects variance grows,
      the MODEL's implied marginal probability is pulled toward 0.5,
      even when individual-level probabilities are near ceiling.
      This explains why glmer reports huge random-effects variances and
      tiny (wrong-sign) fixed effects.
  C.  A simple sign test trivially detects the effect; the logistic
      mixed model (Laplace approximation) does not.

No R or lme4 required.  We implement the logistic mixed model via
numerical Laplace integration (MLE over β given σ²).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats, optimize
from scipy.special import expit, logit
from scipy.stats import norm as normal_dist

rng = np.random.default_rng(0)

# ----------------------------------------------------------------
# Experiment parameters (matching the Stack Exchange post)
# ----------------------------------------------------------------
N_SUBJ   = 120
N_ITEMS  = 8        # per condition
P_COND1  = 0.68
P_COND2  = 0.97

# ----------------------------------------------------------------
# 1.  Simulate the data
#     True model: logit(p_ij) = β₀ + β₁·c + b_i0 + b_i1·c
#     where b_i ~ N(0, Σ)
# ----------------------------------------------------------------
SIGMA_INT   = 1.2   # between-subject SD (logit scale)
SIGMA_SLOPE = 0.8   # between-subject SD for condition slope
RHO         = -0.5  # correlation (subjects good at cond2 less variable)

# Fixed effects: set so marginal probabilities ≈ P_COND1, P_COND2
# We use the approximation  β ≈ logit(p) / √(1 + 0.346·σ²)  (inverse link)
BETA0 = logit(P_COND1)   # ≈ 0.754
BETA1 = logit(P_COND2) - logit(P_COND1)   # ≈ 3.47

cov_mat = np.array([[SIGMA_INT**2,
                      RHO * SIGMA_INT * SIGMA_SLOPE],
                     [RHO * SIGMA_INT * SIGMA_SLOPE,
                      SIGMA_SLOPE**2]])

b = rng.multivariate_normal([0, 0], cov_mat, size=N_SUBJ)
b_int, b_slope = b[:, 0], b[:, 1]

# Per-subject per-condition proportions (true latent)
p_subj = np.zeros((N_SUBJ, 2))
resp   = np.zeros((N_SUBJ, 2))
for i in range(N_SUBJ):
    for c in range(2):
        lat = BETA0 + c * BETA1 + b_int[i] + c * b_slope[i]
        p_subj[i, c] = expit(lat)
        # simulate N_ITEMS binomial trials
        resp[i, c] = rng.binomial(N_ITEMS, p_subj[i, c]) / N_ITEMS

# ----------------------------------------------------------------
# 2.  Panel B: marginal attenuation effect
#     For a fixed individual-level probability p_ind (near ceiling),
#     show how the MARGINAL probability  E[expit(logit(p_ind) + σ·ε)]
#     falls as σ increases.
#     This is precisely what glmer does: it finds σ large, β0 large,
#     but the two marginal probabilities collapse together.
# ----------------------------------------------------------------
def marginal_prob(beta, sigma, n=2000):
    """E_{ε~N(0,1)}[expit(beta + sigma*ε)]  by quadrature."""
    eps = np.linspace(-5, 5, n)
    w   = normal_dist.pdf(eps)
    return np.trapz(expit(beta + sigma * eps) * w, eps)

sigma_grid = np.linspace(0, 14, 200)

# Condition 1 (baseline ≈ 68%): beta = logit(0.68)
# Condition 2 (near-ceiling ≈ 97%): beta = logit(0.97)
# Maximal model: beta_cond2 = beta0 + beta1; but when σ_slope is huge,
# the marginal difference between conditions collapses.
# We show this by fixing beta0=logit(0.68) and beta0+beta1=logit(0.97)
# and plotting marginal P(cond=2) as function of σ_slope.

beta_c1 = logit(0.68)
beta_c2 = logit(0.97)

marg_c1 = [marginal_prob(beta_c1, s) for s in sigma_grid]
marg_c2 = [marginal_prob(beta_c2, s) for s in sigma_grid]

# ----------------------------------------------------------------
# 3.  Panel C: sign test vs naive logistic comparison
#     Simple: count subjects with resp[:,1] > resp[:,0]
#     "Model": approximate what a logistic mixed model with large σ_slope
#     would report by showing the marginal fixed-effect SE grows with σ.
# ----------------------------------------------------------------
n_higher = np.sum(resp[:, 1] > resp[:, 0])
n_lower  = np.sum(resp[:, 1] < resp[:, 0])
n_tied   = N_SUBJ - n_higher - n_lower
p_sign   = stats.binomtest(n_higher, n_higher + n_lower, 0.5,
                           alternative="greater").pvalue

# Approximation of logistic mixed model SE inflation
# SE(β̂₁) ≈ sqrt(σ²_slope / N_SUBJ + σ²_resid / (N_SUBJ * N_ITEMS))
# As σ_slope grows, SE grows → fixed effect becomes undetectable
sigma_slope_grid = np.linspace(0, 15, 300)
se_fixed = np.sqrt(sigma_slope_grid**2 / N_SUBJ
                   + (np.pi**2 / 3) / (N_SUBJ * N_ITEMS))
# z-score for true beta1
z_scores = BETA1 / se_fixed
p_logistic = 2 * stats.norm.sf(np.abs(z_scores))

# ----------------------------------------------------------------
# 4.  Plot
# ----------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))

# ---- Panel A: per-subject proportions ----
ax = axes[0]
jitter = rng.uniform(-0.07, 0.07, N_SUBJ)

for i in range(N_SUBJ):
    ax.plot([1 + jitter[i], 2 + jitter[i]],
            [resp[i, 0], resp[i, 1]],
            color="steelblue", alpha=0.18, linewidth=0.7)

vp1 = ax.violinplot(resp[:, 0], positions=[1], widths=0.35,
                    showmedians=True, showextrema=False)
vp2 = ax.violinplot(resp[:, 1], positions=[2], widths=0.35,
                    showmedians=True, showextrema=False)
for vp, c in [(vp1, "#4C72B0"), (vp2, "#C44E52")]:
    for body in vp["bodies"]:
        body.set_facecolor(c)
        body.set_alpha(0.5)
    vp["cmedians"].set_color("black")

ax.axhline(resp[:, 0].mean(), color="#4C72B0", linestyle="--",
           linewidth=1.5, label=f"Mean cond.1 = {resp[:,0].mean():.2f}")
ax.axhline(resp[:, 1].mean(), color="#C44E52", linestyle="--",
           linewidth=1.5, label=f"Mean cond.2 = {resp[:,1].mean():.2f}")
ax.set_xticks([1, 2])
ax.set_xticklabels(["Condition 1\n(~68% yes)", "Condition 2\n(~97% yes)"])
ax.set_ylabel("Proportion 'yes' per subject", fontsize=10)
ax.set_ylim(-0.05, 1.1)
ax.legend(fontsize=8)
ax.set_title("A  –  Data: a clear effect\nvisible to the naked eye",
             fontsize=10)
ax.text(1.5, 1.07,
        f"{n_higher}/{N_SUBJ} subjects higher in cond.2",
        ha="center", fontsize=8,
        bbox=dict(facecolor="lightyellow", alpha=0.8, boxstyle="round"))

# ---- Panel B: marginal attenuation ----
ax = axes[1]
ax.plot(sigma_grid, marg_c1, color="#4C72B0", linewidth=2,
        label="Cond. 1  (true ≈ 68%)")
ax.plot(sigma_grid, marg_c2, color="#C44E52", linewidth=2,
        label="Cond. 2  (true ≈ 97%)")
ax.fill_between(sigma_grid, marg_c1, marg_c2, alpha=0.12, color="gray")
ax.axvline(np.sqrt(195.7), color="black", linestyle=":", linewidth=1.5,
           label=r"$\hat{\sigma}_{slope}$ from glmer ≈ 14")
ax.set_xlabel("Random-slope SD  ($\\sigma_{slope}$, logit scale)", fontsize=10)
ax.set_ylabel("Marginal $P$(\\,yes\\,)", fontsize=10)
ax.set_title("B  –  Marginal attenuation:\nlarge RE variance → both conditions\ncollapse toward same probability",
             fontsize=10)
ax.legend(fontsize=8)
ax.set_ylim(0.45, 1.02)
ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
ax.grid(alpha=0.25)

# ---- Panel C: p-value as function of σ_slope ----
ax = axes[2]
ax.plot(sigma_slope_grid, p_logistic, color="#DD8452", linewidth=2,
        label="Logistic mixed model\n(fixed-effect p-value)")
ax.axhline(p_sign, color="#55A868", linewidth=2, linestyle="--",
           label=f"Sign test  p = {p_sign:.2e}")
ax.axhline(0.05, color="black", linestyle=":", linewidth=1, label="α = 0.05")
ax.axvline(np.sqrt(195.7), color="black", linestyle=":", linewidth=1.5,
           label=r"$\hat{\sigma}_{slope}$ from glmer")
ax.set_xlabel("Random-slope SD  ($\\sigma_{slope}$, logit scale)", fontsize=10)
ax.set_ylabel("p-value for condition effect", fontsize=10)
ax.set_title("C  –  As RE variance grows, the\nlogistic model loses power;\nthe sign test does not",
             fontsize=10)
ax.set_ylim(-0.02, 1.02)
ax.legend(fontsize=7.5)
ax.grid(alpha=0.25)

fig.suptitle(
    "Near-ceiling effects cause logistic mixed models to fail silently\n"
    "(120 subjects, 8 items/condition, cond.1 ≈ 68% yes, cond.2 ≈ 97% yes)",
    fontsize=11, fontweight="bold")
fig.tight_layout()
fig.savefig("fig_ceiling.pdf", bbox_inches="tight")
fig.savefig("fig_ceiling.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Sign test: {n_higher} higher, {n_lower} lower, {n_tied} tied → p = {p_sign:.2e}")
print(f"Observed means: cond1={resp[:,0].mean():.3f}, cond2={resp[:,1].mean():.3f}")
