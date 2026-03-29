"""
Figure 4: Why log-odds are not a psychologically meaningful scale

Logistic mixed models express effects in log-odds (logits).  This
figure shows that a constant shift on the log-odds scale corresponds
to wildly different probability differences depending on the baseline,
making comparisons across conditions or experiments misleading.

Panel A: mapping of probability to log-odds (the compression at extremes).
Panel B: the same log-odds effect (Δlogit = 0.5) translates to very
         different Δp depending on the baseline probability.
Panel C: a worked example – two experiments with identical Δlogit but
         psychologically very different Δp.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit   # sigmoid / logistic

# ------------------------------------------------------------------
# Panel A – prob → logit transformation
# ------------------------------------------------------------------
p   = np.linspace(0.01, 0.99, 500)
lor = np.log(p / (1 - p))

# ------------------------------------------------------------------
# Panel B – effect size in Δp for a fixed Δlogit
# ------------------------------------------------------------------
delta_logit = 0.5
baselines   = np.linspace(0.05, 0.95, 400)
logit_base  = np.log(baselines / (1 - baselines))
delta_p     = expit(logit_base + delta_logit) - baselines

# ------------------------------------------------------------------
# Panel C – two hypothetical experiments
# ------------------------------------------------------------------
# Exp 1: easy condition, high baseline
b1  = 0.85
# Exp 2: hard condition, low baseline
b2  = 0.30
dlogit = 0.5
dp1 = expit(np.log(b1/(1-b1)) + dlogit) - b1
dp2 = expit(np.log(b2/(1-b2)) + dlogit) - b2

# ------------------------------------------------------------------
# Plot
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

# ---- Panel A ----
ax = axes[0]
ax.plot(p, lor, color="#4C72B0", linewidth=2)
ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
ax.set_xlabel("Probability", fontsize=11)
ax.set_ylabel("Log-odds (logit)", fontsize=11)
ax.set_title("A  –  The logit transformation", fontsize=11)
ax.set_xlim(0, 1)
ax.set_ylim(-5, 5)
# annotate compression zones
ax.fill_betweenx([-5, 5], 0, 0.15, alpha=0.10, color="#C44E52")
ax.fill_betweenx([-5, 5], 0.85, 1.0, alpha=0.10, color="#C44E52")
ax.text(0.05, 4, "compressed", fontsize=8, color="#C44E52", ha="left")
ax.text(0.88, 4, "compressed", fontsize=8, color="#C44E52", ha="left")

# ---- Panel B ----
ax = axes[1]
ax.plot(baselines, delta_p, color="#DD8452", linewidth=2)
ax.axhline(delta_p.mean(), color="gray", linestyle="--", linewidth=1,
           label=f"Mean Δp = {delta_p.mean():.3f}")
ax.set_xlabel("Baseline probability", fontsize=11)
ax.set_ylabel("Δ probability  (for Δlogit = 0.5)", fontsize=11)
ax.set_title("B  –  Same Δlogit, different Δp", fontsize=11)
ax.set_xlim(0, 1)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# ---- Panel C ----
ax = axes[2]
exps   = ["Exp 1\n(easy, baseline=0.85)", "Exp 2\n(hard, baseline=0.30)"]
dp_val = [dp1, dp2]
bars   = ax.bar(exps, dp_val, color=["#4C72B0", "#C44E52"],
                edgecolor="white", width=0.45)
ax.axhline(0, color="black", linewidth=0.8)
for bar, val in zip(bars, dp_val):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.003,
            f"Δp = {val:.3f}", ha="center", va="bottom", fontsize=10,
            fontweight="bold")
ax.text(0.5, max(dp_val)*1.35,
        f"Both have identical\nΔlogit = {dlogit:.1f}",
        ha="center", va="top", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
        transform=ax.transData)
ax.set_ylabel("Δ probability", fontsize=11)
ax.set_title("C  –  Identical logit effect,\ndifferent psychological meaning",
             fontsize=11)
ax.set_ylim(0, max(dp_val) * 1.6)
ax.grid(axis="y", alpha=0.3)

fig.suptitle(
    "Log-odds effects are not interpretable without the baseline",
    fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig("fig4_logodds.pdf", bbox_inches="tight")
fig.savefig("fig4_logodds.png", dpi=150, bbox_inches="tight")
plt.show()
