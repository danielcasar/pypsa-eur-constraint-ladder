"""
Phase 1: synthetic-data validation of a TVP (time-varying parameter)
regression in PyMC. Build a small panel with KNOWN time-varying beta,
fit the TVP model, verify recovery of the truth.

Model fitted (PyMC):
    y_{t, c, n} = alpha_c + beta_t * X_{t, c, n} + eps,  eps ~ N(0, sigma_eps^2)
    beta_t = beta_{t-1} + eta_t,                          eta ~ N(0, sigma_eta^2)
    alpha_c ~ N(0, 1)   (country FE)

Synthetic ground truth:
  T = 11 years, C = 22 countries, N_per_cy = 24 obs (one per hour-like draw).
  True beta_t: a smooth random walk we will try to recover.
  True alpha_c: drawn iid from N(0, 0.5^2).
  Observation noise sigma_eps = 0.3.

Outputs:
  Analysis/results/tvp_phase1_validation_summary.txt
  Analysis/results/plots/tvp_phase1_synthetic_recovery.{pdf,png}
"""

from __future__ import annotations

import warnings
from pathlib import Path

import arviz as az
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

warnings.filterwarnings("ignore")
matplotlib.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial"],
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
})

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PLOTS = RESULTS / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------
SEED = 42
rng = np.random.default_rng(SEED)

T = 11          # years 0..T-1 (matching 2015..2025)
C = 22          # countries
N_per = 24      # observations per (country, year) — mirrors 24 hours

# True smooth beta trajectory: starts near zero, drifts negatively over time
# (mimicking the "solar coefficient gets more negative as deployment grows")
true_beta = np.zeros(T)
true_beta[0] = -0.20
true_inno = rng.normal(0, 0.10, T - 1)
for t in range(1, T):
    true_beta[t] = true_beta[t-1] + true_inno[t-1]

true_alpha = rng.normal(0, 0.5, C)
sigma_eps_true = 0.30

print(f"True beta trajectory (T = {T} years):")
for t, b in enumerate(true_beta):
    print(f"  y{t:>2}: {b:+.3f}")

# Generate panel
rows = []
for t in range(T):
    for c in range(C):
        for n in range(N_per):
            X = rng.uniform(0.05, 0.5)
            y = true_alpha[c] + true_beta[t] * X + rng.normal(0, sigma_eps_true)
            rows.append({"year_idx": t, "country_idx": c, "X": X, "y": y})
df = pd.DataFrame(rows)
print(f"\nSynthetic panel: {len(df)} observations "
      f"({T} years x {C} countries x {N_per} per cell)")


# ---------------------------------------------------------------------------
# PyMC TVP model
# ---------------------------------------------------------------------------
coords = {"year": np.arange(T), "country": np.arange(C)}
with pm.Model(coords=coords) as model:
    # Hyperprior on smoothness of beta evolution (estimated from data)
    sigma_eta = pm.HalfNormal("sigma_eta", sigma=0.20)

    # Random-walk beta: parameterise via initial value + cumulative innovations
    beta_init = pm.Normal("beta_init", mu=0.0, sigma=1.0)
    beta_inn  = pm.Normal("beta_inn", mu=0.0, sigma=sigma_eta, shape=T - 1)
    # beta_t = beta_init + sum_{s<=t} beta_inn_s
    beta = pm.Deterministic(
        "beta",
        pm.math.concatenate([[beta_init], beta_init + pm.math.cumsum(beta_inn)]),
        dims="year",
    )

    # Country fixed effects
    alpha = pm.Normal("alpha", mu=0.0, sigma=1.0, dims="country")

    # Observation variance
    sigma_eps = pm.HalfNormal("sigma_eps", sigma=1.0)

    # Likelihood
    mu = alpha[df["country_idx"].values] + beta[df["year_idx"].values] * df["X"].values
    pm.Normal("y_obs", mu=mu, sigma=sigma_eps, observed=df["y"].values)


# ---------------------------------------------------------------------------
# Sample
# ---------------------------------------------------------------------------
print("\nSampling PyMC NUTS ...")
with model:
    trace = pm.sample(
        draws=1500, tune=1500, chains=4, cores=4,
        target_accept=0.95, random_seed=SEED, progressbar=False,
    )

# Diagnostics
print("\nDiagnostics:")
print(az.summary(trace, var_names=["beta", "sigma_eta", "sigma_eps"]))


# ---------------------------------------------------------------------------
# Compare posterior beta to truth
# ---------------------------------------------------------------------------
beta_post = trace.posterior["beta"].stack(sample=("chain", "draw"))
beta_med   = beta_post.median(dim="sample").values
beta_lo    = beta_post.quantile(0.025, dim="sample").values
beta_hi    = beta_post.quantile(0.975, dim="sample").values

# Coverage: does truth lie inside 95% credible interval?
in_band = ((true_beta >= beta_lo) & (true_beta <= beta_hi)).mean()
rmse_post = float(np.sqrt(((beta_med - true_beta) ** 2).mean()))

out = []
out.append("=" * 72)
out.append("TVP synthetic-data validation (Phase 1)")
out.append("=" * 72)
out.append("\nModel: y = alpha_c + beta_t * X + eps;  beta = random walk; "
           "alpha = country FE.")
out.append(f"Panel: {T} years x {C} countries x {N_per} obs/cell = {len(df)}.")
out.append("\nPosterior summary of key parameters:")
summ = az.summary(trace, var_names=["sigma_eta", "sigma_eps"], round_to=4)
out.append(summ.to_string())

out.append("\nBeta trajectory recovery:")
out.append(f"  {'year':>4}  {'true_beta':>10}  {'post_med':>10}  "
           f"{'lo_2.5':>10}  {'hi_97.5':>10}  {'in_band':>8}")
out.append("-" * 60)
for t in range(T):
    inband = "Y" if (true_beta[t] >= beta_lo[t] and true_beta[t] <= beta_hi[t]) else "N"
    out.append(f"  {t:>4}  {true_beta[t]:>+10.3f}  {beta_med[t]:>+10.3f}  "
               f"{beta_lo[t]:>+10.3f}  {beta_hi[t]:>+10.3f}  {inband:>8}")
out.append(f"\nCoverage of 95% credible intervals: {in_band*100:.0f}%  "
           f"(should be near 95% if model is correct)")
out.append(f"RMSE of posterior median vs truth: {rmse_post:.4f}  "
           f"(close to zero means strong recovery)")

# True sigma vs estimated
sigma_eps_est = float(trace.posterior["sigma_eps"].median())
out.append(f"\nObservation noise: true = {sigma_eps_true:.3f}, "
           f"estimated = {sigma_eps_est:.3f}")

sigma_eta_est = float(trace.posterior["sigma_eta"].median())
sigma_eta_true = float(np.std(true_inno))   # empirical SD of true innovations
out.append(f"Innovation SD:     true (empirical) = {sigma_eta_true:.3f}, "
           f"estimated = {sigma_eta_est:.3f}")

text = "\n".join(out)
print()
print(text)
(RESULTS / "tvp_phase1_validation_summary.txt").write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Plot: posterior fan vs truth
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 3.0))
years_axis = np.arange(T)
ax.fill_between(years_axis, beta_lo, beta_hi, color="#d62728", alpha=0.20,
                linewidth=0, label="95% credible interval (posterior)")
ax.plot(years_axis, beta_med, color="#d62728", linewidth=1.5, marker="s",
        markersize=4, label="posterior median")
ax.plot(years_axis, true_beta, color="black", linewidth=1.2, marker="o",
        markersize=4, label="ground truth")
ax.axhline(0, color="gray", linewidth=0.4, linestyle="--", alpha=0.5)
ax.set_xlabel("Year index")
ax.set_ylabel(r"$\beta_t$")
ax.set_title(f"TVP synthetic-data recovery: {in_band*100:.0f}% coverage, "
             f"RMSE = {rmse_post:.3f}")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, loc="best")
plt.tight_layout(pad=0.4)
fig.savefig(PLOTS / "tvp_phase1_synthetic_recovery.pdf", dpi=300, bbox_inches="tight")
fig.savefig(PLOTS / "tvp_phase1_synthetic_recovery.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"\nWrote: {RESULTS / 'tvp_phase1_validation_summary.txt'}")
print(f"       {PLOTS / 'tvp_phase1_synthetic_recovery.pdf'}")
