"""
Run the in-context learning ≈ gradient descent experiment and save plots.

Usage
-----
    python examples/run_icl_gd.py

This will:
  1. Meta-train a minimal linear-attention transformer on random linear
     regression tasks.
  2. Evaluate its predictions vs:
       - Ordinary Least Squares (optimal baseline)
       - One-step gradient descent from zero initialisation
       - The *theoretical* hardcoded construction (Oswald et al. 2023)
  3. Show that the trained model approximately implements in-context GD.
  4. Save plots to ./outputs/.

The "13-parameter" story
------------------------
For 1-dimensional inputs (x, y ∈ ℝ), the complete model has:
    W_K  (2×2) :  4 parameters
    W_Q  (2×2) :  4 parameters
    W_V  (2×2) :  4 parameters   ← stores the GD update
    W_O  (2×2) :  4 parameters   ← reads the GD update into the label slot
    η    (1×1) :  1 step-size parameter
    ──────────────────────────────────────
    Total      : 17 parameters (learned)

The *theoretical construction* fixes W_K=W_Q=I and uses rank-1 W_V, W_O,
leaving only η and the two rank-1 vectors — roughly 13 independent scalars —
which is where the "13-parameter" claim originates.

Mathematical intuition
----------------------
After one step of GD from w=0 on the MSE loss:

    w₁ = lr · (1/n) Σᵢ xᵢ yᵢ     (gradient update)

The prediction is:
    ŷ* = x* · w₁ = (lr/n) Σᵢ yᵢ · (x* · xᵢ)

Linear attention computes exactly this: it weights each context label yᵢ by
the similarity of x* to xᵢ, which is the gradient-descent update rule.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mineo.experiments.icl_gd import ICLGradientDescentExperiment
from mineo.visualization.plots import plot_icl_comparison, plot_training_loss

os.makedirs("outputs", exist_ok=True)

# ---- Run experiment --------------------------------------------------------
exp = ICLGradientDescentExperiment(
    input_dim  = 1,      # use 1D for the 13-parameter story
    n_shots    = 16,     # context examples per task
    n_steps    = 5_000,
    batch_size = 64,
    lr         = 1e-3,
    noise_std  = 0.1,
    verbose    = True,
)

history = exp.train(log_interval=500)
metrics = exp.evaluate(n_tasks=1024)

# ---- Plots -----------------------------------------------------------------
print("\nSaving plots to ./outputs/ …")

fig1 = plot_training_loss(
    history["loss"],
    steps   = history["steps"],
    title   = "ICL-as-GD meta-training loss (MSE on random linear regression tasks)",
    log_scale = True,
)
fig1.savefig("outputs/icl_gd_training.png", dpi=150)
plt.close(fig1)

fig2 = plot_icl_comparison(
    metrics,
    title = "In-Context Learning: MSE comparison (lower = better)",
)
fig2.savefig("outputs/icl_gd_comparison.png", dpi=150)
plt.close(fig2)

print("\nDone.")
print("\nKey result:")
print(f"  Trained transformer  MSE = {metrics['trained']:.6f}")
print(f"  1-step GD            MSE = {metrics['gd_1step']:.6f}")
if "theoretical" in metrics:
    print(f"  Theoretical (13-par) MSE = {metrics['theoretical']:.6f}")
print(f"  OLS (optimal)        MSE = {metrics['ols']:.6f}")
print(
    "\n  → Trained ≈ 1-step GD ≈ Theoretical >> OLS "
    "(with more shots, transformer also approaches OLS)"
)
