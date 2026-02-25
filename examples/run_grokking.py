"""
Run the grokking experiment and save plots.

Usage
-----
    python examples/run_grokking.py [--fast]

Pass --fast for a quick smoke-test (5k steps instead of 50k; may not grok).

This will:
  1. Train a small transformer on (a + b) mod 97.
  2. Track train/val accuracy over time.
  3. Show the sudden jump in val accuracy ("grokking").
  4. Analyse the Fourier structure of token embeddings after grokking.
  5. Save PNG plots to ./outputs/.

Expected behaviour
------------------
* Train accuracy reaches ~100% early (memorisation).
* Val accuracy stays near ~1% for thousands of steps.
* Then val accuracy suddenly jumps to ~100% (generalisation = grokking).

The key ingredient is strong weight decay (default: 1.0).  Without it,
the model memorises but never generalises.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mineo.experiments.grokking import GrokkingExperiment
from mineo.visualization.plots import (
    plot_grokking_curves,
    plot_fourier_spectrum,
    plot_training_loss,
)

os.makedirs("outputs", exist_ok=True)

fast = "--fast" in sys.argv
n_steps = 5_000 if fast else 50_000

# ---- Run experiment --------------------------------------------------------
exp = GrokkingExperiment(
    p            = 97,
    op           = "+",
    train_frac   = 0.5,
    d_model      = 128,
    n_layers     = 2,
    n_heads      = 4,
    weight_decay = 1.0,
    lr           = 1e-3,
    n_steps      = n_steps,
    batch_size   = 512,
    verbose      = True,
)

history = exp.train(log_interval=max(1, n_steps // 50))

print("\nAnalysing Fourier structure of embeddings …")
fourier = exp.fourier_analysis()
print(f"  Top Fourier frequencies: {fourier['top_freqs'][:5]}")

# ---- Plots -----------------------------------------------------------------
print("\nSaving plots to ./outputs/ …")

fig1 = plot_grokking_curves(history)
fig1.savefig("outputs/grokking_curves.png", dpi=150)
plt.close(fig1)

fig2 = plot_fourier_spectrum(
    fourier["power_spectrum"],
    top_freqs=fourier["top_freqs"][:5],
)
fig2.savefig("outputs/grokking_fourier.png", dpi=150)
plt.close(fig2)

print("\nDone.")
if history["val_acc"]:
    print(f"  Final val accuracy: {history['val_acc'][-1]:.3f}")
    grokked = any(a > 0.95 for a in history["val_acc"])
    print(f"  Grokking occurred: {grokked}")
    if not grokked and fast:
        print("  (Run without --fast for full grokking demonstration)")
