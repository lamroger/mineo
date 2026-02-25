"""
Run the attention-sink experiment and save plots.

Usage
-----
    python examples/run_attention_sinks.py

This will:
  1. Train a small GPT on synthetic text (~2 min on CPU).
  2. Measure attention weights and report the "sink ratio".
  3. Compare perplexity under three streaming KV-cache strategies.
  4. Save PNG plots to ./outputs/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")   # non-interactive backend for script use
import matplotlib.pyplot as plt

from mineo.experiments.attention_sinks import AttentionSinkExperiment
from mineo.visualization.plots import (
    plot_sink_scores,
    plot_cache_comparison,
    plot_training_loss,
)

os.makedirs("outputs", exist_ok=True)

# ---- Run experiment --------------------------------------------------------
exp = AttentionSinkExperiment(
    d_model    = 128,
    n_layers   = 4,
    n_heads    = 4,
    seq_len    = 128,
    n_steps    = 2_000,
    batch_size = 32,
    verbose    = True,
)

print("Training …")
losses = exp.train()

print("\nMeasuring sink scores …")
sink_stats = exp.measure_sink_scores()

print("\nSimulating streaming cache …")
cache_stats = exp.simulate_streaming_cache()

# ---- Plots -----------------------------------------------------------------
print("\nSaving plots to ./outputs/ …")

fig1 = plot_training_loss(losses, title="Attention Sink Experiment — Training Loss")
fig1.savefig("outputs/attention_sink_train_loss.png", dpi=150)
plt.close(fig1)

fig2 = plot_sink_scores(
    sink_stats["mean_attn_by_pos"],
    sink_stats["per_layer"],
    title=f"Attention sink scores  (sink ratio: {sink_stats['sink_ratio']:.1f}×)",
)
fig2.savefig("outputs/attention_sink_scores.png", dpi=150)
plt.close(fig2)

fig3 = plot_cache_comparison(
    cache_stats,
    title="Streaming KV-cache: perplexity comparison",
)
fig3.savefig("outputs/attention_sink_cache_comparison.png", dpi=150)
plt.close(fig3)

print("\nDone.  Sink ratio:", f"{sink_stats['sink_ratio']:.2f}×")
print("Cache perplexities:")
for k, v in cache_stats.items():
    print(f"  {k:<22s}: {v:.2f}")
