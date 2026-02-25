"""
Visualisation utilities for mineo experiments.

Each function accepts pre-computed data (tensors / dicts) and returns a
matplotlib Figure, so they work in both scripts and Jupyter notebooks.

Quick start
-----------
    import matplotlib.pyplot as plt
    from mineo.visualization.plots import plot_grokking_curves

    fig = plot_grokking_curves(history)
    plt.show()         # interactive
    fig.savefig("grokking.png", dpi=150)   # save
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_numpy(x: Union[torch.Tensor, np.ndarray, list]) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().float().numpy()
    return np.asarray(x, dtype=float)


def _make_fig(w: float = 8.0, h: float = 5.0, **kwargs) -> plt.Figure:
    return plt.figure(figsize=(w, h), **kwargs)


# ---------------------------------------------------------------------------
# Attention heatmap
# ---------------------------------------------------------------------------

def plot_attention_heatmap(
    attn_weights: Union[torch.Tensor, np.ndarray],
    tokens: Optional[Sequence[str]] = None,
    title: str = "Attention weights",
    cmap: str = "Blues",
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Plot a single attention weight matrix as a heatmap.

    Parameters
    ----------
    attn_weights : (T, T) or (1, T, T) — attention from each query to each key
    tokens       : optional list of T token strings for axis labels
    title        : plot title
    cmap         : matplotlib colormap
    ax           : optional existing axes (if None, a new figure is created)

    Returns
    -------
    matplotlib Figure
    """
    A = _to_numpy(attn_weights)
    if A.ndim == 3:
        A = A[0]  # take first head if extra dim

    fig, ax_ = (ax.get_figure(), ax) if ax else plt.subplots(figsize=(7, 6))
    im = ax_.imshow(A, cmap=cmap, vmin=0, vmax=A.max(), aspect="auto")
    plt.colorbar(im, ax=ax_, fraction=0.046, pad=0.04)

    T = A.shape[0]
    if tokens is not None and len(tokens) == T:
        ax_.set_xticks(range(T)); ax_.set_xticklabels(tokens, rotation=45, ha="right", fontsize=8)
        ax_.set_yticks(range(T)); ax_.set_yticklabels(tokens, fontsize=8)
    ax_.set_xlabel("Key position (token attended to)")
    ax_.set_ylabel("Query position (attending token)")
    ax_.set_title(title)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Attention sinks
# ---------------------------------------------------------------------------

def plot_sink_scores(
    mean_attn_by_pos: Union[torch.Tensor, np.ndarray],
    per_layer:        Optional[Union[torch.Tensor, np.ndarray]] = None,
    title: str = "Attention mass per token position",
) -> plt.Figure:
    """
    Visualise attention sink scores.

    Top panel  : mean attention mass per position (averaged over all layers/heads).
                 The spike at position 0 is the attention sink.

    Bottom panel (optional) : same broken down by layer.

    Parameters
    ----------
    mean_attn_by_pos : (T,) — mean attention column-sum per position
    per_layer        : (n_layers, T) — optional layer breakdown
    """
    mean = _to_numpy(mean_attn_by_pos)
    T    = len(mean)

    n_panels = 2 if per_layer is not None else 1
    fig, axes = plt.subplots(n_panels, 1, figsize=(9, 4 * n_panels))
    if n_panels == 1:
        axes = [axes]

    # ---- top panel: mean across layers ----
    ax = axes[0]
    colors = ["crimson" if i == 0 else "steelblue" for i in range(T)]
    ax.bar(range(T), mean, color=colors, edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Token position")
    ax.set_ylabel("Mean attention mass")
    ax.set_title(title)

    # Annotate position 0
    ax.annotate(
        f"Sink!\n(pos 0: {mean[0]:.3f})",
        xy=(0, mean[0]), xytext=(T * 0.1, mean[0] * 0.85),
        arrowprops=dict(arrowstyle="->", color="crimson"),
        color="crimson", fontsize=9,
    )
    # Baseline: expected under uniform distribution
    uniform = mean.mean()
    ax.axhline(uniform, linestyle="--", color="gray", linewidth=1, label=f"Uniform baseline ({uniform:.3f})")
    ax.legend(fontsize=8)

    # ---- bottom panel: per-layer ----
    if per_layer is not None:
        pl  = _to_numpy(per_layer)    # (n_layers, T)
        ax2 = axes[1]
        for layer_idx, row in enumerate(pl):
            ax2.plot(range(T), row, label=f"Layer {layer_idx}", alpha=0.8)
        ax2.set_xlabel("Token position")
        ax2.set_ylabel("Attention mass")
        ax2.set_title("Attention mass per position, broken down by layer")
        ax2.legend(fontsize=8, loc="upper right")
        ax2.axvline(0, color="crimson", linestyle=":", linewidth=1.5, label="Sink (pos 0)")

    fig.tight_layout()
    return fig


def plot_cache_comparison(
    cache_results: Dict[str, float],
    title: str = "Streaming KV-cache strategies",
) -> plt.Figure:
    """
    Bar chart comparing perplexity of three streaming KV-cache strategies.

    Lower perplexity = better.  Expect:
        full_context ≈ sink_plus_window  <<  window_only
    """
    labels = list(cache_results.keys())
    values = [cache_results[k] for k in labels]
    colors = ["steelblue", "crimson", "seagreen"][:len(labels)]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, values, color=colors, edgecolor="white")
    ax.set_ylabel("Perplexity (lower = better)")
    ax.set_title(title)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.01,
            f"{val:.1f}",
            ha="center", va="bottom", fontsize=9,
        )
    ax.set_ylim(0, max(values) * 1.2)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Grokking
# ---------------------------------------------------------------------------

def plot_grokking_curves(
    history: Dict[str, List],
    title:   str = "Grokking: generalisation long after memorisation",
) -> plt.Figure:
    """
    Plot the canonical grokking curves: train/val loss and accuracy vs step.

    The "grokking" signature is:
        * Train accuracy reaches 1.0 early (memorisation).
        * Val accuracy stays near random for thousands more steps.
        * Then val accuracy suddenly jumps to ~1.0 (generalisation).

    Parameters
    ----------
    history : dict with keys 'steps', 'train_loss', 'val_loss',
                               'train_acc', 'val_acc'
    """
    steps      = history["steps"]
    train_loss = history["train_loss"]
    val_loss   = history["val_loss"]
    train_acc  = history.get("train_acc", [])
    val_acc    = history.get("val_acc",   [])

    n_panels = 2 if train_acc else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    # ---- loss panel ----
    ax = axes[0]
    ax.semilogy(steps, train_loss, label="Train loss", color="steelblue",  linewidth=1.5)
    ax.semilogy(steps, val_loss,   label="Val loss",   color="orangered",  linewidth=1.5)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss (log scale)")
    ax.set_title(f"{title}\n— Loss")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    # ---- accuracy panel ----
    if train_acc:
        ax2 = axes[1]
        ax2.plot(steps, train_acc, label="Train acc", color="steelblue", linewidth=1.5)
        ax2.plot(steps, val_acc,   label="Val acc",   color="orangered", linewidth=1.5)

        # Annotate the grokking transition
        val_arr = np.array(val_acc)
        grok_idx = np.where(val_arr > 0.95)[0]
        if len(grok_idx) > 0:
            g_step = steps[grok_idx[0]]
            ax2.axvline(g_step, color="green", linestyle="--", linewidth=1.2)
            ax2.annotate(
                f"Grokking!\n(step {g_step:,})",
                xy=(g_step, 0.5),
                xytext=(g_step * 0.7, 0.4),
                arrowprops=dict(arrowstyle="->", color="green"),
                color="green", fontsize=9,
            )

        ax2.set_xlabel("Training step")
        ax2.set_ylabel("Accuracy")
        ax2.set_ylim(-0.05, 1.1)
        ax2.set_title(f"{title}\n— Accuracy")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Fourier spectrum (grokking mechanistic analysis)
# ---------------------------------------------------------------------------

def plot_fourier_spectrum(
    power_spectrum: Union[torch.Tensor, np.ndarray],
    top_freqs:      Optional[List[int]] = None,
    title:          str = "Fourier spectrum of token embeddings",
) -> plt.Figure:
    """
    Visualise the Fourier power spectrum of the embedding matrix.

    After grokking, a few frequencies dominate — corresponding to the
    circular / Fourier representation of modular arithmetic.

    Parameters
    ----------
    power_spectrum : (d_model, n_freqs) — Fourier power per dimension per frequency
    top_freqs      : optional list of dominant frequency indices to highlight
    """
    P  = _to_numpy(power_spectrum)     # (d_model, n_freqs)
    # Sum over embedding dimensions to get total power per frequency
    total = P.sum(axis=0)              # (n_freqs,)
    freqs = np.arange(len(total))

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(freqs, total, color="steelblue", alpha=0.7)

    if top_freqs:
        for f in top_freqs:
            if f < len(total):
                ax.bar(f, total[f], color="crimson", label=f"f={f}")

    ax.set_xlabel("Frequency")
    ax.set_ylabel("Total Fourier power (summed over d_model)")
    ax.set_title(title)
    ax.legend(fontsize=8, title="Top frequencies")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# ICL / gradient descent comparison
# ---------------------------------------------------------------------------

def plot_icl_comparison(
    metrics:   Dict[str, float],
    title:     str = "In-context learning: MSE comparison",
) -> plt.Figure:
    """
    Bar chart comparing prediction MSE of the transformer vs baselines.

    Parameters
    ----------
    metrics : dict with keys 'trained', 'theoretical', 'gd_1step', 'ols'
              (as returned by ICLGradientDescentExperiment.evaluate())
    """
    label_map = {
        "trained":      "Trained\ntransformer",
        "theoretical":  "Theoretical\nconstruction",
        "gd_1step":     "1-step\ngradient descent",
        "ols":          "OLS\n(optimal)",
    }
    color_map = {
        "trained":      "steelblue",
        "theoretical":  "darkorange",
        "gd_1step":     "seagreen",
        "ols":          "mediumpurple",
    }

    keys   = [k for k in label_map if k in metrics]
    labels = [label_map[k] for k in keys]
    values = [metrics[k]   for k in keys]
    colors = [color_map[k] for k in keys]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=colors, edgecolor="white")
    ax.set_ylabel("MSE (lower = better)")
    ax.set_title(title)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.02,
            f"{val:.5f}",
            ha="center", va="bottom", fontsize=8,
        )

    ax.set_ylim(0, max(values) * 1.3)
    fig.tight_layout()
    return fig


def plot_training_loss(
    losses:    List[float],
    steps:     Optional[List[int]] = None,
    title:     str = "Training loss",
    log_scale: bool = True,
) -> plt.Figure:
    """
    Simple training-loss curve.

    Parameters
    ----------
    losses : list of scalar loss values
    steps  : optional x-axis values (defaults to 1, 2, …)
    """
    xs = steps if steps else list(range(1, len(losses) + 1))
    fig, ax = plt.subplots(figsize=(8, 4))
    if log_scale:
        ax.semilogy(xs, losses, color="steelblue", linewidth=1.2)
    else:
        ax.plot(xs, losses, color="steelblue", linewidth=1.2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    return fig
