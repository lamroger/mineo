"""
Generate the three educational Jupyter notebooks for mineo.

Run from the repo root:
    python scripts/generate_notebooks.py
"""

import json
import os

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def md(source: str, cell_id: str) -> dict:
    return {"cell_type": "markdown", "id": cell_id, "metadata": {}, "source": source}

def code(source: str, cell_id: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": source,
    }

METADATA = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "codemirror_mode": {"name": "ipython", "version": 3},
        "file_extension": ".py",
        "mimetype": "text/x-python",
        "name": "python",
        "version": "3.11.0",
    },
}

def notebook(cells: list) -> dict:
    return {"cells": cells, "metadata": METADATA, "nbformat": 4, "nbformat_minor": 5}

def save(nb: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"  Wrote {path}")


# ===========================================================================
# Notebook 1 — Attention Sinks
# ===========================================================================

NB1_CELLS = [

md("""\
# Attention Sinks
### *Efficient Streaming Language Models with Attention Sinks* — Xiao et al. 2023

**Paper:** [arXiv:2309.17453](https://arxiv.org/abs/2309.17453)

---

## The big picture

When you peek inside a trained language model's attention weights, you find something
surprising: **the very first token** in the sequence — even if it's just punctuation or
a BOS marker — consistently absorbs a huge fraction of the total attention mass across
*every* head and *every* layer.

The first token is called an **attention sink**.

This notebook will:
1. Train a small GPT on synthetic text
2. Measure the sink effect quantitatively
3. Show why removing the sink token breaks streaming inference
4. Verify the fix proposed by Xiao et al.
""", "m-00"),

md("""\
## Setup

Install dependencies if needed:
```
pip install torch numpy matplotlib tqdm
```
""", "m-01"),

code("""\
import sys, os
# Add repo root to path (works whether running from notebooks/ or root)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath('.')), ''))
sys.path.insert(0, os.path.abspath('..'))

import torch
import matplotlib.pyplot as plt
%matplotlib inline
plt.rcParams['figure.dpi'] = 120

from mineo.experiments.attention_sinks import AttentionSinkExperiment
from mineo.visualization.plots import (
    plot_sink_scores, plot_cache_comparison, plot_training_loss,
    plot_attention_heatmap,
)

print(f"PyTorch {torch.__version__}")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")
""", "c-01"),

md("""\
## Theory: Why does an attention sink form?

### The softmax constraint

Attention weights at each query position must sum to 1 (softmax normalisation).
When the model wants to \"not attend to anything in particular\", it still needs
somewhere to put the surplus weight.

### Why position 0?

The causal mask means every query can only attend to past tokens.  Position 0 is
the *only* token reachable from **every** query position.  So it becomes the model's
universal dump site — a token that's always available as an overflow receptacle.

Over training, this pattern gets reinforced: since the model uses position 0 as a
sink, it stops encoding useful information there, which makes it an even better sink.

### The streaming cache problem

![Streaming cache diagram](https://i.imgur.com/placeholder.png)

During streaming inference, a **sliding-window KV cache** keeps only the last $W$
tokens to bound memory.  Once position 0 slides out of the window:

- The model's attention can no longer access the sink
- The softmax has no natural dump site
- Attention weights become erratic → perplexity spikes

**Fix (Xiao et al.):** Always keep the first $k$ tokens (\"sink tokens\") in the cache
alongside the sliding window.
""", "m-02"),

md("""\
## Step 1: Train a small GPT
""", "m-03"),

code("""\
exp = AttentionSinkExperiment(
    d_model    = 128,   # embedding dimension
    n_layers   = 4,     # transformer depth
    n_heads    = 4,     # attention heads
    seq_len    = 128,   # context length
    n_steps    = 2_000, # training steps (~2 min on CPU)
    batch_size = 32,
    device     = device,
    verbose    = True,
)

print(f"Model parameters: {exp.model.n_params:,}")
""", "c-02"),

code("""\
losses = exp.train(log_interval=400)
fig = plot_training_loss(losses, title='Attention Sink Model — Training Loss')
plt.show()
""", "c-03"),

md("""\
## Step 2: Measure the attention sink

For each sample, we compute how much **total attention mass** each token position
receives (summed over all query positions, averaged over heads and layers).

Under a uniform distribution this would be $1/T$ at every position.  We look for
position 0 to be a significant outlier.
""", "m-04"),

code("""\
sink_stats = exp.measure_sink_scores(n_samples=64)

print(f"Sink ratio (pos-0 vs. uniform): {sink_stats['sink_ratio']:.2f}×")
print()
print("Attention mass at the first 10 positions:")
for i, v in enumerate(sink_stats['mean_attn_by_pos'][:10].tolist()):
    bar = '█' * int(v * 200)
    print(f"  pos {i:3d}: {v:.4f}  {bar}")
""", "c-04"),

code("""\
fig = plot_sink_scores(
    sink_stats['mean_attn_by_pos'],
    sink_stats['per_layer'],
    title=f"Attention mass per token position  (sink ratio: {sink_stats['sink_ratio']:.1f}×)",
)
plt.show()
""", "c-05"),

md("""\
### What you should see

- Position 0 (red bar) has dramatically more attention mass than any other position.
- The effect persists **across all layers** (bottom panel), not just early or late ones.
- This is the attention sink.

Try varying `n_heads` and `n_layers` — the sink appears regardless of architecture size.
""", "m-05"),

md("""\
## Step 3: The streaming cache experiment

We now simulate three KV-cache strategies for streaming inference:

| Strategy | What it keeps |
|---|---|
| `full_context` | All tokens — memory grows with sequence length |
| `window_only` | Last $W$ tokens — **evicts the sink token** |
| `sink_plus_window` | Sink token(s) + last $W-1$ tokens — Xiao et al. fix |

We measure **perplexity** (lower = better) for each strategy.
""", "m-06"),

code("""\
cache_stats = exp.simulate_streaming_cache(cache_size=32, n_eval_tokens=256)

print("\\nPerplexity comparison (lower = better):")
for strategy, ppl in cache_stats.items():
    indicator = ' ←  BEST' if ppl == min(cache_stats.values()) else ''
    print(f"  {strategy:<22s}: {ppl:.2f}{indicator}")
""", "c-06"),

code("""\
fig = plot_cache_comparison(cache_stats, title='Streaming KV-cache: perplexity comparison')
plt.show()
""", "c-07"),

md("""\
### Result

`sink_plus_window` should match `full_context` closely, while `window_only`
(which loses the sink token) has higher perplexity.

This directly validates the Xiao et al. fix: **keeping just the sink token(s)**
is enough to recover full-context performance.

---

## Visualising a single attention matrix

Let's look at what the attention pattern actually looks like for one sequence.
""", "m-07"),

code("""\
exp.model.eval()
x, _ = exp.dataset.get_batch(1, torch.device(device))
_, _, attn_list = exp.model(x, return_attn=True)

# Show layer 0, head 0
attn_layer0 = attn_list[0][0, 0].cpu()  # (T, T)

fig, axes = plt.subplots(1, len(attn_list), figsize=(4 * len(attn_list), 4))
for i, attn in enumerate(attn_list):
    a = attn[0].mean(0).cpu().numpy()  # average over heads
    axes[i].imshow(a, cmap='Blues', aspect='auto')
    axes[i].set_title(f'Layer {i}\\n(avg over heads)')
    axes[i].set_xlabel('Key pos')
    if i == 0:
        axes[i].set_ylabel('Query pos')

plt.suptitle('Attention weights: brighter = more attention\\n'
             'Notice the bright left column — the attention sink at pos 0', y=1.02)
plt.tight_layout()
plt.show()
""", "c-08"),

md("""\
## Summary

| Observation | Explanation |
|---|---|
| Position 0 gets high attention | Softmax needs a dump site; pos 0 is always reachable |
| Effect appears in all layers | The pattern is reinforced during training |
| Removing sink → high perplexity | Model relies on the sink being present |
| Keeping 1–4 sink tokens fixes it | Cheap, constant-memory solution |

**Key paper takeaway:** When designing long-context or streaming inference systems,
always reserve a few slots for "sink tokens" in your KV cache.

### Further reading
- [StreamingLLM repo](https://github.com/mit-han-lab/streaming-llm)
- Xiao et al. 2023 — [arXiv:2309.17453](https://arxiv.org/abs/2309.17453)
- "Lost in the Middle" (Liu et al. 2023) — related work on positional bias
""", "m-08"),
]


# ===========================================================================
# Notebook 2 — Grokking
# ===========================================================================

NB2_CELLS = [

md("""\
# Grokking
### *Generalization Beyond Overfitting on Small Algorithmic Datasets* — Power et al. 2022

**Paper:** [arXiv:2201.02177](https://arxiv.org/abs/2201.02177)

---

## The phenomenon

Train a small transformer to compute $(a + b) \\bmod 97$.  Use 50% of all
$97^2 = 9{,}409$ pairs for training.

Watch what happens to validation accuracy over time:

```
Step    1,000 : train_acc = 100%,  val_acc =   1%   ← memorised!
Step   10,000 : train_acc = 100%,  val_acc =   2%   ← still memorised
Step   40,000 : train_acc = 100%,  val_acc =  99%   ← GROKKING!
```

The model first memorises the training set, then — **thousands of steps later** —
suddenly generalises. This is **grokking**.
""", "m-00"),

code("""\
import sys, os
sys.path.insert(0, os.path.abspath('..'))

import torch
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
plt.rcParams['figure.dpi'] = 120

from mineo.experiments.grokking import GrokkingExperiment, ModularDataset
from mineo.visualization.plots import plot_grokking_curves, plot_fourier_spectrum

print(f"PyTorch {torch.__version__}")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")
""", "c-01"),

md("""\
## The dataset: modular arithmetic

We train on the task $(a + b) \\bmod p$ for prime $p = 97$.

- Vocabulary: integers $\\{0, \\ldots, 96\\}$ plus `+` and `=` tokens (99 tokens total)
- Input sequence: `[a, +, b, =]`  (length 4)
- Target: predict $c = (a + b) \\bmod 97$ at the `=` position

With 50% of pairs for training, the task is **hard enough to require generalisation**
but **small enough to run on a laptop**.
""", "m-01"),

code("""\
# Explore the dataset
p = 97
ds = ModularDataset(p=p, train_frac=0.5, op='+')

print(f"Vocabulary size : {ds.vocab_size}")
print(f"Train pairs     : {len(ds.X_tr)}")
print(f"Val pairs       : {len(ds.X_val)}")
print()

# Show a few examples (tokens: 0..p-1 = numbers, p = '+', p+1 = '=')
OP_TOKEN = p
EQ_TOKEN = p + 1
print("Example input sequences → answer:")
for i in range(5):
    a, op, b, eq = ds.X_tr[i].tolist()
    c = ds.Y_tr[i].item()
    print(f"  [{a}, '+', {b}, '=']  →  {c}  (check: ({a}+{b}) mod {p} = {c})")
""", "c-02"),

md("""\
## Why does grokking happen?

### Two competing solutions

The model can solve the task in (at least) two ways:

1. **Memorisation** — store each $(a, b) \\to c$ mapping as a lookup table.
   - Works perfectly on training data
   - Requires large, unstructured weights
   - Fails on unseen pairs

2. **Generalisation** — learn the underlying algorithm (modular addition).
   - Requires discovering the mathematical structure
   - Works on any $(a, b)$ pair
   - Implemented with smaller, more regular weights

### The role of weight decay

Weight decay adds an $L_2$ penalty: $\\mathcal{L}_{total} = \\mathcal{L}_{task} + \\lambda \\|\\theta\\|^2$.

This **continuously shrinks** the model's weights. Large-norm memorisation solutions
get penalised more than compact generalisation solutions. Over thousands of steps,
weight decay slowly prunes away the memorisation solution until the generalisation
solution emerges.

**Without weight decay, grokking does not happen** (the model stays memorised forever).
""", "m-02"),

md("""\
## Training

We train with strong weight decay ($\\lambda = 1.0$), AdamW optimiser, and
log train/val accuracy every 1,000 steps.

> **Time estimate:** ~5–10 min on CPU for 50k steps.
> Use `n_steps=5_000` for a quick smoke test (model may not fully grok).
""", "m-03"),

code("""\
N_STEPS = 50_000  # reduce to 5_000 for a quick test

exp = GrokkingExperiment(
    p            = 97,
    op           = '+',
    train_frac   = 0.5,
    d_model      = 128,
    n_layers     = 2,
    n_heads      = 4,
    weight_decay = 1.0,   # critical — set to 0 to disable grokking
    lr           = 1e-3,
    n_steps      = N_STEPS,
    batch_size   = 512,
    device       = device,
    verbose      = True,
)

print(f"Model parameters: {exp.model.n_params:,}")
""", "c-03"),

code("""\
history = exp.train(log_interval=1_000)
""", "c-04"),

code("""\
fig = plot_grokking_curves(history)
plt.show()
""", "c-05"),

md("""\
### What you should see

- **Left panel (loss):** Train loss drops rapidly. Val loss stays high, then collapses.
- **Right panel (accuracy):** Train accuracy hits 100% early.  Val accuracy stays near
  $1/97 \\approx 1\\%$ (random chance), then **jumps sharply to 100%** — grokking!

The green dashed line marks the grokking transition.

> If you used fewer steps, val accuracy may not have jumped yet.
> Run with `N_STEPS = 50_000` for the full effect.
""", "m-05"),

md("""\
## Mechanistic interpretation: Fourier features

*Nanda et al. (2023) — [arXiv:2301.05217](https://arxiv.org/abs/2301.05217)*

After grokking, the model uses a beautiful mathematical trick.  It represents
integer $k$ as a **point on a circle**:

$$\\text{emb}(k) \\approx \\left[\\cos\\!\\left(\\frac{2\\pi f k}{p}\\right),\\; \\sin\\!\\left(\\frac{2\\pi f k}{p}\\right), \\ldots\\right]$$

for a small set of dominant frequencies $f$.

Modular addition then becomes **angle addition** on the unit circle:

$$\\cos\\!\\left(\\frac{2\\pi f (a+b)}{p}\\right) = \\cos\\!\\left(\\frac{2\\pi f a}{p}\\right)\\cos\\!\\left(\\frac{2\\pi f b}{p}\\right) - \\sin\\!\\left(\\frac{2\\pi f a}{p}\\right)\\sin\\!\\left(\\frac{2\\pi f b}{p}\\right)$$

This is a **compact, structured representation** — hence small weights, hence favoured
by weight decay.

We can see this in the **Fourier power spectrum** of the learned embeddings.
""", "m-06"),

code("""\
fourier = exp.fourier_analysis(n_freqs=15)
print(f"Top Fourier frequencies: {fourier['top_freqs']}")
print("(After grokking, a handful of frequencies dominate)")
""", "c-06"),

code("""\
fig = plot_fourier_spectrum(
    fourier['power_spectrum'],
    top_freqs=fourier['top_freqs'][:5],
    title='Fourier spectrum of token embeddings\\n(dominant peaks = Fourier representation of modular arithmetic)',
)
plt.show()
""", "c-07"),

md("""\
### What you should see

After grokking: a few sharp peaks in the spectrum (highlighted in red).
Before grokking: a flat/noisy spectrum — no structure.

This is direct evidence that the model has learned to represent integers as
points on a circle rather than memorised lookup tables.

---

## Ablation: weight decay matters

Run a quick comparison with `weight_decay=0`:
""", "m-07"),

code("""\
# Quick ablation: train without weight decay
exp_no_wd = GrokkingExperiment(
    p=97, n_steps=10_000, weight_decay=0.0,
    device=device, verbose=False,
)
hist_no_wd = exp_no_wd.train(log_interval=1_000)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(hist_no_wd['steps'], hist_no_wd['val_acc'],
        label='val acc (weight_decay=0)', color='crimson', linestyle='--')
if history['val_acc']:
    ax.plot(history['steps'][:len(hist_no_wd['steps'])],
            history['val_acc'][:len(hist_no_wd['steps'])],
            label='val acc (weight_decay=1.0)', color='steelblue')
ax.set_xlabel('Step')
ax.set_ylabel('Val accuracy')
ax.set_title('Grokking requires weight decay')
ax.legend()
ax.grid(alpha=0.3)
plt.show()

print(f"Final val acc (wd=0):   {hist_no_wd['val_acc'][-1]:.3f}")
if history['val_acc']:
    print(f"Final val acc (wd=1.0): {history['val_acc'][-1]:.3f}")
""", "c-08"),

md("""\
## Summary

| Concept | Key point |
|---|---|
| Grokking | Generalisation can happen thousands of steps *after* memorisation |
| Mechanism | Weight decay shrinks memorisation solutions; generalisation solutions survive |
| Representation | The model learns Fourier/circular features for modular arithmetic |
| Hyper-parameters | `weight_decay`, `train_frac`, and `n_steps` all affect the grokking delay |

### Further reading
- Power et al. 2022 — [arXiv:2201.02177](https://arxiv.org/abs/2201.02177) (original grokking paper)
- Nanda et al. 2023 — [arXiv:2301.05217](https://arxiv.org/abs/2301.05217) (mechanistic interpretation)
- Liu et al. 2022 — "Towards Understanding Grokking" — theoretical analysis
""", "m-08"),
]


# ===========================================================================
# Notebook 3 — In-Context Learning as Gradient Descent
# ===========================================================================

NB3_CELLS = [

md("""\
# In-Context Learning as Gradient Descent
### The "13-parameter" minimal transformer

**Papers:**
- Oswald et al. (NeurIPS 2023) — [arXiv:2212.07677](https://arxiv.org/abs/2212.07677)
  *Transformers Learn In-Context Learning by Gradient Descent*
- Akyürek et al. (ICLR 2023) — [arXiv:2211.15661](https://arxiv.org/abs/2211.15661)
  *What Learning Algorithm is In-Context Learning? Investigations with Linear Models*

---

## The big picture

**In-context learning (ICL):** A language model sees a few $(x_i, y_i)$ examples in
its prompt, then predicts $y$ for a new $x$ — without any weight updates.

**The insight from Oswald et al.:**
A *single-layer linear attention* transformer can implement this exactly — and the
operation it performs is equivalent to **one step of gradient descent** on the training
examples.

This means:
- Transformers don't need explicit optimisation algorithms
- The attention mechanism *is* an optimiser
- Even a **13-parameter** model can do in-context learning
""", "m-00"),

code("""\
import sys, os
sys.path.insert(0, os.path.abspath('..'))

import torch
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
plt.rcParams['figure.dpi'] = 120

from mineo.experiments.icl_gd import (
    ICLGradientDescentExperiment,
    LinearICLModel,
    LinearRegressionTaskGenerator,
    one_step_gd_predict,
    ols_predict,
)
from mineo.visualization.plots import plot_icl_comparison, plot_training_loss

print(f"PyTorch {torch.__version__}")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")
""", "c-01"),

md("""\
## Mathematical derivation

### The task

Each "task" is a **linear regression problem** with an unknown ground-truth weight $w^*$:

$$y_i = x_i \\cdot w^* + \\varepsilon_i, \\quad \\varepsilon_i \\sim \\mathcal{N}(0, \\sigma^2)$$

Given $n$ context pairs $\\{(x_1, y_1), \\ldots, (x_n, y_n)\\}$ and a query $x_*$,
we want to predict $\\hat{y}_* \\approx x_* \\cdot w^*$.

### One step of gradient descent

Starting from $w_0 = \\mathbf{0}$, one GD step minimising MSE gives:

$$\\nabla_{w} \\mathcal{L}\\big|_{w=0} = -\\frac{1}{n}\\sum_i x_i y_i$$

$$w_1 = w_0 - \\eta \\cdot \\nabla_{w} \\mathcal{L}\\big|_{w=0} = \\frac{\\eta}{n}\\sum_i x_i y_i$$

$$\\hat{y}_* = x_* \\cdot w_1 = \\frac{\\eta}{n}\\sum_i y_i (x_* \\cdot x_i)$$

### Linear attention

Represent each $(x_i, y_i)$ pair as a token $z_i = [x_i;\\ y_i] \\in \\mathbb{R}^2$ (for 1D inputs).
Query token: $z_* = [x_*;\\ 0]$ (unknown label, initialised to 0).

Linear (softmax-free) attention computes:

$$\\text{Attn}(Z) = \\frac{1}{n} Z \\cdot (W_K W_Q^\\top) \\cdot Z^\\top \\cdot W_V \\cdot Z$$

With $W_K = W_Q = I$ and $W_V = -e_1 e_2^\\top$ (reads $y$, writes to $x$-slot):

$$\\text{output at query} \\propto \\frac{1}{n}\\sum_i y_i (x_* \\cdot x_i)$$

**This is exactly the one-step GD prediction.** ✓
""", "m-01"),

md("""\
## The 13-parameter construction

For 1-dimensional inputs ($x, y \\in \\mathbb{R}$), the token space is 2D ($z = [x, y]$).
The complete model has these weight matrices:

```
W_K  (2×2):  identity — 4 matrix entries (structure fixed: 0 free params)
W_Q  (2×2):  identity — 4 matrix entries (structure fixed: 0 free params)
W_V  (2×2):  [[ 0,  0],   — reads y-slot, writes to x-slot
              [-1,  0]]     4 entries, but rank-1 structure → 2 free
W_O  (2×2):  [[ 0, -1],   — reads x-slot, writes to y-slot
              [ 0,  0]]     4 entries, but rank-1 structure → 2 free
η    (1×1):  step size     1 free parameter
─────────────────────────────────────────────────────────────────
Total free scalars:  0 + 0 + 2 + 2 + 1  =  5  (minimal theoretical)
Total matrix entries: 4+4+4+4+1         = 17  (full parameterisation)
"13 parameters" counts the raw entries of the *minimal architecture*:
    W_V (4) + W_O (4) + η (1) + 2 rank vectors × 2 = 13
```

The exact count of 13 depends on parameterisation, but the key point is:
**a handful of parameters is sufficient to implement in-context gradient descent**.
""", "m-02"),

code("""\
# Inspect the minimal model
model = LinearICLModel(input_dim=1)
print("LinearICLModel (input_dim=1)")
print(f"  Total parameters: {model.n_params}")
print()
for name, p in model.named_parameters():
    print(f"  {name:<20s}: shape {tuple(p.shape)}, {p.numel()} scalars")
""", "c-02"),

md("""\
## Demo 1: The theoretical construction (no training needed)

We can **hardcode** the Oswald et al. weight matrices directly and verify that
the model already performs in-context linear regression — without any meta-training.
""", "m-03"),

code("""\
# Set the theoretical weights (provably implements 1-step GD)
theoretical = LinearICLModel(input_dim=1)
theoretical.set_theoretical_weights()

print("Theoretical weight matrices:")
for name, p in theoretical.named_parameters():
    print(f"\\n  {name}:")
    print("  ", p.data.numpy())
""", "c-03"),

code("""\
# Generate some random linear regression tasks and compare predictions
gen = LinearRegressionTaskGenerator(input_dim=1, noise_std=0.05, device=torch.device('cpu'))
xs, ys, x_q, y_q, w_star = gen.sample(batch_size=500, n_shots=8)

with torch.no_grad():
    y_theory = theoretical(xs, ys, x_q)
    y_gd     = one_step_gd_predict(xs, ys, x_q, lr=1.0)
    y_ols    = ols_predict(xs, ys, x_q)

mse_theory = ((y_theory - y_q) ** 2).mean().item()
mse_gd     = ((y_gd     - y_q) ** 2).mean().item()
mse_ols    = ((y_ols    - y_q) ** 2).mean().item()

print("MSE on 500 random tasks (n_shots=8):")
print(f"  Theoretical construction : {mse_theory:.6f}")
print(f"  1-step gradient descent  : {mse_gd:.6f}")
print(f"  OLS (optimal)            : {mse_ols:.6f}")
print()
print("Theoretical ≈ GD:  ", abs(mse_theory - mse_gd) < 0.01)
""", "c-04"),

code("""\
# Scatter plot: theoretical predictions vs ground truth
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
y_q_np = y_q.numpy().ravel()

for ax, (y_pred, label, color) in zip(axes, [
    (y_theory, 'Theoretical (13-param)', 'darkorange'),
    (y_gd,     '1-step GD',              'seagreen'),
    (y_ols,    'OLS (optimal)',           'steelblue'),
]):
    y_pred_np = y_pred.detach().numpy().ravel()
    ax.scatter(y_q_np, y_pred_np, alpha=0.3, s=10, color=color)
    lim = max(abs(y_q_np).max(), abs(y_pred_np).max()) * 1.1
    ax.plot([-lim, lim], [-lim, lim], 'k--', linewidth=1)
    ax.set_xlabel('True y*')
    ax.set_ylabel('Predicted ŷ*')
    ax.set_title(f'{label}\\nMSE={((y_pred_np-y_q_np)**2).mean():.4f}')
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)

plt.suptitle('Prediction scatter plots (perfect = points on diagonal)', y=1.02)
plt.tight_layout()
plt.show()
""", "c-05"),

md("""\
## Demo 2: Meta-training

Now we **learn** the weights by meta-training: sample a fresh linear regression
task every step, run the model, compute MSE loss, backprop.

After training, the learned weights should approximately match the theoretical
construction.
""", "m-04"),

code("""\
exp = ICLGradientDescentExperiment(
    input_dim  = 1,      # 1D inputs → the 13-parameter story
    n_shots    = 16,     # context examples per task
    n_steps    = 5_000,  # meta-training steps
    batch_size = 64,
    lr         = 1e-3,
    noise_std  = 0.1,
    device     = device,
    verbose    = True,
)
""", "c-06"),

code("""\
history = exp.train(log_interval=500)

fig = plot_training_loss(
    history['loss'],
    steps   = history['steps'],
    title   = 'ICL meta-training loss (MSE on random linear regression tasks)',
)
plt.show()
""", "c-07"),

code("""\
# Evaluate all methods
metrics = exp.evaluate(n_tasks=1024)
print("\\nEvaluation (MSE, lower = better):")
for k, v in metrics.items():
    print(f"  {k:<20s}: {v:.6f}")
""", "c-08"),

code("""\
fig = plot_icl_comparison(metrics, title='In-Context Learning: MSE comparison')
plt.show()
""", "c-09"),

md("""\
### What you should see

- **Trained** ≈ **1-step GD** ≈ **Theoretical** — the meta-learned weights approximately
  implement gradient descent, matching the theoretical construction without being told to.
- **OLS** is the optimal predictor for linear regression with many shots; the GD
  approximation gets worse as $n \\to \\infty$ (GD is only exact at the optimum, not
  at one step from zero).

---

## Demo 3: How does ICL improve with more shots?

With more context examples, the GD prediction improves.  Let's see how MSE changes
as a function of `n_shots`.
""", "m-05"),

code("""\
gen = LinearRegressionTaskGenerator(input_dim=1, noise_std=0.05)
shots_range = [1, 2, 4, 8, 16, 32, 64]
mse_theory_list, mse_gd_list, mse_ols_list = [], [], []

theoretical = LinearICLModel(input_dim=1)
theoretical.set_theoretical_weights()

with torch.no_grad():
    for n in shots_range:
        xs, ys, x_q, y_q, _ = gen.sample(batch_size=1000, n_shots=n)
        mse_theory_list.append(((theoretical(xs, ys, x_q) - y_q)**2).mean().item())
        mse_gd_list.append(((one_step_gd_predict(xs, ys, x_q) - y_q)**2).mean().item())
        mse_ols_list.append(((ols_predict(xs, ys, x_q) - y_q)**2).mean().item())

fig, ax = plt.subplots(figsize=(8, 4))
ax.loglog(shots_range, mse_theory_list, 'o-', label='Theoretical (13-param)', color='darkorange')
ax.loglog(shots_range, mse_gd_list,     's--', label='1-step GD',             color='seagreen')
ax.loglog(shots_range, mse_ols_list,    '^:', label='OLS (optimal)',          color='steelblue')
ax.set_xlabel('Number of context shots (n)')
ax.set_ylabel('MSE (log scale)')
ax.set_title('ICL accuracy improves with more shots')
ax.legend()
ax.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.show()

print("More shots = lower MSE for all methods.")
print("OLS converges to 0 fastest; 1-step GD slows after ~n=d (here d=1).")
""", "c-10"),

md("""\
## Summary

| Concept | Key point |
|---|---|
| Linear attention | Softmax-free; exactly implements weighted sum = GD step |
| 13 parameters | Sufficient to implement in-context GD for 1D linear regression |
| Meta-learning | Model *discovers* GD internally, without being told the algorithm |
| Shots vs accuracy | More context → better prediction, approaching OLS |

### What this tells us about LLMs

Large language models may implement **multiple steps of gradient descent** via their
attention layers — essentially running a learned optimisation algorithm at inference
time. This is why GPT-style models can learn from demonstrations in the prompt without
weight updates.

### Further reading
- Oswald et al. 2023 — [arXiv:2212.07677](https://arxiv.org/abs/2212.07677)
- Akyürek et al. 2023 — [arXiv:2211.15661](https://arxiv.org/abs/2211.15661)
- Dai et al. 2023 — "Why Can GPT Learn In-Context?" — dual form of attention
- Ahn et al. 2023 — "Transformers Learn to Implement Preconditioned GD for In-Context Learning"
""", "m-06"),
]


# ===========================================================================
# Write notebooks
# ===========================================================================

if __name__ == "__main__":
    base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "notebooks")
    save(notebook(NB1_CELLS), os.path.join(base, "01_attention_sinks.ipynb"))
    save(notebook(NB2_CELLS), os.path.join(base, "02_grokking.ipynb"))
    save(notebook(NB3_CELLS), os.path.join(base, "03_icl_gradient_descent.ipynb"))
    print("Done.")
