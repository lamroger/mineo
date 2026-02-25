# mineo

**Minimal educational LLM research toolkit.**

Reproduce key results from the scientific literature with small, readable
implementations — designed for learning as much as for cutting-edge research.

---

## Experiments

| Experiment | Paper | What you'll see |
|---|---|---|
| **Attention Sinks** | Xiao et al. 2023 | Position 0 hoards attention mass; removing it breaks streaming inference |
| **Grokking** | Power et al. 2022 | Sudden generalisation on modular arithmetic, thousands of steps after memorisation |
| **ICL as Gradient Descent** | Oswald et al. 2023 | A 13-parameter linear transformer implements one step of gradient descent |

---

## Quick start

### Jupyter notebooks (recommended)

```bash
pip install -r requirements.txt
pip install jupyter

jupyter notebook notebooks/
```

| Notebook | Experiment | Contents |
|---|---|---|
| `01_attention_sinks.ipynb` | Attention Sinks | Theory → train → measure → streaming cache demo |
| `02_grokking.ipynb` | Grokking | Dataset exploration → train → grokking curves → Fourier analysis |
| `03_icl_gradient_descent.ipynb` | ICL as GD | Math derivation → theoretical 13-param model → meta-train → comparison |

### Python scripts (fast, no Jupyter needed)

```bash
pip install -r requirements.txt

# Attention sinks (~2 min on CPU)
python examples/run_attention_sinks.py

# Grokking — fast smoke-test
python examples/run_grokking.py --fast

# Grokking — full demo (may take 15–30 min on CPU)
python examples/run_grokking.py

# In-context learning as gradient descent
python examples/run_icl_gd.py
```

Plots are saved to `./outputs/`.

### Regenerate notebooks

```bash
python scripts/generate_notebooks.py
```

---

## Experiment deep-dives

### 1. Attention Sinks

**Paper:** "Efficient Streaming Language Models with Attention Sinks"
— Xiao, Tian, Chen, Han, Lewis (2023)
[arXiv:2309.17453](https://arxiv.org/abs/2309.17453)

**The finding:**

When you visualise attention weight matrices in a trained language model, you
see something unexpected: the very first token in the sequence — even if it's
just a period or a whitespace character — accumulates a disproportionately
large fraction of the total attention mass across nearly every head and layer.

```
Attention weights for a typical sequence:
  Position:    0     1     2     3  …  127
  Attn mass: 0.42  0.01  0.02  0.01 … 0.01   ← position 0 is the "sink"
```

**Why does this happen?**

Softmax must produce weights that sum to 1. When the model doesn't need to
attend to anything specific, it has nowhere to "dump" the surplus weight —
so it learns to use the first token as a permanent dump site. The first token
is the only one that every other position can attend to (due to the causal
mask), making it the ideal sink.

**Why does it matter?**

Streaming inference often uses a *sliding-window* KV cache to stay within
memory bounds. If you naively slide the window forward, you eventually evict
the sink token — and perplexity spikes sharply. The fix: always keep a small
number of initial "sink" tokens in the cache, even as the window slides.

**What the code demonstrates:**

```python
from mineo.experiments.attention_sinks import AttentionSinkExperiment

exp = AttentionSinkExperiment()
exp.train()
sink_stats  = exp.measure_sink_scores()   # sink_ratio ≫ 1.0
cache_stats = exp.simulate_streaming_cache()
# full_context ≈ sink_plus_window  <  window_only  (lower perplexity = better)
```

---

### 2. Grokking

**Paper:** "Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets"
— Power, Burda, Edwards, Babuschkin, Misra (2022)
[arXiv:2201.02177](https://arxiv.org/abs/2201.02177)

**The phenomenon:**

Train a small transformer to compute `(a + b) mod 97`. With ~50% of all
9,409 pairs as training data:

```
Step    1,000 : train_acc = 100%,  val_acc =   1%   (memorised!)
Step   10,000 : train_acc = 100%,  val_acc =   2%   (still memorised)
Step   40,000 : train_acc = 100%,  val_acc =  99%   (suddenly generalised!)
                                              ↑
                                        GROKKING
```

**Why does delayed generalisation happen?**

With strong weight decay, the model is under constant pressure to reduce
the norm of its weights. Memorisation solutions tend to require large weights.
Over time, weight decay slowly prunes away the high-norm memorisation
"shortcuts", forcing the model to find a compact, structured representation
that generalises — but this takes a very long time.

**The mechanistic explanation (Nanda et al. 2023):**

After grokking, the model represents integer `k` as a point on a circle:
```
embedding(k) ≈ [cos(2πfk/p), sin(2πfk/p), cos(2πf'k/p), sin(2πf'k/p), …]
```
for a small set of dominant frequencies `f, f', …`. Addition mod p then
becomes angle addition on the circle — a beautifully regular structure
that the MLP can implement with very small weights.

You can see this in the Fourier spectrum of the learned embeddings:

```python
from mineo.experiments.grokking import GrokkingExperiment

exp = GrokkingExperiment(p=97, weight_decay=1.0)
history = exp.train(n_steps=50_000)   # watch val_acc jump!
fourier = exp.fourier_analysis()
print(fourier["top_freqs"])   # a handful of dominant frequencies
```

**Key hyper-parameters for grokking:**

| Parameter | Value | Effect |
|---|---|---|
| `weight_decay` | 1.0 | **Critical.** Causes grokking. Set to 0 → no generalisation |
| `train_frac` | 0.5 | Lower → harder task, longer grokking delay |
| `p` | 97 (prime) | Primes avoid divisor structure; 113, 67 also work |
| `n_steps` | 50,000+ | Grokking is slow; don't stop at memorisation |

---

### 3. In-Context Learning as Gradient Descent

**Papers:**
- "Transformers Learn In-Context Learning by Gradient Descent"
  — von Oswald et al. (NeurIPS 2023)
  [arXiv:2212.07677](https://arxiv.org/abs/2212.07677)
- "What Learning Algorithm is In-Context Learning? Investigations with Linear Models"
  — Akyürek et al. (ICLR 2023)
  [arXiv:2211.15661](https://arxiv.org/abs/2211.15661)

**The key insight:**

A *single-layer linear attention* transformer can exactly implement one step
of gradient descent on a least-squares objective.

Given a context `{(x₁, y₁), …, (xₙ, yₙ)}` and query `x*`:

```
GD step from w=0:
    w₁  = (lr/n) Σᵢ xᵢ yᵢ
    ŷ*  = x* · w₁  =  (lr/n) Σᵢ yᵢ (x* · xᵢ)

Linear attention:
    Attn(z_query) = (1/n) Σᵢ yᵢ · (W_Q x*)ᵀ (W_K xᵢ)  ← same!
```

With the right choice of `W_K = W_Q = I`, linear attention *is* one step
of gradient descent.

**The 13-parameter construction:**

For 1-dimensional inputs (x, y ∈ ℝ), represent each pair as a 2D token
`z = [x, y]`. The minimal linear-attention transformer that provably implements
in-context GD has these weight matrices:

```
W_K = W_Q = I₂       (identity — 0 free parameters given the structure)
W_V = [[0, 0],        (read y-slot, write to x-slot)
       [-1, 0]]
W_O = [[0, -1],       (read from x-slot, write to y-slot)
       [0,  0]]
η   = step size       (1 free parameter)
```

Counting the free scalars in the matrix entries + η gives **13 parameters**
— a remarkably small model that can do in-context linear regression.

**What the code does:**

```python
from mineo.experiments.icl_gd import (
    ICLGradientDescentExperiment,
    LinearICLModel,
    one_step_gd_predict,
    ols_predict,
)

exp = ICLGradientDescentExperiment(input_dim=1, n_shots=16)
exp.run()
# Prints: trained ≈ 1-step GD ≈ theoretical construction  (all close in MSE)
#         OLS is better with many shots (optimal for linear regression)

# Inspect the theoretical model directly
model = LinearICLModel(input_dim=1)
model.set_theoretical_weights()   # hardcode the Oswald et al. construction
```

---

## Architecture

```
mineo/
├── model/
│   └── transformer.py       # clean GPT: attention, FFN, Block, GPT
├── experiments/
│   ├── attention_sinks.py   # Xiao et al. 2023
│   ├── grokking.py          # Power et al. 2022
│   └── icl_gd.py            # Oswald et al. 2023 (13-parameter model)
└── visualization/
    └── plots.py             # attention heatmaps, grokking curves, …

notebooks/                   # interactive educational notebooks
├── 01_attention_sinks.ipynb
├── 02_grokking.ipynb
└── 03_icl_gradient_descent.ipynb

examples/                    # standalone scripts
├── run_attention_sinks.py
├── run_grokking.py
└── run_icl_gd.py

scripts/
└── generate_notebooks.py    # regenerates notebooks from Python source
```

The core transformer (`mineo/model/transformer.py`) is intentionally minimal
and self-contained — about 200 lines. Read it first.

---

## Design philosophy

- **Clarity over performance.** No fused kernels, no complex abstractions.
- **Self-contained.** No giant datasets to download; synthetic data is generated in code.
- **Config-driven.** All hyper-parameters in one dataclass — easy to modify.
- **Educational docstrings.** Every class explains the math it's implementing.
- **Reproducible.** Fixed seeds; outputs are deterministic.

---

## References

1. Vaswani et al. (2017). *Attention Is All You Need.* NeurIPS.
2. Radford et al. (2019). *Language Models are Unsupervised Multitask Learners.* (GPT-2)
3. Power et al. (2022). *Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets.* arXiv:2201.02177
4. Akyürek et al. (2023). *What Learning Algorithm is In-Context Learning?* ICLR 2023. arXiv:2211.15661
5. Oswald et al. (2023). *Transformers Learn In-Context Learning by Gradient Descent.* NeurIPS 2023. arXiv:2212.07677
6. Xiao et al. (2023). *Efficient Streaming Language Models with Attention Sinks.* arXiv:2309.17453
7. Nanda et al. (2023). *Progress measures for grokking via mechanistic interpretability.* ICLR 2023. arXiv:2301.05217
