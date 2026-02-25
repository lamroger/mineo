"""
Attention Sinks — reproducing Xiao et al. 2023.

Paper
-----
"Efficient Streaming Language Models with Attention Sinks"
Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, Mike Lewis
https://arxiv.org/abs/2309.17453

The key finding
---------------
When you look at attention weight matrices in large (and small) trained
language models, the first few tokens — especially the very first token —
accumulate a disproportionately large fraction of the total attention mass.
This happens *regardless* of whether those tokens are semantically important.

Why does this matter?
---------------------
Streaming / long-context inference often uses a sliding-window KV cache
to stay within memory.  If you naively evict old tokens, you lose the
"sink" tokens that the model has learned to use as a kind of attention
overflow valve.  Removing them causes a sharp spike in perplexity.

The fix: always keep a handful of initial "sink" tokens in the cache,
even if they are semantically stale.

What this experiment does
-------------------------
1. Train a small GPT on synthetic text (repeated character bigrams so the
   model learns real patterns quickly).
2. After training, run inference and collect attention weights.
3. Compute the mean attention mass on position-0 across all heads and layers.
4. Show that position-0 is an outlier — an attention sink.
5. Simulate the streaming-cache failure: compare perplexity with and without
   the sink token in the KV cache.

Usage
-----
    from mineo.experiments.attention_sinks import AttentionSinkExperiment
    exp = AttentionSinkExperiment()
    exp.train()
    stats = exp.measure_sink_scores()
    exp.simulate_streaming_cache()
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from mineo.model.transformer import GPT, TransformerConfig


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------

def make_text_corpus(n_chars: int = 50_000) -> str:
    """
    Generate a synthetic character-level corpus.

    We use a simple Markov-like structure so the model has real patterns to
    learn (not pure noise), but the corpus is self-contained — no file I/O.
    """
    rng    = random.Random(42)
    # Bigram transition table: after char c, next char follows a distribution
    alphabet = "abcdefghijklmnopqrstuvwxyz .,\n"
    # Simple trigram-ish structure via a few "topic" patterns
    topics = [
        "the quick brown fox jumps over the lazy dog. ",
        "attention is all you need for language models. ",
        "transformers learn in context by gradient descent. ",
        "the first token is an attention sink in every layer. ",
        "grokking happens when generalisation suddenly appears. ",
    ]
    buf = []
    topic = rng.choice(topics)
    i     = 0
    while len(buf) < n_chars:
        buf.append(topic[i % len(topic)])
        i += 1
        if rng.random() < 0.1:
            topic = rng.choice(topics)
            i     = 0
    return "".join(buf)


class CharDataset:
    """Character-level dataset. Tokenises by ord() → integer."""

    def __init__(self, text: str, seq_len: int):
        self.seq_len = seq_len
        # Build vocab from characters present in text
        chars      = sorted(set(text))
        self.stoi  = {c: i for i, c in enumerate(chars)}
        self.itos  = {i: c for c, i in self.stoi.items()}
        self.data  = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def get_batch(
        self, batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ix = torch.randint(len(self.data) - self.seq_len - 1, (batch_size,))
        x  = torch.stack([self.data[i     : i + self.seq_len    ] for i in ix])
        y  = torch.stack([self.data[i + 1 : i + self.seq_len + 1] for i in ix])
        return x.to(device), y.to(device)


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

class AttentionSinkExperiment:
    """
    End-to-end attention-sink demonstration.

    Parameters
    ----------
    d_model   : embedding dimension
    n_layers  : number of transformer layers
    n_heads   : number of attention heads
    seq_len   : training sequence length
    n_steps   : training steps
    batch_size: training batch size
    device    : "cpu" / "cuda" / "mps"
    """

    def __init__(
        self,
        d_model:    int   = 128,
        n_layers:   int   = 4,
        n_heads:    int   = 4,
        seq_len:    int   = 128,
        n_steps:    int   = 2_000,
        batch_size: int   = 32,
        lr:         float = 3e-4,
        device:     Optional[str] = None,
        verbose:    bool  = True,
    ):
        self.seq_len    = seq_len
        self.n_steps    = n_steps
        self.batch_size = batch_size
        self.verbose    = verbose
        self.device     = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        corpus = make_text_corpus()
        self.dataset = CharDataset(corpus, seq_len)

        cfg = TransformerConfig(
            vocab_size  = self.dataset.vocab_size,
            max_seq_len = seq_len,
            d_model     = d_model,
            n_heads     = n_heads,
            n_layers    = n_layers,
            d_ff        = d_model * 4,
            dropout     = 0.1,
        )
        self.model = GPT(cfg).to(self.device)
        self.opt   = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self._trained = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, log_interval: int = 200) -> List[float]:
        """
        Train the model and return per-step losses.

        The model is intentionally small so this runs in a few minutes on CPU.
        """
        self.model.train()
        losses = []
        for step in range(1, self.n_steps + 1):
            x, y = self.dataset.get_batch(self.batch_size, self.device)
            _, loss, _ = self.model(x, targets=y)
            self.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step()
            losses.append(loss.item())
            if self.verbose and step % log_interval == 0:
                print(f"  step {step:5d}/{self.n_steps}  loss={loss.item():.4f}")
        self._trained = True
        return losses

    # ------------------------------------------------------------------
    # Sink measurement
    # ------------------------------------------------------------------

    @torch.no_grad()
    def measure_sink_scores(
        self, n_samples: int = 32
    ) -> Dict[str, object]:
        """
        Collect attention weights over `n_samples` random sequences.

        Returns
        -------
        dict with keys:
            'mean_attn_by_pos'  : (T,) tensor — mean attention mass per position
                                   averaged over all heads, layers, samples
            'sink_ratio'        : scalar — how much more attention pos-0 gets
                                   vs. the average non-sink position
            'per_layer'         : (n_layers, T) — broken down by layer
        """
        self.model.eval()
        T       = self.seq_len
        n_lay   = self.model.config.n_layers
        n_heads = self.model.config.n_heads

        # Accumulate attention column-sums over samples
        # Shape: (n_layers, T)
        layer_col_sum = torch.zeros(n_lay, T)

        for _ in range(n_samples):
            x, _ = self.dataset.get_batch(1, self.device)
            _, _, attn_list = self.model(x, return_attn=True)

            for layer_idx, attn in enumerate(attn_list):
                # attn: (1, n_heads, T, T)
                # For each query position q, attn[q, k] = weight from q to k.
                # We want "how much does position k receive" = column sum.
                col_sum = attn[0].mean(dim=0).sum(dim=0)   # (T,)  average over heads then sum over queries
                layer_col_sum[layer_idx] += col_sum.cpu()

        layer_col_sum /= n_samples                 # average over samples
        mean_by_pos    = layer_col_sum.mean(dim=0) # average over layers

        # Sink ratio: attention to pos-0 vs uniform expectation
        uniform = mean_by_pos.mean().item()
        sink_ratio = mean_by_pos[0].item() / (uniform + 1e-8)

        return {
            "mean_attn_by_pos": mean_by_pos,
            "sink_ratio":       sink_ratio,
            "per_layer":        layer_col_sum,
        }

    # ------------------------------------------------------------------
    # Streaming cache simulation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def simulate_streaming_cache(
        self, cache_size: int = 32, n_eval_tokens: int = 256
    ) -> Dict[str, float]:
        """
        Compare perplexity of three KV-cache streaming strategies.

        window_only
            Keep only the last `cache_size` tokens.  Evicts the sink token
            once it falls outside the window.

        sink_plus_window
            Always keep token 0 (the sink) plus the last `cache_size - 1`
            recent tokens.  This is the fix proposed by Xiao et al.

        full_context
            No eviction — upper-bound baseline (requires O(T) memory).

        Returns perplexity for each strategy.
        """
        self.model.eval()
        corpus = make_text_corpus(n_chars=n_eval_tokens + self.seq_len + 10)
        dataset = CharDataset(corpus, self.seq_len)
        data    = dataset.data.to(self.device)

        def compute_ppl(strategy: str) -> float:
            total_nll = 0.0
            count     = 0
            ctx_len   = self.model.config.max_seq_len

            for start in range(self.seq_len, len(data) - 1, 8):
                if count >= n_eval_tokens:
                    break

                if strategy == "full_context":
                    ctx = data[max(0, start - ctx_len) : start]
                elif strategy == "window_only":
                    ctx = data[max(0, start - cache_size) : start]
                else:  # sink_plus_window
                    sink    = data[self.seq_len : self.seq_len + 1]   # token at pos 0
                    recent  = data[max(0, start - (cache_size - 1)) : start]
                    ctx     = torch.cat([sink, recent])

                if len(ctx) == 0:
                    continue

                ctx_batch = ctx.unsqueeze(0)     # (1, ctx_len)
                logits, _, _ = self.model(ctx_batch)
                last_logits  = logits[0, -1, :]  # (vocab_size,)
                target       = data[start]
                nll          = F.cross_entropy(last_logits.unsqueeze(0), target.unsqueeze(0))
                total_nll   += nll.item()
                count        += 1

            return math.exp(total_nll / count) if count > 0 else float("inf")

        results = {
            "full_context":      compute_ppl("full_context"),
            "window_only":       compute_ppl("window_only"),
            "sink_plus_window":  compute_ppl("sink_plus_window"),
        }

        if self.verbose:
            print("\n--- Streaming cache comparison ---")
            for k, v in results.items():
                print(f"  {k:<20s}  perplexity = {v:.2f}")
            print(
                "\n  Prediction: sink_plus_window ≈ full_context "
                "< window_only (higher ppl = worse)"
            )

        return results

    # ------------------------------------------------------------------
    # One-shot demo (train + measure)
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, object]:
        """Train, measure sinks, simulate streaming.  Returns all results."""
        print("=== Attention Sink Experiment ===")
        print(f"  model params : {self.model.n_params:,}")
        print(f"  training steps: {self.n_steps}")
        print()
        losses = self.train()
        print(f"\n  Final training loss: {losses[-1]:.4f}")

        print("\nMeasuring attention sink scores …")
        sink_stats = self.measure_sink_scores()
        print(f"  Sink ratio (pos-0 vs average): {sink_stats['sink_ratio']:.2f}×")

        print("\nSimulating streaming KV-cache …")
        cache_stats = self.simulate_streaming_cache()

        return {"train_losses": losses, **sink_stats, "cache": cache_stats}
