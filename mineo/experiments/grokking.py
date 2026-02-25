"""
Grokking — reproducing Power et al. 2022.

Paper
-----
"Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets"
Alethea Power, Yuri Burda, Harri Edwards, Igor Babuschkin, Vedant Misra
https://arxiv.org/abs/2201.02177

The phenomenon
--------------
Train a small transformer on modular arithmetic:

    (a + b) mod p = c       (p prime, e.g. p = 97)

With ~50% of pairs for training, the model first *memorises* the training set
(train loss → 0, val loss stays high), and then — thousands of steps later —
*suddenly generalises* (val loss also → 0).

This "grokking" transition is:
* Delayed far past the point of memorisation
* Sharp (not gradual)
* Dependent on weight decay (crucial hyper-parameter)
* Absent without sufficient regularisation

Why grokking is interesting for AI research
--------------------------------------------
* It shows that "loss ≈ 0" does not mean "done learning".
* Generalisation can require much longer training than memorisation.
* Weight decay slowly shrinks low-generalisation solutions until the model is
  forced onto a more compact, generalisable representation.

What the generalised solution looks like (mechanistic interpretation)
---------------------------------------------------------------------
After grokking, the model implements modular addition via Fourier features:
it learns to represent integers as points on a circle (embedding ≈ [cos(2πk/p),
sin(2πk/p)] for several frequency bands k) and uses attention + MLP to compute
the angle sum.  This was discovered by Nanda et al. 2023.

Dataset
-------
Input:  [a, op_token, b, eq_token]   → predict c
We use a single-token vocabulary {0, …, p-1} ∪ {'+', '='}.
The model sees the full input and must predict the answer at the '=' position.

Usage
-----
    from mineo.experiments.grokking import GrokkingExperiment
    exp = GrokkingExperiment(p=97, train_frac=0.5)
    history = exp.train(n_steps=50_000)
    # history['train_acc'] and history['val_acc'] show the grokking curve
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
# Dataset
# ---------------------------------------------------------------------------

def _build_modular_dataset(p: int, op: str = "+") -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build the full (a OP b) mod p dataset.

    Returns
    -------
    inputs  : (p*p, 4)  — [a, op_token, b, eq_token]
    targets : (p*p,)    — answer c = (a OP b) mod p
    """
    # Vocab: 0..p-1 are numbers; p = '+'; p+1 = '='
    OP_TOKEN = p
    EQ_TOKEN = p + 1

    rows_x, rows_y = [], []
    for a in range(p):
        for b in range(p):
            if op == "+":
                c = (a + b) % p
            elif op == "*":
                c = (a * b) % p
            elif op == "-":
                c = (a - b) % p
            else:
                raise ValueError(f"Unknown op: {op}")
            rows_x.append([a, OP_TOKEN, b, EQ_TOKEN])
            rows_y.append(c)

    return (
        torch.tensor(rows_x, dtype=torch.long),
        torch.tensor(rows_y, dtype=torch.long),
    )


class ModularDataset:
    """Train/val split of a modular arithmetic task."""

    VOCAB_OFFSET = 2  # '+' and '=' tokens beyond the numeric tokens

    def __init__(self, p: int, train_frac: float = 0.5, op: str = "+", seed: int = 42):
        self.p          = p
        self.vocab_size = p + self.VOCAB_OFFSET   # {0..p-1, +, =}
        self.seq_len    = 4                        # a, +, b, =

        X, Y = _build_modular_dataset(p, op)

        # Shuffle deterministically then split
        g    = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(X), generator=g)
        X, Y = X[perm], Y[perm]

        n_train     = int(train_frac * len(X))
        self.X_tr   = X[:n_train]
        self.Y_tr   = Y[:n_train]
        self.X_val  = X[n_train:]
        self.Y_val  = Y[n_train:]

    def get_batch(
        self, split: str, batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        X = self.X_tr if split == "train" else self.X_val
        Y = self.Y_tr if split == "train" else self.Y_val
        ix = torch.randint(len(X), (batch_size,))
        return X[ix].to(device), Y[ix].to(device)


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

class GrokkingExperiment:
    """
    Full grokking demonstration.

    Parameters
    ----------
    p           : prime modulus (97 is the canonical choice from the paper)
    op          : one of '+', '*', '-'
    train_frac  : fraction of all p² pairs used for training
    d_model     : transformer hidden dimension
    n_layers    : transformer depth
    n_heads     : number of attention heads
    weight_decay: L2 regularisation — *critical* for grokking to occur
    lr          : learning rate
    n_steps     : total training steps
    batch_size  : mini-batch size
    device      : "cpu" / "cuda" / "mps"
    """

    def __init__(
        self,
        p:            int   = 97,
        op:           str   = "+",
        train_frac:   float = 0.5,
        d_model:      int   = 128,
        n_layers:     int   = 2,
        n_heads:      int   = 4,
        weight_decay: float = 1.0,    # strong weight decay is crucial
        lr:           float = 1e-3,
        n_steps:      int   = 50_000,
        batch_size:   int   = 512,
        device:       Optional[str] = None,
        verbose:      bool  = True,
    ):
        self.n_steps    = n_steps
        self.batch_size = batch_size
        self.verbose    = verbose
        self.device     = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.dataset = ModularDataset(p=p, train_frac=train_frac, op=op)

        cfg = TransformerConfig(
            vocab_size  = self.dataset.vocab_size,
            max_seq_len = self.dataset.seq_len,
            d_model     = d_model,
            n_heads     = n_heads,
            n_layers    = n_layers,
            d_ff        = d_model * 4,
            dropout     = 0.0,   # no dropout — grokking paper used none
        )
        self.model = GPT(cfg).to(self.device)
        self.opt   = torch.optim.AdamW(
            self.model.parameters(),
            lr           = lr,
            weight_decay = weight_decay,
            betas        = (0.9, 0.98),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _evaluate(self, split: str, n_batches: int = 8) -> Tuple[float, float]:
        """Return (loss, accuracy) on a split."""
        self.model.eval()
        total_loss, total_acc, total_n = 0.0, 0.0, 0
        for _ in range(n_batches):
            x, y = self.dataset.get_batch(split, self.batch_size, self.device)
            logits, _, _ = self.model(x)
            # predict at the last (EQ) position
            last_logits = logits[:, -1, :]          # (B, vocab)
            loss   = F.cross_entropy(last_logits, y)
            preds  = last_logits.argmax(dim=-1)
            acc    = (preds == y).float().mean()
            total_loss += loss.item()
            total_acc  += acc.item()
            total_n    += 1
        self.model.train()
        return total_loss / total_n, total_acc / total_n

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, log_interval: int = 1_000) -> Dict[str, List]:
        """
        Train and return history dict with keys:
            steps, train_loss, val_loss, train_acc, val_acc

        Grokking signature to look for
        --------------------------------
        Around step 1k–5k: train_acc → 1.0, val_acc stays near random (~1/p).
        Around step 10k–50k: val_acc suddenly jumps to 1.0.
        The gap between memorisation and generalisation is the "grokking delay".
        """
        history: Dict[str, List] = {
            "steps":      [],
            "train_loss": [],
            "val_loss":   [],
            "train_acc":  [],
            "val_acc":    [],
        }

        if self.verbose:
            print("=== Grokking Experiment ===")
            print(f"  model params   : {self.model.n_params:,}")
            print(f"  training steps : {self.n_steps:,}")
            print(f"  dataset (train): {len(self.dataset.X_tr)} pairs")
            print(f"  dataset (val)  : {len(self.dataset.X_val)} pairs")
            print()

        self.model.train()
        for step in range(1, self.n_steps + 1):
            x, y = self.dataset.get_batch("train", self.batch_size, self.device)
            logits, _, _ = self.model(x)
            loss = F.cross_entropy(logits[:, -1, :], y)

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            if step % log_interval == 0:
                tr_loss, tr_acc = self._evaluate("train")
                va_loss, va_acc = self._evaluate("val")
                history["steps"].append(step)
                history["train_loss"].append(tr_loss)
                history["val_loss"].append(va_loss)
                history["train_acc"].append(tr_acc)
                history["val_acc"].append(va_acc)

                if self.verbose:
                    print(
                        f"  step {step:6d} │ "
                        f"train: loss={tr_loss:.4f} acc={tr_acc:.3f}  │  "
                        f"val: loss={va_loss:.4f} acc={va_acc:.3f}"
                    )

                # Early-stop once generalisation is achieved
                if va_acc > 0.99:
                    if self.verbose:
                        print(f"\n  ✓ Grokking achieved at step {step}!")
                    break

        return history

    # ------------------------------------------------------------------
    # Analysis: Fourier features
    # ------------------------------------------------------------------

    @torch.no_grad()
    def fourier_analysis(self, n_freqs: int = 10) -> Dict[str, object]:
        """
        Analyse whether the token embeddings use Fourier (circular) structure.

        After grokking, the model encodes integer k as approximately
            [cos(2π·f·k/p), sin(2π·f·k/p)]
        for one or a few dominant frequencies f.

        We compute the DFT of each embedding dimension over the numeric tokens
        {0, …, p-1} and return the power spectrum.

        Returns
        -------
        power_spectrum : (d_model, p//2) tensor of Fourier power per frequency
        top_freqs      : list of the dominant frequencies
        """
        p         = self.dataset.p
        emb       = self.model.tok_emb.weight[:p].cpu().float()  # (p, d_model)
        emb_T     = emb.T                                         # (d_model, p)

        fft       = torch.fft.rfft(emb_T, dim=-1)                # (d_model, p//2+1)
        power     = fft.abs() ** 2                                # Fourier power

        # Top frequencies (by max power across embedding dims)
        freq_power = power[:, 1:].sum(dim=0)                      # ignore DC
        top_freq_idx = freq_power.topk(n_freqs).indices + 1       # +1 to skip DC

        return {
            "power_spectrum": power,
            "top_freqs":      top_freq_idx.tolist(),
        }
