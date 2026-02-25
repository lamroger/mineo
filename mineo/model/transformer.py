"""
Minimal GPT-style transformer, written for clarity.

Design principles
-----------------
* Every operation is explicit — no magic, no fused kernels.
* Attention weights can be returned for any forward pass (needed for the
  attention-sink and interpretability experiments).
* Config-driven so that tiny (research) and medium (demo) models share the
  same code path.

References
----------
* Vaswani et al. 2017  — "Attention Is All You Need"
* Radford et al. 2019  — GPT-2
* Brown et al. 2020    — GPT-3
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TransformerConfig:
    """All hyper-parameters for a GPT model in one place."""

    vocab_size: int = 256       # number of distinct tokens
    max_seq_len: int = 256      # maximum context length
    d_model: int = 128          # embedding / hidden dimension
    n_heads: int = 4            # number of attention heads
    n_layers: int = 4           # number of transformer blocks
    d_ff: int = 512             # feed-forward inner dimension (usually 4×d_model)
    dropout: float = 0.1
    bias: bool = True           # bias terms in linear layers

    @property
    def d_head(self) -> int:
        """Dimension per attention head."""
        return self.d_model // self.n_heads


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    """
    Multi-head causal (masked) self-attention.

    The "causal" mask ensures position t can only attend to positions ≤ t,
    which is required for autoregressive language modelling.

    Math recap
    ----------
    Given input X ∈ R^{T×d}:

        Q = X W_Q,  K = X W_K,  V = X W_V        (project to queries/keys/values)
        A = softmax( Q K^T / sqrt(d_head) + mask ) (attention weights)
        out = A V W_O                              (weighted sum → project out)

    The mask sets future positions to -inf before softmax, making their weight ≈ 0.

    Attention sinks
    ---------------
    A key empirical observation (Xiao et al. 2023) is that A[:, :, :, 0] —
    the column of attention weights for token 0 — is often very large, even
    when token 0 is semantically irrelevant.  This module stores the last
    computed attention weights in `self.last_attn` so experiments can
    inspect them without re-running the forward pass.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0, \
            "d_model must be divisible by n_heads"

        self.n_heads = config.n_heads
        self.d_head  = config.d_head
        self.d_model = config.d_model

        # Single fused projection for Q, K, V (3× the width)
        self.qkv_proj  = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        self.out_proj   = nn.Linear(config.d_model, config.d_model,     bias=config.bias)
        self.attn_drop  = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)

        # Lower-triangular causal mask, registered as a buffer so it moves
        # to the right device automatically with .to(device).
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.max_seq_len, config.max_seq_len, dtype=torch.bool))
            .view(1, 1, config.max_seq_len, config.max_seq_len),
        )

        self.last_attn: Optional[torch.Tensor] = None  # shape (B, H, T, T)

    def forward(
        self,
        x: torch.Tensor,               # (B, T, d_model)
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T, C = x.shape

        # ---- project -------------------------------------------------------
        qkv = self.qkv_proj(x)                      # (B, T, 3·d_model)
        Q, K, V = qkv.split(self.d_model, dim=-1)   # each (B, T, d_model)

        # reshape to (B, n_heads, T, d_head)
        def split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        Q, K, V = split_heads(Q), split_heads(K), split_heads(V)

        # ---- scaled dot-product attention ----------------------------------
        scale  = 1.0 / math.sqrt(self.d_head)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale   # (B, H, T, T)

        # mask future positions
        scores = scores.masked_fill(~self.causal_mask[:, :, :T, :T], float("-inf"))

        attn = F.softmax(scores, dim=-1)   # (B, H, T, T)

        # store for external inspection (attention sinks analysis)
        self.last_attn = attn.detach()

        attn_dropped = self.attn_drop(attn)
        out = torch.matmul(attn_dropped, V)              # (B, H, T, d_head)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_drop(self.out_proj(out))

        return out, (attn.detach() if return_attn else None)


# ---------------------------------------------------------------------------
# Feed-forward network
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """
    Position-wise feed-forward block.

    Applied identically at every position:
        FFN(x) = dropout( W2 · GELU( W1 · x ) )

    The expansion ratio d_ff / d_model is typically 4.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.fc1  = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
        self.fc2  = nn.Linear(config.d_ff, config.d_model, bias=config.bias)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(F.gelu(self.fc1(x))))


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------

class Block(nn.Module):
    """
    One transformer layer = attention + feed-forward, both with pre-norm.

    Pre-norm (LayerNorm before the sub-layer) is more stable than the
    original post-norm and is used by GPT-2 onward.

        x ← x + Attention( LayerNorm(x) )
        x ← x + FFN( LayerNorm(x) )
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model)
        self.mlp  = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        attn_out, attn_weights = self.attn(self.ln_1(x), return_attn=return_attn)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, attn_weights


# ---------------------------------------------------------------------------
# Full GPT model
# ---------------------------------------------------------------------------

class GPT(nn.Module):
    """
    GPT-style autoregressive language model.

    Takes a sequence of token ids and returns a distribution over the next
    token at every position.

    Architecture summary
    --------------------
        tokens → embedding + positional embedding
               → dropout
               → N × Block (attention + FFN)
               → LayerNorm
               → linear head → logits over vocab

    Weight tying: the token embedding matrix is shared with the final linear
    projection, halving parameter count and often improving perplexity.

    Usage
    -----
        cfg   = TransformerConfig(vocab_size=100, d_model=64, n_layers=2)
        model = GPT(cfg)
        logits, loss, attn_weights = model(idx, targets=targets)
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.drop    = nn.Dropout(config.dropout)
        self.blocks  = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.ln_f    = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying
        self.tok_emb.weight = self.lm_head.weight

        self.apply(self._init_weights)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_weights(self, module: nn.Module) -> None:
        """GPT-style weight initialisation."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_params(self) -> int:
        """Total trainable parameters (embedding counted once due to weight tying)."""
        return sum(p.numel() for p in self.parameters())

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        idx: torch.Tensor,                       # (B, T)  token ids
        targets: Optional[torch.Tensor] = None,  # (B, T)  next-token ids
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], List[torch.Tensor]]:
        """
        Parameters
        ----------
        idx      : integer token ids, shape (B, T)
        targets  : optional ground-truth next tokens; if given, loss is computed
        return_attn : if True, collect attention weight tensors from every block

        Returns
        -------
        logits       : (B, T, vocab_size)
        loss         : scalar cross-entropy loss, or None
        attn_weights : list of (B, n_heads, T, T) tensors, one per layer
                       (empty list when return_attn=False)
        """
        B, T = idx.shape
        assert T <= self.config.max_seq_len, \
            f"Sequence length {T} exceeds max_seq_len {self.config.max_seq_len}"

        device = idx.device
        pos = torch.arange(T, device=device)   # [0, 1, ..., T-1]

        # Token + positional embeddings
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))

        # Transformer blocks
        all_attn: List[torch.Tensor] = []
        for block in self.blocks:
            x, attn = block(x, return_attn=return_attn)
            if return_attn and attn is not None:
                all_attn.append(attn)

        x      = self.ln_f(x)
        logits = self.lm_head(x)    # (B, T, vocab_size)

        loss: Optional[torch.Tensor] = None
        if targets is not None:
            # Flatten to (B*T, vocab) vs (B*T,) for cross-entropy
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )

        return logits, loss, all_attn

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,       # (B, T)  prompt token ids
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Autoregressively sample `max_new_tokens` tokens.

        Parameters
        ----------
        temperature : > 1 makes distribution flatter (more random),
                      < 1 makes it peakier (more greedy)
        top_k       : if set, only sample from the top-k most likely tokens
        """
        for _ in range(max_new_tokens):
            # Truncate context to max_seq_len
            idx_cond = idx[:, -self.config.max_seq_len:]
            logits, _, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # (B, vocab_size)

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs      = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx        = torch.cat([idx, next_token], dim=1)

        return idx
