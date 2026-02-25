"""
mineo — minimal educational LLM research toolkit.

Reproduce key results from the literature:
  - Attention Sinks (Xiao et al. 2023)
  - Grokking (Power et al. 2022)
  - In-Context Learning as Gradient Descent (Oswald et al. 2023)
"""

from mineo.model.transformer import GPT, TransformerConfig

__all__ = ["GPT", "TransformerConfig"]
