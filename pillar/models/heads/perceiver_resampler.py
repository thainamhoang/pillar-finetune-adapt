"""Per-modality Perceiver-style resampler.

Compresses a long vision-token sequence (e.g. 16384 tokens from one
Pillar0-ChestCT encoder) down to a small fixed number of latent queries
(default 128) that can be spliced into an LLM prompt as visual tokens.

Architecturally PETRG-aligned but NOT release-identical — see "Paper
alignment" note below.

Layout::

    queries  ─→  [cross-attn → self-attn → FFN] x N layers  ─→ latents
    (B, Q, D)        ↑ K,V from vision tokens                (B, Q, D)

One instance per modality (CT and PET independently). No positional or
modality-type embeddings here — the LLM gets modality identity via
prompt-level ``<ct>`` / ``<pet>`` special tokens, not via embedding
arithmetic in the resampler.

Paper alignment
~~~~~~~~~~~~~~~

Our per-layer block is **separate** cross-attn → self-attn → FFN (a
standard transformer-decoder layout). The PETRG-3D *released* code uses
the **Flamingo-style** Perceiver Resampler from Alayrac et al. 2022,
where each layer concatenates the latents to the vision sequence and
performs ONE attention pass over ``[vision || latents]`` (with the
queries being the latents), then FFN — no separate self-attention block.

The paper text (§4.1) just calls it "Perceiver Sampler" without naming
the layer style, so both are defensible against the prose. Shape and
intent (compress N→Q with cross-modal information flow into queries)
are identical; block internals differ. Our form trains and converges
fine empirically and is marginally easier to LoRA-fy if we ever
expand the trainable set beyond the LM.

Implementation notes:

- Hand-rolled (no external Q-Former / Perceiver-IO lib) so the dep graph
  stays light.
- Pre-norm transformer style. GELU FFN with ``ffn_mult * dim`` hidden.
- Cross-attn K/V come from the *flattened* encoder activation: input is
  expected as ``(B, C, H, W, D)`` (the Pillar0 Atlas output shape) OR
  ``(B, N, C)`` (already flattened). Both are handled.
- Queries are learnable parameters trunc-normal-init'd. Layer weights use
  the default ``nn.MultiheadAttention`` / ``nn.Linear`` inits.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn


class PerceiverResamplerLayer(nn.Module):
    """One layer = cross-attn(queries → vision) + self-attn(queries) + FFN."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        ffn_mult: int = 4,
        attn_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                f"dim={dim} must be divisible by num_heads={num_heads}"
            )

        self.norm_cross_q = nn.LayerNorm(dim)
        self.norm_cross_kv = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )

        self.norm_self = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )

        self.norm_ffn = nn.LayerNorm(dim)
        hidden = ffn_mult * dim
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(ffn_dropout),
        )

    def forward(
        self,
        queries: torch.Tensor,        # (B, Q, D)
        vision_tokens: torch.Tensor,  # (B, N, D)
    ) -> torch.Tensor:
        # Cross-attention: queries attend to vision tokens.
        q_in = self.norm_cross_q(queries)
        kv_in = self.norm_cross_kv(vision_tokens)
        attn_out, _ = self.cross_attn(query=q_in, key=kv_in, value=kv_in, need_weights=False)
        queries = queries + attn_out

        # Self-attention among the (now vision-informed) queries.
        q_in = self.norm_self(queries)
        attn_out, _ = self.self_attn(query=q_in, key=q_in, value=q_in, need_weights=False)
        queries = queries + attn_out

        # FFN.
        queries = queries + self.ffn(self.norm_ffn(queries))
        return queries


class PerceiverResampler(nn.Module):
    """Compress a vision-token sequence to ``num_queries`` latents.

    Parameters
    ----------
    dim:
        Channel dim of vision tokens AND queries (1152 for Pillar0 Atlas-small).
    num_queries:
        Number of learnable latent queries that comprise the output. Default 128
        matches PETRG-3D.
    depth:
        Number of (cross-attn + self-attn + FFN) layers. Paper used 6.
    num_heads:
        Attention heads. 8 by default (dim/8 = 144 per head for 1152).
    ffn_mult:
        FFN hidden = ``ffn_mult * dim``.
    """

    def __init__(
        self,
        dim: int = 1152,
        num_queries: int = 128,
        depth: int = 6,
        num_heads: int = 8,
        ffn_mult: int = 4,
        attn_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_queries = num_queries
        self.depth = depth

        # Learnable latent queries (B-broadcast at forward time).
        self.queries = nn.Parameter(torch.zeros(1, num_queries, dim))
        nn.init.trunc_normal_(self.queries, std=0.02)

        self.layers = nn.ModuleList(
            [
                PerceiverResamplerLayer(
                    dim=dim,
                    num_heads=num_heads,
                    ffn_mult=ffn_mult,
                    attn_dropout=attn_dropout,
                    ffn_dropout=ffn_dropout,
                )
                for _ in range(depth)
            ]
        )

        self.norm_out = nn.LayerNorm(dim)

    @staticmethod
    def _flatten_if_5d(vision_tokens: torch.Tensor) -> torch.Tensor:
        """Accept either ``(B, C, H, W, D)`` (Pillar0 Atlas output) or ``(B, N, C)``.

        Returns ``(B, N, C)`` in channels-last sequence form.
        """
        if vision_tokens.ndim == 5:
            # (B, C, H, W, D) -> (B, H*W*D, C)
            return vision_tokens.flatten(start_dim=2).transpose(1, 2).contiguous()
        if vision_tokens.ndim == 3:
            return vision_tokens
        raise ValueError(
            f"Expected vision_tokens of rank 3 (B,N,C) or 5 (B,C,H,W,D); "
            f"got {tuple(vision_tokens.shape)}"
        )

    def forward(self, vision_tokens: torch.Tensor) -> torch.Tensor:
        """``(B, *vision_shape) -> (B, num_queries, dim)``."""
        vision_tokens = self._flatten_if_5d(vision_tokens)
        if vision_tokens.shape[-1] != self.dim:
            raise ValueError(
                f"vision_tokens last-dim {vision_tokens.shape[-1]} != "
                f"resampler dim {self.dim}"
            )

        B = vision_tokens.shape[0]
        queries = self.queries.expand(B, -1, -1).contiguous()
        for layer in self.layers:
            queries = layer(queries, vision_tokens)
        return self.norm_out(queries)

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, num_queries={self.num_queries}, depth={self.depth}"
        )
