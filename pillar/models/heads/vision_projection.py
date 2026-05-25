"""Vision-token → LLM hidden-dim projection adapter.

One instance per modality. Paper-alignment note:

* PETRG-3D §4.2 text: "projected into the LLM's hidden space via linear
  transformations" -- ambiguous; could mean single Linear or MLP.
* PETRG-3D released code: uses a single ``nn.Linear`` per modality.
* LLaVA / BLIP-2 convention: 2-layer MLP with GELU.

We default to ``depth=2`` (LLaVA-style MLP) because in our setting the
LM has to bind PET-uptake numerics + CT anatomy to its embedding space
and the extra capacity helps. Set ``depth=1`` for a paper-exact
ablation (single Linear, matches the released PETRG-3D code).
"""

from __future__ import annotations

import torch
from torch import nn


class VisionProjection(nn.Module):
    """Linear or 2-layer-MLP adapter from perceiver dim to LLM hidden dim.

    Parameters
    ----------
    in_dim:
        Resampler output channels (1152 for Pillar0 Atlas).
    out_dim:
        LLM hidden_size (e.g. 2560 for MedGemma 1.5 4B).
    hidden_dim:
        Intermediate dim when ``depth=2``. Defaults to ``out_dim``.
    depth:
        ``1`` -> single ``nn.Linear(in_dim, out_dim)`` (PETRG-3D
        release-exact). ``2`` -> ``Linear(in,hid) -> GELU -> Linear(hid,out)``
        (LLaVA convention; our default).
    dropout:
        Applied between the two linear layers when ``depth=2``;
        ignored when ``depth=1``.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int | None = None,
        depth: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if depth not in (1, 2):
            raise ValueError(f"depth must be 1 or 2; got {depth}")
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.depth = depth
        self.hidden_dim = hidden_dim if hidden_dim is not None else out_dim

        if depth == 1:
            self.proj = nn.Linear(in_dim, out_dim)
        else:
            self.fc1 = nn.Linear(in_dim, self.hidden_dim)
            self.act = nn.GELU()
            self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            self.fc2 = nn.Linear(self.hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, N, in_dim) -> (B, N, out_dim)``."""
        if self.depth == 1:
            return self.proj(x)
        return self.fc2(self.drop(self.act(self.fc1(x))))

    def extra_repr(self) -> str:
        if self.depth == 1:
            return f"in_dim={self.in_dim}, out_dim={self.out_dim}, depth=1"
        return (
            f"in_dim={self.in_dim}, hidden_dim={self.hidden_dim}, "
            f"out_dim={self.out_dim}, depth=2"
        )
